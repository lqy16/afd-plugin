# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded eager work descriptors. Tensor data travels in the UMDK window."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.distributed import Store

logger = logging.getLogger(__name__)

POLL_SECONDS = 0.001
PUBLISH_TIMEOUT_SECONDS = 120
# Diagnostics only: how long a gate skip or a publish wait may last before the
# stall is reported, and how often it is re-reported while it lasts.
STALL_REPORT_SECONDS = 2.0
STALL_REPORT_INTERVAL_SECONDS = 10.0


class ExpertWorkQueue:
    """One mailbox per (Attention DP, FFN rank), each with exactly one reader."""

    def __init__(
        self,
        store: Store,
        sources: int,
        experts: int,
        wrap: Callable[[int], int],
        *,
        read_published: Callable[[], Sequence[Sequence[int]]] | None = None,
    ):
        self.store = store
        self.sources = sources
        self.experts = experts
        self.wrap = wrap
        self.last = [0] * sources
        self.cursor = 0
        self.read_published = read_published
        # Diagnostics: first skip / last report per source, last publish report.
        self.gate_waiting: dict[int, float] = {}
        self.gate_reported: dict[int, float] = {}
        self.publish_reported = 0.0

    def next_sequence(self, source: int) -> int:
        return self.wrap(self.last[source] + 1)

    def publish(self, source: int, layer: int, sequence: int) -> None:
        if sequence != self.next_sequence(source):
            raise RuntimeError("Out-of-order MegaMoE publication")
        keys = [f"work/{source}/{ffn}" for ffn in range(self.experts)]
        started = time.monotonic()
        deadline = started + PUBLISH_TIMEOUT_SECONDS
        for key in keys:
            while self.store.check([key]):
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(f"FFN did not claim previous work in {key}")
                if (
                    now - started >= STALL_REPORT_SECONDS
                    and now - self.publish_reported >= STALL_REPORT_INTERVAL_SECONDS
                ):
                    self.publish_reported = now
                    logger.warning(
                        "MegaMoE publish stalled %.1fs on %s "
                        "(source=%d layer=%d sequence=%d) published=%s",
                        now - started,
                        key,
                        source,
                        layer,
                        sequence,
                        self.read_published() if self.read_published is not None else None,
                    )
                time.sleep(POLL_SECONDS)
            self.store.set(key, f"{layer} {sequence}")
        self.last[source] = sequence

    def poll(self, ffn: int) -> tuple[int, int, int] | None:
        published = None
        for offset in range(self.sources):
            source = (self.cursor + offset) % self.sources
            key = f"work/{source}/{ffn}"
            if self.store.check([key]):
                layer, sequence = map(int, self.store.get(key).split())
                if sequence != self.next_sequence(source):
                    raise RuntimeError(f"Out-of-order MegaMoE work in {key}")
                if self.read_published is not None:
                    if published is None:
                        # Take one fresh snapshot per scan, only when work exists.
                        published = self.read_published()
                    # The producer needs TAKEN from this FFN to reuse the slot,
                    # so an unclaimed generation cannot already be superseded.
                    if any(word != sequence for word in published[source]):
                        now = time.monotonic()
                        waiting_since = self.gate_waiting.setdefault(source, now)
                        if (
                            now - waiting_since >= STALL_REPORT_SECONDS
                            and now - self.gate_reported.get(source, 0.0)
                            >= STALL_REPORT_INTERVAL_SECONDS
                        ):
                            self.gate_reported[source] = now
                            logger.warning(
                                "MegaMoE gate skipped %.1fs: source=%d ffn=%d "
                                "sequence=%d published=%s",
                                now - waiting_since,
                                source,
                                ffn,
                                sequence,
                                published[source],
                            )
                        continue
                    self.gate_waiting.pop(source, None)
                self.store.delete_key(key)
                self.last[source] = sequence
                self.cursor = (source + 1) % self.sources
                return source, layer, sequence
        return None
