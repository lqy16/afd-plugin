# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Checkpoint ownership before the native V4.1 loader remaps parameter names."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

AFDRole = Literal["attention", "ffn"]
_BOTH_ROLES = frozenset(("attention", "ffn"))


def checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Keep native checkpoint spellings; route each layer path to its owner."""
    # Native V4.1 defers these modules outside its text backbone. FFN has no
    # Engram, so its native loader branch does not apply the deferred filter.
    # Exclude them on both roles before native remapping, keeping names intact.
    if name.removeprefix("model.").startswith(
        ("aligner.", "vision.", "image_", "mtp.")
    ):
        return frozenset()
    parts = name.split(".")
    for index, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            int(parts[index + 1])
        except ValueError:
            continue
        stage = parts[index + 2]
        tail = parts[index + 3 :]
        if stage in ("ffn", "mlp"):
            if tail and tail[0] == "gate":
                # Native FFN MoE retains its gate parameters for loader/quant
                # lifecycle, although CAM supplies already routed activations.
                return _BOTH_ROLES
            if tail and tail[0] == "shared_experts":
                return frozenset(("attention",))
            return frozenset(("ffn",))
        # Attention, Engram, layer norms and delayed-HC coefficients are local
        # to the Attention role; connector-driven FFN never runs the decoder.
        return frozenset(("attention",))
    # Preserve the native embedding/final-norm/head loader lifecycle on both
    # roles, matching the existing Ascend V4 adapter's common-weight policy.
    return _BOTH_ROLES


def iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]], *, role: AFDRole
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if role in checkpoint_weight_roles(name):
            yield name, tensor
