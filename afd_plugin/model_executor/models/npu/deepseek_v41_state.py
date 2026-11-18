# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep V4.1 bounded replay local to each asynchronous Attention DP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.config import CUDAGraphMode
from vllm.v1.utils import CpuGpuBuffer
from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run
from vllm_ascend.models.deepseek_v41.decoder_replay_state import (
    DecoderReplayInputs,
    DecoderReplayState,
)
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.model_states.default import AscendModelState

if TYPE_CHECKING:
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata


class AFDDeepseekV41ReplayState(DecoderReplayState):
    # Upstream source: vllm-ascend f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e,
    # models/deepseek_v41/decoder_replay_state.py::DecoderReplayState.prepare.
    # Patch reason: native replay skips async DP or performs a DP all-reduce;
    # independent AFD DPs must plan their retained rows without other replicas.
    # Patch functionality: reuse native row selection/metadata/context/scatter,
    # replacing cross-DP admission with this DP's compaction decision.
    # Signature: matches upstream; no added parameters.
    def prepare(
        self,
        inputs: DecoderReplayInputs,
        mode: CUDAGraphMode,
        *,
        for_capture: bool = False,
    ) -> None:
        self.layers.rows = None
        self.layers.context_factory = None
        self.layers.num_padded_rows = None
        self.replay_pcp_context = None
        self.dp_tokens = None
        self.metadata: dict[str, AscendDSAV41Metadata] = {}
        if self.slots is None:
            self._initialize(inputs)
        if for_capture or mode != CUDAGraphMode.NONE or get_mrv2_in_profile_run():
            return
        # ### PATCH START: independent AFD replay batch
        plan = self._build_replay_plan(inputs)
        self.num_actual_tokens = (
            0 if inputs.batch.is_dummy else int(plan.kept_lens.sum())
        )
        self.num_tokens = plan.num_tokens
        if plan.requires_compaction:
            self._prepare_compact_replay(inputs, plan)
        # ### PATCH END: independent AFD replay batch


class AFDDeepseekV41ModelState(EngramModelState):
    # Upstream source: vllm-ascend f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e,
    # models/deepseek_v41/engram/model_state.py::EngramModelState.__init__.
    # Patch reason: EngramModelState hardcodes DecoderReplayState construction.
    # Patch functionality: select the local-DP replay class at construction,
    # retaining native Engram hooks and buffers without allocating two states.
    # Signature: matches upstream; no added parameters.
    def __init__(self, vllm_config, model, encoder_cache, device):
        # ### PATCH START: native parent setup and AFD replay factory
        AscendModelState.__init__(self, vllm_config, model, encoder_cache, device)
        # ### PATCH END: native parent setup and AFD replay factory
        self._replay_start_np: NDArray[np.int32] = np.zeros(
            self.max_num_reqs, dtype=np.int32
        )
        max_num_local_reqs = self.max_num_reqs
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            max_num_local_reqs *= 2
        self._replay_start = CpuGpuBuffer(
            max_num_local_reqs, dtype=torch.int32, device=device, pin_memory=False
        )
        self._global_replay_start = CpuGpuBuffer(
            self.max_num_reqs, dtype=torch.int32, device=device, pin_memory=False
        )
        self._replay_groups = None
        layers = model.decoder_replay_layers
        # ### PATCH START: independent AFD replay batch
        self.decoder_replay_state = (
            AFDDeepseekV41ReplayState(vllm_config, layers, device)
            if layers is not None
            else None
        )
        # ### PATCH END: independent AFD replay batch
        depth = model.token_lookback_depth
        self.lookback_token_ids = None
        self._cg_mode = None
        if depth > 0:
            model.register_forward_pre_hook(
                self._prepare_engram_before_forward, with_kwargs=True
            )
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )
            if model.supports_engram_graph_producer:
                self._engram_graph_inputs = {
                    "engram_query_start_loc": torch.zeros(
                        self.max_num_reqs + 1, dtype=torch.int32, device=device
                    ),
                    "engram_valid_token_count": torch.zeros(
                        1, dtype=torch.int32, device=device
                    ),
                }
