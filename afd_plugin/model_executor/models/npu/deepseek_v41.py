# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DSV4.1 text AFD adapter for the A5 container's f9c8e0bdb Ascend source.

Attention retains native delayed-HC, shared-KV, Engram and bounded replay.
Only the MoE boundary uses the existing A3 Async CAM protocol. Operator/SOC
and MXFP8/MXFP4 communication compatibility remain device integration work.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_ascend.models.deepseek_v41 import model as native

from afd_plugin.config import AFD_ASYNC_CONNECTOR, MEGAMOE_CONNECTOR, parse_afd_config
from afd_plugin.connectors import AFDF2ATransferPayload
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy
from afd_plugin.model_executor.models.npu.deepseek_v4 import (
    AFDDeepseekV4AttentionGateRemoteMoE,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_weights import (
    AFDRole,
    iter_role_weights,
)

if TYPE_CHECKING:
    from afd_plugin.model_executor.npu.async_cam_layered import LayeredFFNWeights


class AFDDeepseekV41AttentionGateRemoteMoE(AFDDeepseekV4AttentionGateRemoteMoE):
    """Reuse V4's CAM transaction with V4.1's native router/shared weights."""

    def __init__(
        self, *, config, layer_idx: int, prefix: str, vllm_config: VllmConfig
    ) -> None:
        RemoteFFNProxy.__init__(self, layer_idx=layer_idx)
        # The V4 proxy selects the MegaMoE transfer path from this flag. This
        # class bypasses V4's __init__, so resolve it here before the inherited
        # forward reads it.
        self.megamoe = (
            parse_afd_config(vllm_config, validate=False).connector
            == MEGAMOE_CONNECTOR
        )
        self.use_sequence_parallel_moe = False
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.num_expert_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.shared_experts = (
            native.DeepseekV41MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                quant_config=vllm_config.quant_config,
                reduce_results=False,
                is_sequence_parallel=False,
                prefix=f"{prefix}.shared_experts",
            )
            if config.n_shared_experts
            else None
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.precast_fp32_weight = True
        self.gate.bias_vl = None
        # Keep the native checkpoint schema even for text-only requests.
        # Disabling image inputs does not remove this checkpoint parameter.
        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        if layer_idx < config.num_hash_layers:
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size, config.num_experts_per_tok, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )

    # Native V4.1 adds already_sequence_parallel to the V4 MoE call contract.
    # TP1 is the first implementation slice, so native model sharding is off.
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states, input_ids=input_ids, hidden_states_fp32=hidden_states_fp32
        )


class AFDDeepseekV41DecoderLayer(native.DeepseekV41DecoderLayer):
    """Inherit native forward and its (hidden_states, pre_mix) contract."""

    # Upstream source: vllm-ascend f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e,
    # models/deepseek_v41/model.py::DeepseekV41DecoderLayer.__init__.
    # Patch reason: the native layer has no MoE factory hook and always allocates
    # Attention and routed experts. Construct only the selected AFD role.
    # Patch functionality: replace the Attention MoE with a CAM proxy; FFN builds
    # native routed MoE without Attention, Engram, shared experts or HC weights.
    # Signature: matches upstream; no added parameters.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: role-local construction
        nn.Module.__init__(self)
        afd = parse_afd_config(vllm_config, validate=False)
        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
        self.vllm_config = vllm_config
        self.config = config
        self.afd_role = afd.role
        self.layer_idx = int(prefix.split(".")[-1])
        self.compute_gate_on_attention = True
        self.is_moe_layer = True
        self.engram = None
        if afd.role == "ffn":
            self.self_attn = native.PPMissingLayer()
            moe_config = copy(config)
            # The native constructor omits shared experts only for None.
            object.__setattr__(moe_config, "n_shared_experts", None)
            # Match the V4 adapter: Ascend replaces the factory during platform
            # setup, after the native model imported its own factory binding.
            native.FusedMoEFactory = fused_moe.FusedMoEFactory
            self.mlp = native.DeepseekV41MoE(
                config=moe_config,
                parallel_config=vllm_config.parallel_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=is_draft_layer,
            )
            return
        # ### PATCH END: role-local construction

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.self_attn = self.attention_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
        )
        # ### PATCH START: remote routed MoE
        self.mlp = AFDDeepseekV41AttentionGateRemoteMoE(
            config=config,
            layer_idx=self.layer_idx,
            prefix=f"{prefix}.mlp",
            vllm_config=vllm_config,
        )
        # ### PATCH END: remote routed MoE
        self.input_layernorm = native.RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=self.norm_eps
        )
        self.routed_scaling_factor = config.routed_scaling_factor
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.use_sequence_parallel = parallel_config.use_sequence_parallel_moe
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        if (
            native._engram_enabled_for_runtime(config, vllm_config)
            and not is_draft_layer
            and self.layer_idx in config.engram_layer_ids
        ):
            self.engram = native.AscendEngram(config, quant_config, f"{prefix}.engram")

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        group_list: torch.Tensor,
        dynamic_scales: torch.Tensor | None = None,
        group_list_type: int = 1,
        **_: Any,
    ) -> AFDF2ATransferPayload:
        from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
            compute_attention_gate_moe_ffn,
        )

        return compute_attention_gate_moe_ffn(
            self,
            hidden_states=hidden_states,
            group_list=group_list,
            dynamic_scales=dynamic_scales,
            expand_x_shared=None,
            dynamic_scales_shared=None,
            topk_scales=None,
            group_list_type=group_list_type,
            routed_scale_applied_in_topk=True,
        )


class AFDDeepseekV41Model(native.DeepseekV41Model):
    """Use the native Attention backbone and a separate lightweight FFN shell."""

    decoder_layer_cls = AFDDeepseekV41DecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd.role
        if afd.role == "attention":
            # Native construction binds shared KV/index state, Engram tables,
            # OProj pools and bounded replay to our injected decoder layers.
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return
        # No native Attention constructor or temporary full-model allocation.
        nn.Module.__init__(self)
        config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        self.config = config
        self.device = native.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.has_engram = False
        self.engram_hash = None
        self.use_sequence_parallel_moe = False
        self.use_sequence_parallel = False
        self.topk_indices_buffer = None
        self._mtp_hidden_buffer = None
        self.decoder_replay_layers = None
        self.decoder_replay_start = config.num_hidden_layers
        self.hc_mult = config.hc_mult
        self.embed_tokens = native.VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("engram_rotation", torch.eye(32), persistent=False)
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors

    def compute_ffn_output(
        self, hidden_states: torch.Tensor, layer_idx: int, **kwargs: Any
    ) -> AFDF2ATransferPayload:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer.layer_idx
            for layer in self.layers
            if isinstance(layer.mlp, native.DeepseekV41MoE)
        )


class AFDDeepseekV41ForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    """Keep native V4.1 loading, sampling and Attention-side model state."""

    model_cls = AFDDeepseekV41Model
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd = parse_afd_config(vllm_config, validate=False)
        parallel = vllm_config.parallel_config
        if (
            afd.connector not in (AFD_ASYNC_CONNECTOR, MEGAMOE_CONNECTOR)
            or not afd.compute_gate_on_attention
        ):
            raise ValueError(
                "DSV4.1 uses CAMAsyncAFDConnector (or its MegaMoE variant) with "
                "Attention-side routing"
            )
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_sequence_parallel_moe
            or parallel.use_ubatching
        ):
            raise ValueError(
                "DSV4.1 initial AFD path uses TP1, PP1, PCP1, DCP1 and no DBO"
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "DSV4.1 initial AFD path uses ordinary generation without a draft model"
            )
        if not vllm_config.model_config.enforce_eager:
            raise ValueError(
                "DSV4.1 initial AFD path uses eager execution on both roles"
            )
        config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        if (
            afd.role == "attention"
            and native._engram_enabled_for_runtime(config, vllm_config)
            and native.get_engram_dp_size() > 1
            and not vllm_config.engram_config.dp_shared_memory
        ):
            raise ValueError(
                "Async DSV4.1 Attention requires Engram dp_shared_memory or EDP1"
            )
        self.afd_config = afd
        self.afd_role = cast(AFDRole, afd.role)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # One-pass filtering keeps exact names/tensors and all native packed,
        # quantization, expert sharding and loaded-parameter bookkeeping.
        return super().load_weights(iter_role_weights(weights, role=self.afd_role))

    def process_weights_after_loading(self) -> None:
        if self.afd_role == "ffn":
            from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

            deepseek_v2_attention_gate.prepare_attention_gate_w8a8_scales(
                self.model.layers
            )

    def get_model_state_cls(self):
        from afd_plugin.model_executor.models.npu.deepseek_v41_state import (
            AFDDeepseekV41ModelState,
        )

        return AFDDeepseekV41ModelState

    def compute_ffn_output(
        self, hidden_states: torch.Tensor, layer_idx: int, **kwargs: Any
    ) -> AFDF2ATransferPayload:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_async_cam_layered_weights(self) -> tuple[list[LayeredFFNWeights], str]:
        # Retain the release's ordinary DR -> native GMM -> CS execution. Its
        # INT8 layered tables do not describe this checkpoint's MXFP4 bank.
        return [], "DSV4.1 currently uses the ordinary Async CAM FFN path"


__all__ = ["AFDDeepseekV41ForCausalLM"]
