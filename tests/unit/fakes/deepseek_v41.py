# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU doubles for f9c8e0bdb V4.1 model seams; no NPU arithmetic is simulated."""

from __future__ import annotations

import ast
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn

from tests.unit.fakes.async_ffn import load_host_module
from tests.unit.fakes.vllm_config import npu_vllm_config

ROOT = Path(__file__).resolve().parents[3]


class GraphMode(Enum):
    NONE = 0
    FULL = 1


class RemoteProxy(nn.Module):
    def __init__(self, *, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, bias=False)

    @property
    def weight_fp32(self):
        return self.weight.float()


class Attention(nn.Module):
    def __init__(self, *, config, **kwargs):
        super().__init__()
        self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=object()))


class SharedMLP(nn.Module):
    def __init__(self, *, hidden_size, intermediate_size, **kwargs):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden):
        return torch.zeros_like(hidden)


class MoE(nn.Module):
    def __init__(self, *, config, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.gate = Linear(config.hidden_size, config.n_routed_experts)
        self.experts = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.shared_experts = (
            SharedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
            )
            if config.n_shared_experts is not None
            else None
        )


class Engram(nn.Module):
    def __init__(self, config, quant, prefix):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)


class NativeLayer(nn.Module):
    attention_cls = Attention

    # Signature and delayed-HC handoff from f9c8e0bdb. The native mathematical
    # implementation is exercised only on the server; this is an API sentinel.
    def forward(
        self, positions, hidden_states, pre_mix, llama_4_scaling=None, input_ids=None
    ):
        result = self.mlp(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states.float(),
            already_sequence_parallel=self.use_sequence_parallel,
        )
        return result, pre_mix


class NativeModel(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.has_engram = vllm_config.engram_config is not None
        self.layers = nn.ModuleList(
            self.decoder_layer_cls(vllm_config, f"{prefix}.layers.{i}")
            for i in range(self.config.num_hidden_layers)
        )
        self.decoder_replay_layers = SimpleNamespace(window=self.config.sliding_window)
        self.shared_attention_state = object()

    def forward(self, input_ids, positions, intermediate_tensors, **kwargs):
        raise AssertionError("native forward is a server-side numerical control")

    def _make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return torch.empty(
            batch_size,
            self.hc_mult,
            self.config.hidden_size,
            dtype=dtype,
            device=device,
        )


class NativeCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = self.model_cls(vllm_config=vllm_config, prefix=f"{prefix}model")
        self.loaded_weights = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)
        return {name for name, _ in self.loaded_weights}


class ReplayState:
    def __init__(self, config, layers, device):
        self.layers = layers
        self.slots = None
        self.keeps_rows = np.zeros(config.scheduler_config.max_num_seqs, dtype=np.bool_)
        self.config = config

    def _initialize(self, inputs):
        self.slots = object()

    def _build_replay_plan(self, inputs):
        lens = np.diff(inputs.batch.query_start_loc_np)
        trim = (
            inputs.batch.is_prefilling_np
            & ~self.keeps_rows[inputs.batch.idx_mapping_np]
        )
        kept = np.where(trim, np.minimum(lens, self.layers.window), lens)
        return SimpleNamespace(
            kept_lens=kept,
            num_tokens=int(kept.sum()),
            requires_compaction=bool((kept < lens).any()),
        )

    def _coordinate_replay_across_dp(self, plan):
        raise AssertionError("independent AFD DP must never rendezvous here")

    def _prepare_compact_replay(self, inputs, plan):
        offsets = inputs.batch.query_start_loc_np
        self.layers.rows = np.concatenate(
            [
                np.arange(end - kept, end)
                for end, kept in zip(offsets[1:], plan.kept_lens, strict=True)
            ]
        )
        self.layers.context_factory = lambda: None


class BaseModelState:
    def __init__(self, config, model, encoder_cache, device):
        self.max_num_reqs = config.scheduler_config.max_num_seqs


class NativeEngramState(BaseModelState):
    pass


class CpuGpuBuffer:
    def __init__(self, size, *, dtype, device, pin_memory):
        self.np = np.zeros(size, dtype=np.int32)
        self.gpu = torch.zeros(size, dtype=dtype, device=device)


def config_for_role(role="attention", *, small=False, engram=False):
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncMegaAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        num_attention_ranks=2,
        num_ffn_ranks=2,
        extra_config={"dynamicQuant": 1},
        data_parallel_size=2,
    )
    text = SimpleNamespace(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        num_hidden_layers=3 if small else 40,
        hidden_size=8 if small else 5120,
        vocab_size=16 if small else 129280,
        n_routed_experts=8 if small else 384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=128 if small else 2304,
        hidden_act="silu",
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=1,
        rope_parameters={"original_max_position_embeddings": 131072},
        rms_norm_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        sliding_window=128,
        kv_source_layer_ids=[2, 8, 14, 20],
        engram_layer_ids=[1, 14],
    )
    config.model_config.hf_config = text
    config.model_config.hf_text_config = text
    config.model_config.enforce_eager = True
    config.cache_config = SimpleNamespace(swa_bounded_replay=True)
    config.quant_config = None
    config.engram_config = SimpleNamespace(dp_shared_memory=True) if engram else None
    config.scheduler_config = SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=16384
    )
    return config


def replay_inputs(lengths, *, dummy=False):
    return SimpleNamespace(
        batch=SimpleNamespace(
            query_start_loc_np=np.array([0, *np.cumsum(lengths)], dtype=np.int32),
            is_prefilling_np=np.array([True] * len(lengths)),
            idx_mapping_np=np.arange(len(lengths)),
            is_dummy=dummy,
        )
    )


def load_adapter(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("vllm.config", VllmConfig=SimpleNamespace, CUDAGraphMode=GraphMode)
    layers = stub(
        "vllm.model_executor.layers", fused_moe=SimpleNamespace(FusedMoEFactory=object)
    )
    stub("vllm.model_executor.layers.linear", ReplicatedLinear=Linear)
    native = stub(
        "vllm_ascend.models.deepseek_v41.model",
        DeepseekV41DecoderLayer=NativeLayer,
        DeepseekV41Model=NativeModel,
        AscendDeepseekV41LLMForCausalLM=NativeCausalLM,
        DeepseekV41MLP=SharedMLP,
        DeepseekV41MoE=MoE,
        AscendEngram=Engram,
        normalize_deepseek_v41_config=lambda config: config,
        PPMissingLayer=nn.Identity,
        RMSNorm=lambda size, eps: nn.LayerNorm(size, eps=eps),
        VocabParallelEmbedding=lambda vocab, hidden, **kw: nn.Embedding(vocab, hidden),
        make_layers=lambda count, factory, prefix: (
            0,
            count,
            nn.ModuleList(factory(prefix=f"{prefix}.{i}") for i in range(count)),
        ),
        current_platform=SimpleNamespace(device_type="cpu"),
        _engram_enabled_for_runtime=lambda config, vllm_config: (
            vllm_config.engram_config is not None
        ),
        get_engram_dp_size=lambda: 1,
    )
    stub("vllm_ascend.models.deepseek_v41", model=native)
    stub("afd_plugin.connectors", AFDF2ATransferPayload=SimpleNamespace)
    stub("afd_plugin.model_executor.models.deepseek_v2", RemoteFFNProxy=RemoteProxy)
    # Execute V4's real proxy forward to test the retained transaction protocol.
    source = ast.parse(
        (ROOT / "afd_plugin/model_executor/models/npu/deepseek_v4.py").read_text()
    )
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef)
        and n.name == "AFDDeepseekV4AttentionGateRemoteMoE"
    )
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    ]
    namespace = {"RemoteFFNProxy": RemoteProxy, "torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        cls,
                    ],
                    type_ignores=[],
                )
            ),
            "<v4-proxy>",
            "exec",
        ),
        namespace,
    )
    stub(
        "afd_plugin.model_executor.models.npu.deepseek_v4",
        AFDDeepseekV4AttentionGateRemoteMoE=namespace[cls.name],
    )
    load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.npu.deepseek_v41_weights"
    )
    adapter = load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.npu.deepseek_v41"
    )
    return SimpleNamespace(
        adapter=adapter, native=native, proxy_namespace=namespace, layers=layers
    )


def load_replay(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    stub("vllm.config", CUDAGraphMode=GraphMode)
    stub("vllm.v1.utils", CpuGpuBuffer=CpuGpuBuffer)
    stub("vllm_ascend.ascend_forward_context", get_mrv2_in_profile_run=lambda: False)
    stub(
        "vllm_ascend.models.deepseek_v41.decoder_replay_state",
        DecoderReplayInputs=SimpleNamespace,
        DecoderReplayState=ReplayState,
    )
    stub(
        "vllm_ascend.models.deepseek_v41.engram.model_state",
        EngramModelState=NativeEngramState,
    )
    stub("vllm_ascend.worker.v2.model_states.default", AscendModelState=BaseModelState)
    return load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.npu.deepseek_v41_state"
    )


def prepare_proxy_protocol(monkeypatch, contract):
    """Use real CAM layout/metadata and V4 proxy, with a CPU transport peer."""

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("vllm.forward_context", DPMetadata=SimpleNamespace)
    metadata = load_host_module(monkeypatch, "afd_plugin.connectors.metadata")
    stub(
        "vllm.distributed", tensor_model_parallel_all_gather=lambda tensor, dim: tensor
    )
    stub(
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    stub("vllm.logger", init_logger=lambda name: __import__("logging").getLogger(name))
    layout = load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.async_cam_layout"
    )
    gate = stub(
        "afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate",
        compute_attention_gate_topk=lambda moe, hidden, **kw: (
            torch.full((len(hidden), 6), 1 / 6),
            torch.zeros((len(hidden), 6), dtype=torch.int32),
        ),
    )
    stub("afd_plugin.model_executor.models.npu", deepseek_v4_attention_gate=gate)
    messages = []

    def send(connector, hidden, context, **routing):
        messages.append(("dispatch", context))

        def receive():
            messages.append(("combine", context))
            return hidden * (context.metadata.layer_idx + 1)

        return receive

    contract.proxy_namespace.update(
        prepare_cam_dispatch_payload=layout.prepare_cam_dispatch_payload,
        restore_cam_dispatch_output=layout.restore_cam_dispatch_output,
        get_afd_metadata_from_forward_context=lambda: SimpleNamespace(
            stage_idx=0, connector=object()
        ),
        AFDTransferContext=metadata.AFDTransferContext,
        AFDTransferMetadata=metadata.AFDTransferMetadata,
        send_async_cam_remote_call=send,
        maybe_apply_dbo_yield=lambda *args, **kw: None,
    )
    return messages
