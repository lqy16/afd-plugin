# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role/loader/protocol contracts for V4.1; hardware arithmetic is not tested."""

import pytest

torch = pytest.importorskip("torch")

from tests.unit.fakes.deepseek_v41 import (  # noqa: E402
    NativeLayer,
    NativeModel,
    config_for_role,
    load_adapter,
    prepare_proxy_protocol,
)


@pytest.fixture
def contract(monkeypatch):
    return load_adapter(monkeypatch)


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_forty_layer_role_has_only_its_execution_weights(contract, role):
    config = config_for_role(role, engram=True)
    with torch.device("meta"):
        model = contract.adapter.AFDDeepseekV41ForCausalLM(vllm_config=config)
    parameters = dict(model.named_parameters())
    assert len(model.model.layers) == 40
    assert model.model.config.n_routed_experts == 384
    assert model.model.config.n_shared_experts == 1
    if role == "attention":
        assert "model.layers.1.engram.proj.weight" in parameters
        assert "model.layers.0.hc_ffn_fn" in parameters
        assert "model.layers.0.self_attn.wq.weight" in parameters
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" in parameters
        assert not any(".mlp.experts." in name for name in parameters)
        assert model.model.decoder_replay_layers.window == 128
    else:
        assert model.get_experts_layer_indices() == tuple(range(40))
        assert "model.layers.0.mlp.experts.weight" in parameters
        assert not any(
            ".self_attn." in n or ".engram." in n or ".hc_" in n for n in parameters
        )
        assert not any(".shared_experts." in n for n in parameters)
        assert model.model.decoder_replay_layers is None


def test_native_forward_inheritance_keeps_hc_and_replay_contract(contract):
    assert contract.adapter.AFDDeepseekV41DecoderLayer.forward is NativeLayer.forward
    assert contract.adapter.AFDDeepseekV41Model.forward is NativeModel.forward


@pytest.mark.parametrize("vision_layers", [0, 1])
@pytest.mark.parametrize("layer_idx", [0, 2])
def test_attention_gate_loads_native_optional_vision_bias(
    contract, vision_layers, layer_idx
):
    config = config_for_role(small=True)
    text = config.model_config.hf_config
    text.vision_n_layers = vision_layers
    proxy = contract.adapter.AFDDeepseekV41AttentionGateRemoteMoE(
        config=text,
        layer_idx=layer_idx,
        prefix=f"model.layers.{layer_idx}.mlp",
        vllm_config=config,
    )
    if vision_layers == 0:
        assert proxy.gate.bias_vl is None
        assert "bias_vl" not in dict(proxy.gate.named_parameters())
    else:
        weight = torch.arange(text.n_routed_experts, dtype=torch.float32)
        proxy.gate.load_state_dict({"bias_vl": weight}, strict=False)
        torch.testing.assert_close(proxy.gate.bias_vl, weight)
        assert proxy.gate.bias_vl.dtype == torch.float32
        assert not proxy.gate.bias_vl.requires_grad


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_native_loader_receives_original_names_and_tensor_objects(contract, role):
    config = config_for_role(role, small=True)
    model = contract.adapter.AFDDeepseekV41ForCausalLM(vllm_config=config)
    supplied = [
        ("layers.0.attn.wq_a.weight", torch.ones(1)),
        ("layers.0.ffn.gate.weight", torch.ones(1)),
        ("layers.0.ffn.experts.0.w1.scale", torch.ones(1)),
        ("layers.0.ffn.shared_experts.w1.weight", torch.ones(1)),
        ("layers.0.hc_ffn_fn", torch.ones(1)),
        ("layers.1.engram.embed.weight", torch.ones(1)),
        ("head.weight", torch.ones(1)),
        ("aligner.w1.weight", torch.ones(1)),
        ("model.aligner.gate_up_proj.weight", torch.ones(1)),
        ("vision.layers.0.attn.qkv.weight", torch.ones(1)),
        ("model.vision.layers.0.attn.qkv.weight", torch.ones(1)),
        ("image_newline", torch.ones(1)),
        ("model.image_embedding.weight", torch.ones(1)),
        ("mtp.0.ffn.experts.0.w1.weight", torch.ones(1)),
        ("model.mtp.0.ffn.experts.0.w1.weight", torch.ones(1)),
    ]
    loaded = model.load_weights(iter(supplied))
    expected = {"layers.0.ffn.gate.weight", "head.weight"}
    if role == "ffn":
        expected.add("layers.0.ffn.experts.0.w1.scale")
    else:
        expected.update(
            {
                "layers.0.attn.wq_a.weight",
                "layers.0.ffn.shared_experts.w1.weight",
                "layers.0.hc_ffn_fn",
                "layers.1.engram.embed.weight",
            }
        )
    assert loaded == expected
    originals = dict(supplied)
    assert all(tensor is originals[name] for name, tensor in model.loaded_weights)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("tensor_parallel_size", 2, "TP1"),
        ("pipeline_parallel_size", 2, "PP1"),
        ("use_ubatching", True, "DBO"),
        ("prefill_context_parallel_size", 2, "PCP1"),
    ],
)
def test_initial_execution_scope_is_explicit(contract, field, value, match):
    config = config_for_role(small=True)
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match=match):
        contract.adapter.AFDDeepseekV41ForCausalLM(vllm_config=config)


def test_ffn_is_selected_by_transmitted_layer_index(contract):
    config = config_for_role("ffn", small=True)
    model = contract.adapter.AFDDeepseekV41ForCausalLM(vllm_config=config)
    for layer in model.model.layers:
        layer.compute_ffn_output = lambda hidden, layer_id=layer.layer_idx, **kw: (
            layer_id,
            hidden,
        )
    hidden = torch.ones(2, 8)
    assert [model.compute_ffn_output(hidden, layer_idx=n)[0] for n in (2, 0, 1)] == [
        2,
        0,
        1,
    ]


def test_engram_async_dp_requires_local_or_shared_table(contract):
    config = config_for_role(small=True, engram=True)
    config.engram_config.dp_shared_memory = False
    contract.native.get_engram_dp_size = lambda: 2
    with pytest.raises(ValueError, match="Engram dp_shared_memory"):
        contract.adapter.AFDDeepseekV41ForCausalLM(vllm_config=config)


def test_native_decoder_accepts_v41_extra_moe_argument(contract, monkeypatch):
    config = config_for_role(small=True)
    proxy = contract.adapter.AFDDeepseekV41AttentionGateRemoteMoE(
        config=config.model_config.hf_config,
        layer_idx=2,
        prefix="model.layers.2.mlp",
        vllm_config=config,
    )
    # The external parent-forward seam is shared with V4. Its transaction itself
    # is already covered there; this test targets the V4.1 keyword extension.
    parent = contract.adapter.AFDDeepseekV4AttentionGateRemoteMoE

    def forward(self, hidden_states, input_ids=None, hidden_states_fp32=None):
        return hidden_states + hidden_states_fp32.to(hidden_states.dtype)

    monkeypatch.setattr(parent, "forward", forward)
    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    result = proxy(
        hidden,
        input_ids=torch.ones(2),
        hidden_states_fp32=hidden.float(),
        already_sequence_parallel=False,
    )
    torch.testing.assert_close(result, hidden * 2)


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_opposite_role_is_never_constructed(contract, monkeypatch, role):
    def forbidden(*args, **kwargs):
        raise AssertionError("opposite execution role was constructed")

    if role == "attention":
        monkeypatch.setattr(contract.native.DeepseekV41MoE, "__init__", forbidden)
    else:
        monkeypatch.setattr(NativeLayer.attention_cls, "__init__", forbidden)
    config = config_for_role(role, small=True)
    contract.adapter.AFDDeepseekV41ForCausalLM(vllm_config=config)


def test_existing_cam_proxy_sends_v41_layer_and_fp32_routing_input(
    contract, monkeypatch
):
    protocol = prepare_proxy_protocol(monkeypatch, contract)
    config = config_for_role(small=True)
    layer = contract.adapter.AFDDeepseekV41DecoderLayer(
        vllm_config=config, prefix="model.layers.2"
    )
    hidden = torch.ones(3, 8, dtype=torch.bfloat16)
    pre_mix = torch.ones(3, 4)
    output, result_mix = layer(
        torch.arange(3), hidden, pre_mix, input_ids=torch.arange(3)
    )
    assert [message[0] for message in protocol] == ["dispatch", "combine"]
    dispatch = protocol[0][1]
    assert dispatch.metadata.layer_idx == 2
    assert dispatch.metadata.total_tokens == 3
    assert result_mix is pre_mix
    torch.testing.assert_close(output, hidden * 3)
