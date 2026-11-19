# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

from tests.unit.fakes.async_ffn import load_host_module


@pytest.fixture
def weights(monkeypatch):
    return load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.npu.deepseek_v41_weights"
    )


@pytest.mark.parametrize(
    "path,roles",
    [
        ("layers.14.engram.embed.weight", {"attention"}),
        ("model.layers.20.self_attn.wo_a.scale", {"attention"}),
        ("layers.0.ffn_norm.weight", {"attention"}),
        ("layers.39.hc_ffn_base", {"attention"}),
        ("layers.2.ffn.shared_experts.w1.weight", {"attention"}),
        ("model.layers.2.mlp.shared_experts.down_proj.weight", {"attention"}),
        ("layers.2.ffn.experts.383.w1.scale", {"ffn"}),
        ("model.layers.2.mlp.experts.47.w2.weight", {"ffn"}),
        ("layers.0.ffn.gate.tid2eid", {"attention", "ffn"}),
        ("layers.2.ffn.gate.bias", {"attention", "ffn"}),
        ("layers.0.ffn.gate.bias_vl", {"attention"}),
        ("model.layers.2.mlp.gate.bias_vl", {"attention"}),
        ("embed.weight", {"attention", "ffn"}),
        ("head.weight", {"attention", "ffn"}),
        ("aligner.w1.weight", set()),
        ("model.aligner.gate_up_proj.weight", set()),
        ("vision.layers.0.attn.qkv.weight", set()),
        ("model.vision.layers.0.attn.qkv.weight", set()),
        ("image_newline", set()),
        ("model.image_embedding.weight", set()),
        ("mtp.0.ffn.experts.0.w1.weight", set()),
        ("model.mtp.0.ffn.experts.0.w1.weight", set()),
    ],
)
def test_checkpoint_paths_follow_role_before_native_name_remapping(
    weights, path, roles
):
    assert weights.checkpoint_weight_roles(path) == frozenset(roles)
