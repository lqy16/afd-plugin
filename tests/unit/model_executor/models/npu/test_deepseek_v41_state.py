# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tests.unit.fakes.deepseek_v41 import (  # noqa: E402
    GraphMode,
    config_for_role,
    load_replay,
    replay_inputs,
)


@pytest.fixture
def replay(monkeypatch):
    module = load_replay(monkeypatch)
    layers = SimpleNamespace(
        window=128, rows=None, context_factory=None, num_padded_rows=None
    )
    state = module.AFDDeepseekV41ReplayState(config_for_role(), layers, "cpu")
    return module, state, layers


@pytest.mark.parametrize("length", [256, 16384, 32768])
def test_async_dp_retains_tail_without_global_coordination(replay, length):
    _, state, layers = replay
    state.prepare(replay_inputs([length]), GraphMode.NONE)
    assert list(layers.rows) == list(range(length - 128, length))
    assert callable(layers.context_factory)


def test_full_prompt_logprobs_keep_all_rows(replay):
    _, state, layers = replay
    state.keeps_rows[0] = True
    state.prepare(replay_inputs([16384]), GraphMode.NONE)
    assert layers.rows is None
    assert layers.context_factory is None


@pytest.mark.parametrize(
    "mode,capture",
    [(GraphMode.FULL, False), (GraphMode.NONE, True), (GraphMode.NONE, False)],
)
def test_capture_and_next_short_batch_do_not_reuse_previous_rows(replay, mode, capture):
    _, state, layers = replay
    state.prepare(replay_inputs([16384]), GraphMode.NONE)
    state.prepare(replay_inputs([16]), mode, for_capture=capture)
    assert layers.rows is None
    assert layers.context_factory is None


def test_model_state_constructs_local_replay_once_and_keeps_native_request_state(
    replay,
):
    module, _, layers = replay
    model = SimpleNamespace(decoder_replay_layers=layers, token_lookback_depth=0)
    state = module.AFDDeepseekV41ModelState(config_for_role(), model, None, "cpu")
    state.decoder_replay_state.prepare(replay_inputs([512]), GraphMode.NONE)
    assert list(layers.rows) == list(range(384, 512))
