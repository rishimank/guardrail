"""Tests for the SUT seam — the interface, the mock, and the factory.

EVERY test here runs against MockSUT. Nothing downloads a model, nothing needs an
API key, nothing touches the network. That is the point: this suite must pass on a
linux GitHub Actions runner where mlx cannot even be installed, and it must run in
milliseconds so it is cheap enough to run constantly.

What is deliberately NOT tested: whether Qwen gives good answers. That is not a
unit test, it is Phase 5's measurement. These tests check the *instrument*, not
what it measures — if they fail, our harness is broken, not the model.

PSEUDOCODE
    1. Contract tests: MockSUT satisfies SUT; Response is frozen and sums tokens.
    2. MockSUT behaviour: canned lookup, substring + case-insensitive match, the
       default reply for unknown prompts, scripted responses, max_tokens truncation.
    3. Factory tests: default is mock (free!), explicit names, env var is honoured,
       bad name -> ValueError, lora without an adapter -> FileNotFoundError.
    4. Shape tests: MLXSUT satisfies the protocol WITHOUT instantiating it (no 1.6GB
       download in CI) — checked structurally via isinstance on the class's shape.
"""

from __future__ import annotations

import dataclasses

import pytest

from guardrail.sut import (
    DEFAULT_MODEL_ID,
    MLXSUT,
    MockSUT,
    SUT,
    VALID_SUTS,
    get_sut,
)
from guardrail.sut.mock import DEFAULT_REPLY


# --- the contract -----------------------------------------------------------


def test_mock_satisfies_the_sut_protocol() -> None:
    assert isinstance(MockSUT(), SUT)


def test_response_is_frozen() -> None:
    # A Response is a measurement. Nothing downstream may rewrite an answer
    # before the judge sees it.
    r = MockSUT().generate("hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.text = "tampered"  # type: ignore[misc]


def test_response_carries_everything_the_harness_needs() -> None:
    r = MockSUT().generate("What is the capital of France?")
    assert r.text
    assert r.model_id == "mock"
    assert r.latency_s >= 0
    assert r.prompt_tokens > 0
    assert r.completion_tokens > 0
    assert r.total_tokens == r.prompt_tokens + r.completion_tokens


# --- MockSUT behaviour ------------------------------------------------------


def test_canned_answer_is_returned() -> None:
    assert MockSUT().generate("What is the capital of France?").text == "Paris."


def test_matching_is_case_insensitive_and_substring_based() -> None:
    # The prompt is a full sentence; the canned key is a fragment inside it.
    assert MockSUT().generate("Tell me: the CAPITAL OF FRANCE, please").text == "Paris."


def test_unknown_prompt_returns_the_default_reply() -> None:
    assert MockSUT().generate("Who wrote the novel Zorgon?").text == DEFAULT_REPLY


def test_responses_can_be_scripted_for_a_test() -> None:
    # This is what lets a Phase 3 metric test pin its input exactly.
    sut = MockSUT(responses={"zorgon": "Zorgon is a real novel by Jane Doe."})
    assert "Jane Doe" in sut.generate("Summarise the novel Zorgon").text


def test_custom_default_reply_is_used() -> None:
    sut = MockSUT(responses={}, default_reply="nope")
    assert sut.generate("anything at all").text == "nope"


def test_max_tokens_truncates_output() -> None:
    sut = MockSUT(responses={}, default_reply="one two three four five")
    assert sut.generate("x", max_tokens=2).text == "one two"


def test_mock_is_deterministic() -> None:
    # The property the whole eval leans on: same prompt in -> same answer out.
    sut = MockSUT()
    assert sut.generate("capital of france").text == sut.generate("capital of france").text


# --- the factory ------------------------------------------------------------


def test_default_is_mock_when_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wrong default must cost $0 and download nothing.
    monkeypatch.delenv("GUARDRAIL_SUT", raising=False)
    assert get_sut().model_id == "mock"


def test_explicit_name_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAIL_SUT", "mlx")
    assert get_sut("mock").model_id == "mock"


def test_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAIL_SUT", "mock")
    assert get_sut().model_id == "mock"


def test_name_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GUARDRAIL_SUT", raising=False)
    assert get_sut("  MOCK ").model_id == "mock"


def test_unknown_sut_raises_with_the_valid_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GUARDRAIL_SUT", raising=False)
    with pytest.raises(ValueError, match="Unknown GUARDRAIL_SUT"):
        get_sut("gpt4")


def test_lora_without_an_adapter_fails_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Phase 6 has not happened yet, so this must not raise something opaque from
    # inside mlx several frames down.
    monkeypatch.setenv("GUARDRAIL_ADAPTER_PATH", "/nonexistent/adapters")
    with pytest.raises(FileNotFoundError, match="Phase 6"):
        get_sut("lora")


def test_valid_suts_are_all_accepted_names() -> None:
    assert set(VALID_SUTS) == {"mock", "mlx", "lora"}


# --- MLXSUT's shape, without loading 1.6GB ----------------------------------


def test_mlx_sut_declares_the_protocol_without_being_instantiated() -> None:
    # runtime_checkable protocols only check method/attr presence, so this proves
    # MLXSUT's shape matches without importing mlx or downloading a model — which
    # is exactly what CI needs.
    assert hasattr(MLXSUT, "generate")
    assert hasattr(MLXSUT, "model_id")


def test_default_model_id_is_the_documented_one() -> None:
    # CLAUDE.md's memory-budget reasoning (M1 Pro / 16GB) depends on 3B/4-bit.
    assert DEFAULT_MODEL_ID == "mlx-community/Qwen2.5-3B-Instruct-4bit"
