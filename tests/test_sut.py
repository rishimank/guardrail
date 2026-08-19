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
    5. LoRA checkpoint resolution: the pure, mlx-free half of LoRASUT — discovering
       checkpoints on disk and choosing between them. Exercised against a FAKE adapter
       directory (empty files), because the logic under test is filename and config
       parsing, not weight loading.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from guardrail.sut import (
    DEFAULT_MODEL_ID,
    MLXSUT,
    LoRASUT,
    MockSUT,
    SUT,
    VALID_SUTS,
    checkpoints,
    get_sut,
)
from guardrail.sut.lora_sut import read_config, resolve_checkpoint
from guardrail.sut.mock import DEFAULT_REPLY


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    """A structurally valid adapter dir with empty weight files.

    Nothing here is loadable by mlx, and nothing needs to be: every test using this
    fixture is about WHICH file gets chosen, which is decided entirely from names and
    adapter_config.json. That is what keeps these tests runnable on linux CI.
    """
    d = tmp_path / "v2"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps(
            {
                "model": DEFAULT_MODEL_ID,
                "data": "training",
                "fine_tune_type": "lora",
                "num_layers": 8,
                "iters": 150,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "mask_prompt": True,
                "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
            }
        )
    )
    for it in (25, 50, 75, 100, 125, 150):
        (d / f"{it:07d}_adapters.safetensors").touch()
    (d / "adapters.safetensors").touch()
    return d


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
    # Must not raise something opaque from inside mlx several frames down.
    monkeypatch.setenv("GUARDRAIL_ADAPTER_PATH", "/nonexistent/adapters")
    with pytest.raises(FileNotFoundError, match="not a trained adapter"):
        get_sut("lora")


def test_lora_pointed_at_the_parent_folder_fails_actionably(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `adapters/` EXISTS but holds v1, v2, ... — a plain exists() check would pass it
    # through and mlx would then die on a missing adapter_config.json. This is the
    # realistic mistake, so it gets its own test.
    (tmp_path / "v1").mkdir()
    monkeypatch.setenv("GUARDRAIL_ADAPTER_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        get_sut("lora")


def test_non_integer_checkpoint_is_rejected_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch, adapter_dir: Path
) -> None:
    monkeypatch.setenv("GUARDRAIL_ADAPTER_PATH", str(adapter_dir))
    monkeypatch.setenv("GUARDRAIL_ADAPTER_CHECKPOINT", "best")
    with pytest.raises(ValueError, match="not an integer iteration"):
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


# --- LoRA checkpoint resolution (no mlx, no weights) ------------------------


def test_checkpoints_are_discovered_by_iteration(adapter_dir: Path) -> None:
    assert sorted(checkpoints(adapter_dir)) == [25, 50, 75, 100, 125, 150]


def test_final_weights_are_not_reported_as_a_checkpoint(adapter_dir: Path) -> None:
    # `adapters.safetensors` carries no iteration in its name; its iteration comes
    # from the config. Listing it as checkpoint 0 would be a lie.
    assert all(p.name != "adapters.safetensors" for p in checkpoints(adapter_dir).values())


def test_checkpoints_of_a_missing_dir_is_empty_not_an_error(tmp_path: Path) -> None:
    assert checkpoints(tmp_path / "nope") == {}


def test_unrelated_safetensors_are_ignored(adapter_dir: Path) -> None:
    (adapter_dir / "model.safetensors").touch()
    (adapter_dir / "notes_adapters.safetensors").touch()
    assert sorted(checkpoints(adapter_dir)) == [25, 50, 75, 100, 125, 150]


def test_default_resolution_is_the_final_weights_labelled_from_config(
    adapter_dir: Path,
) -> None:
    path, label = resolve_checkpoint(adapter_dir)
    assert path.name == "adapters.safetensors"
    # The label must come from the config's iters, not from the filename, which has none.
    assert label == 150


def test_explicit_checkpoint_selects_that_file(adapter_dir: Path) -> None:
    path, label = resolve_checkpoint(adapter_dir, 100)
    assert path.name == "0000100_adapters.safetensors"
    assert label == 100


def test_unavailable_checkpoint_lists_what_exists(adapter_dir: Path) -> None:
    # Silently falling back to another checkpoint would mean reporting a number for
    # weights nobody chose — and the verdicts would look completely normal.
    with pytest.raises(ValueError, match=r"No checkpoint at iteration 137"):
        resolve_checkpoint(adapter_dir, 137)
    with pytest.raises(ValueError, match=r"\[25, 50, 75, 100, 125, 150\]"):
        resolve_checkpoint(adapter_dir, 137)


def test_config_missing_names_the_likely_mistake(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not an mlx_lm adapter directory"):
        read_config(tmp_path)


def test_provenance_keys_needed_by_benchmarks_are_in_the_config(
    adapter_dir: Path,
) -> None:
    # BENCHMARKS.md quotes rank/layers/iters from the adapter itself rather than from
    # a retyped shell command, so the config must actually carry them.
    cfg = read_config(adapter_dir)
    assert cfg["lora_parameters"]["rank"] == 8
    assert cfg["num_layers"] == 8
    assert cfg["mask_prompt"] is True


def test_lora_sut_declares_the_protocol_without_being_instantiated() -> None:
    # Same trick as MLXSUT: prove the shape without loading 1.6GB of weights.
    assert hasattr(LoRASUT, "generate")
    assert hasattr(LoRASUT, "model_id")
