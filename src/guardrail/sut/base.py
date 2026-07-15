"""The SUT (System Under Test) interface — the load-bearing seam of this project.

Everything that attacks, scores, or serves the model talks to `SUT`, never to a
concrete implementation. Three classes satisfy it: `MLXSUT` (real Qwen), `LoRASUT`
(fine-tuned, Phase 6), `MockSUT` (deterministic, offline, free — what CI runs).

This file exists so that Phase 6 is a one-variable change: fine-tune, swap the
implementation, re-run the *identical* harness. If the harness had to change to
accommodate the new model, we could not claim it measured both the same way — and
that claim is the entire reduction number.

PSEUDOCODE
    1. Define `Response`: what one generation returns.
       - text              -> the answer; this is what the judge grades
       - model_id          -> WHICH sut produced it (base vs lora vs mock)
       - latency_s         -> wall-clock; reported by Phase 5, served by Phase 7
       - prompt_tokens     -> input token count
       - completion_tokens -> output token count
       Frozen: a Response is a measurement. Measurements do not get edited.
    2. Define the `SUT` protocol: `model_id` + `generate(prompt, ...) -> Response`.
       - Structural (typing.Protocol), not inheritance: an implementation just has
         to have the right shape. MockSUT imports nothing from here at runtime.
       - temperature defaults to 0.0 (greedy) because an eval harness must be
         reproducible: same prompt in -> same answer out, so a score change is
         attributable to OUR change, not to the model rolling dice.
    3. Export both. No implementations live here — this file is the contract only,
       and must stay importable with zero heavy deps (no mlx) so CI can load it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Greedy decoding. See the honesty note in CLAUDE.md: a non-deterministic SUT would
# mean re-running the same suite twice gives two different "rates", and any measured
# reduction could be sampling noise rather than the fine-tune.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True, slots=True)
class Response:
    """One generation, plus the metadata needed to attribute and cost it."""

    text: str
    model_id: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class SUT(Protocol):
    """The one interface. Implemented by MLXSUT, LoRASUT, and MockSUT."""

    @property
    def model_id(self) -> str:
        """Identifies which system produced a Response (e.g. the HF repo id).

        Phase 6 compares two result sets; without this they are indistinguishable.
        """
        ...

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Response:
        """Send one prompt to the system under test and return its Response.

        `prompt` is the raw user turn. Applying any chat template is the
        implementation's job, not the caller's — the harness must not need to know
        that Qwen wants ChatML while a mock wants nothing.
        """
        ...
