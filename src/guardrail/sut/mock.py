"""MockSUT — a fake system under test: deterministic, offline, free.

This is what CI, Docker, and every unit test run against. It loads no model, needs
no API key, and answers in microseconds. It exists so that a failing test means
*our harness is broken*, not that a 3B model had an off day — it separates "is the
measuring instrument correct?" from "what does it measure?".

Why deterministic and not random: a fake that returns random text can only tell you
the plumbing didn't crash. A fake with *known* answers lets a test assert a real
expectation ("the hallucination metric must flag this response"), because the input
to the metric is pinned. Random output would make every downstream assertion either
trivial or flaky.

PSEUDOCODE
    1. Hold a dict of canned {prompt substring -> reply} plus a default reply.
       Callers can pass their own via `responses=` so a test can script the SUT.
    2. On generate(prompt):
       a. Find the first canned key that appears in the prompt (case-insensitive).
       b. Fall back to `default_reply` if nothing matches.
       c. Fake the token counts by splitting on whitespace — no tokenizer, because
          importing one would defeat the "zero heavy deps" purpose.
       d. Measure real (tiny) latency anyway, so the Response is shaped exactly
          like a real one and nothing downstream can tell the difference.
    3. Return a Response with model_id="mock" — so a result file produced against
       the mock can never be mistaken for a real measurement.
"""

from __future__ import annotations

import time

from guardrail.sut.base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, Response

MODEL_ID = "mock"

# Deliberately boring, safe, in-scope answers. A mock that "passes" every category
# by default means a test asserting a FAILURE has to opt in explicitly, which keeps
# accidental green from looking like real green.
DEFAULT_CANNED: dict[str, str] = {
    "capital of france": "Paris.",
    "2+2": "4.",
}
DEFAULT_REPLY = "I don't have enough information to answer that."


class MockSUT:
    """Satisfies the SUT protocol with canned replies. No model, no network."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_reply: str = DEFAULT_REPLY,
    ) -> None:
        # Keys lowercased once here so generate() stays a cheap lookup.
        source = DEFAULT_CANNED if responses is None else responses
        self._responses = {k.lower(): v for k, v in source.items()}
        self._default_reply = default_reply

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Response:
        t0 = time.perf_counter()

        needle = prompt.lower()
        text = self._default_reply
        for key, reply in self._responses.items():
            if key in needle:
                text = reply
                break
        else:
            if len(prompt) > 58:
                text = f"Sure — here's what you asked for: {prompt}"

        # max_tokens is honoured approximately (word-wise). It is not a real
        # tokenizer, and it does not need to be: the mock's job is shape fidelity,
        # not token fidelity.
        words = text.split()
        if len(words) > max_tokens:
            text = " ".join(words[:max_tokens])

        return Response(
            text=text,
            model_id=MODEL_ID,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )
