"""The judge package — Claude Haiku wired up as DeepEval's grading model.

DeepEval defaults to grading with an OpenAI model. This package overrides that: every
G-Eval metric in Phase 3 grades with Claude Haiku instead. `get_judge()` is
deliberately a thin factory (like `get_sut()`), not a custom `DeepEvalBaseLLM`
subclass — 4.1.0 ships a native `AnthropicModel` that DeepEval already recognizes, so
it tracks cost automatically and needs no hand-written wrapper. Verified against the
installed 4.1.0. (Phase 3.2's per-category metrics will live alongside this __init__.)

Two facts pinned here on purpose, both discovered by reading the installed package:
  * `claude-haiku-4-5` is in DeepEval's model table with the real ($1/$5 per 1M) prices,
    so `EvaluationCost` is correct out of the box — we do NOT pass pricing by hand.
  * `supports_log_probs` is False for this model. G-Eval's headline trick — a
    probability-weighted score from token logprobs — therefore does NOT apply: the
    judge falls back to a raw INTEGER score (see calculate_weighted_summed_score's
    fallback). This is expected, not a bug. It means the pass/fail THRESHOLD carries
    the weight, which is exactly what Phase 3.4 calibration (Cohen's kappa) pins down.

Why temperature 0: the same reason the SUT is greedy. A judge that rolls dice would
give a different rate on a re-run, and any measured reduction could be judge noise
rather than a real change. A reproducible judge is what makes a delta attributable.

PSEUDOCODE
    1. DEFAULT_JUDGE_MODEL = the Haiku id confirmed live in Phase 0.5.
    2. get_judge(model=None, temperature=0.0):
       a. model defaults to $GUARDRAIL_JUDGE, then to DEFAULT_JUDGE_MODEL — so the
          judge can be swapped by env var without touching metric code, but the
          default is the cheap, calibrated one.
       b. Build a native AnthropicModel(model, temperature). Native => auto cost.
       c. Return it. Callers (the Phase 3.2 metrics) pass it in as `model=`.
"""

from __future__ import annotations

import os

from deepeval.models import AnthropicModel

__all__ = ["DEFAULT_JUDGE_MODEL", "DEFAULT_JUDGE_TEMPERATURE", "get_judge"]

# The judge model. Haiku keeps a full ~1000-call run near $1.35, under the $2 ceiling;
# its id was confirmed to return a real completion in Phase 0.5.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"

# Greedy judging. See the module docstring: reproducibility is what lets a score
# change be attributed to our change and not to the judge sampling differently.
DEFAULT_JUDGE_TEMPERATURE = 0.0


def get_judge(
    model: str | None = None,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
) -> AnthropicModel:
    """Build the DeepEval judge model that every G-Eval metric grades with.

    `model` defaults to $GUARDRAIL_JUDGE, then to DEFAULT_JUDGE_MODEL. Swapping the
    judge (e.g. to test judge sensitivity in calibration) is an env change, never a
    code change — the metrics just receive whatever this returns.
    """
    resolved = (model or os.getenv("GUARDRAIL_JUDGE") or DEFAULT_JUDGE_MODEL).strip()
    # Native model => DeepEval auto-tracks EvaluationCost from the real price table.
    # We pass temperature explicitly; pricing is intentionally left to the table.
    return AnthropicModel(model=resolved, temperature=temperature)
