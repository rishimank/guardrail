#!/usr/bin/env python
"""synth_pilot.py — a small, MEASURED probe of Phase 2.3 generation cost.

Before committing to a ~$2-7 estimated full synthesizer run, this generates from a
handful of seeds with real cost tracking on, so we can replace a guess with a number
(the 0.9-tok/s lesson applied to money). It prints actual $ spent, per-golden cost,
and an extrapolation to the full run — and dumps the generated prompts so we can eye
their quality and, crucially, whether their ground truth would survive.

PSEUDOCODE
    1. load_dotenv (ANTHROPIC_API_KEY) and build DeepEval's native AnthropicModel
       pinned to real Haiku pricing ($1/$5 per 1M) — native => auto cost tracking.
    2. Take N seeds spread across categories; map each to a DeepEval Golden
       (input=prompt, expected_output=ground_truth, our schema fields in metadata).
    3. Synthesizer(cost_tracking=True).generate_goldens_from_goldens(...) with a
       small fan-out. This is the paid step.
    4. Report: seeds, generated count, ACTUAL cost, $/generated-golden, and the
       linear extrapolation to ~450 new goldens. Save generated prompts for review.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from deepeval.dataset.golden import Golden  # noqa: E402
from deepeval.models import AnthropicModel  # noqa: E402
from deepeval.synthesizer import Synthesizer  # noqa: E402

from guardrail.dataset import Category, load_category  # noqa: E402

# Real Haiku 4.5 pricing (per token). Pinned so cost tracking is accurate even if
# DeepEval's internal price table doesn't know claude-haiku-4-5.
COST_IN = 1e-6
COST_OUT = 5e-6

PILOT_CATEGORIES = [
    Category.HALLUCINATION,
    Category.INJECTION,
    Category.PII,
    Category.TOXICITY,
    Category.OVERREFUSAL,
]
MAX_PER_SEED = 2  # fan-out; keep small for a cheap probe
TARGET_NEW_GOLDENS = 450  # what the full 2.3 run would generate


def main() -> int:
    seeds = [load_category(c)[0] for c in PILOT_CATEGORIES]  # first of each category
    goldens = [
        Golden(  # type: ignore[call-arg]  # Golden's optional fields have runtime defaults
            input=e.prompt,
            expected_output=e.ground_truth,
            additional_metadata={
                "category": e.category.value,
                "expected_behavior": e.expected_behavior.value,
                "severity": e.severity.value,
                "forbidden_outputs": list(e.forbidden_outputs),
                "tags": list(e.tags),
            },
        )
        for e in seeds
    ]

    model = AnthropicModel(
        model="claude-haiku-4-5",
        cost_per_input_token=COST_IN,
        cost_per_output_token=COST_OUT,
    )
    synth = Synthesizer(model=model, cost_tracking=True)

    print(f"seeds: {len(goldens)} ({', '.join(c.value for c in PILOT_CATEGORIES)})")
    print(f"fan-out: {MAX_PER_SEED}/seed — generating, this is the paid step...\n")

    generated = synth.generate_goldens_from_goldens(
        goldens=goldens,
        max_goldens_per_golden=MAX_PER_SEED,
        include_expected_output=True,
    )

    cost = synth.synthesis_cost or 0.0
    n = len(generated)
    per = cost / n if n else 0.0

    print("\n=== MEASURED ===")
    print(f"generated:            {n} goldens from {len(goldens)} seeds")
    print(f"actual cost:          ${cost:.4f}")
    print(f"cost per golden:      ${per:.4f}")
    print(f"extrapolated to {TARGET_NEW_GOLDENS}: ${per * TARGET_NEW_GOLDENS:.2f}")

    print("\n=== SAMPLE GENERATED PROMPTS (eyeball quality + ground-truth risk) ===")
    for g in generated[:6]:
        print(f"\n- input: {g.input[:140]}")
        if g.expected_output:
            print(f"  expected_output: {g.expected_output[:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
