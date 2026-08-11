#!/usr/bin/env python
"""reference_calibration.py — Phase 3.4, reference-judge variant (paid).

Instead of a human labelling the 90 golden responses, a STRONGER Claude model
(the "reference judge", Opus 4.8 by default) grades them, and Cohen's kappa then
measures whether the cheap production judge (Haiku, in judge_verdicts.jsonl) grades
the way the strong one does. High kappa => using cheap Haiku at scale is justified.

Honest caveat baked into the method: this proves "cheap judge agrees with strong
judge", NOT "agrees with a human". Opus and Haiku are both Anthropic models and may
share blind spots, so a high kappa is weaker evidence than a human calibration would
be. The claim wording downstream must reflect that.

Only the FOUR JUDGMENT categories are graded (hallucination, toxicity, scope,
overrefusal) = 60 rows. injection/pii are deterministic substring checks: both judges
run the identical code, so kappa there is trivially 1.0 and no API call is warranted.

Resumable + cost-safe: verdicts are appended as they complete and already-graded ids
are skipped, so a re-run (or a --limit probe followed by a full run) never pays twice
for the same row. Prints DeepEval-tracked cost at the end.

PSEUDOCODE
    1. Parse --model (default claude-opus-4-8) and optional --limit (probe N rows).
    2. Load responses.jsonl (Step 1 cache) and index the handwritten golden seeds.
    3. Keep only rows whose category is a JUDGMENT category (skip deterministic).
    4. Load any existing reference_verdicts.jsonl; skip those ids (resume).
    5. build_metrics(get_judge(model)) once — one reference judge reused across rows.
    6. For each remaining (and --limit-capped) row: grade -> append
       {id, category, judge_label, method, score, reason}. flush after each so a
       crash mid-run still banks completed rows.
    7. Print counts + note that DeepEval tracked the native-model cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from guardrail.dataset import load_corpus  # noqa: E402
from guardrail.dataset.schema import Source  # noqa: E402
from guardrail.judge import get_judge  # noqa: E402
from guardrail.judge.metrics import (  # noqa: E402
    JUDGMENT_CATEGORIES,
    build_metrics,
    grade,
)

CAL = Path(__file__).resolve().parent.parent / "calibration"
RESP_PATH = CAL / "responses.jsonl"
REF_VERDICTS_PATH = CAL / "reference_verdicts.jsonl"

DEFAULT_REFERENCE_MODEL = "claude-opus-4-8"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_REFERENCE_MODEL,
        help=f"reference judge model (default {DEFAULT_REFERENCE_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="grade at most N new rows this run (a cheap probe before the full run)",
    )
    args = parser.parse_args()

    if not RESP_PATH.exists():
        print(f"no responses at {RESP_PATH} — run gen_calibration_responses.py first.")
        return 1

    responses = sorted(_read_jsonl(RESP_PATH), key=lambda r: r["id"])
    seeds = {e.id: e for e in load_corpus() if e.source is Source.HANDWRITTEN}

    # Only the judgment categories need a reference judge; deterministic rows are
    # identical across any judge (kappa == 1.0) and would just cost API calls.
    judgment_rows = [
        r for r in responses if seeds[r["id"]].category in JUDGMENT_CATEGORIES
    ]
    done = {r["id"] for r in _read_jsonl(REF_VERDICTS_PATH)}
    todo = [r for r in judgment_rows if r["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"judgment rows: {len(judgment_rows)} | already graded: {len(done)} | "
        f"grading now: {len(todo)}  (reference model: {args.model})"
    )
    if not todo:
        print("nothing to do.")
        return 0

    metrics = build_metrics(judge=get_judge(model=args.model))  # one reference judge
    graded = 0
    with REF_VERDICTS_PATH.open("a") as f:
        for i, row in enumerate(todo, 1):
            entry = seeds[row["id"]]
            v = grade(entry, row["output"], metrics)
            f.write(
                __import__("json").dumps(
                    {
                        "id": entry.id,
                        "category": entry.category.value,
                        "judge_label": "pass" if v.passed else "fail",
                        "method": v.method,
                        "score": v.score,
                        "reason": v.reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            graded += 1
            print(
                f"  [{i:2d}/{len(todo)}] {entry.id:9s} {v.method:13s} "
                f"{'PASS' if v.passed else 'FAIL'} ({v.score:.2f})"
            )

    print(f"\ngraded {graded} responses -> {REF_VERDICTS_PATH}")
    print("(DeepEval tracks native-model cost; Opus 4.8 on 60 rows is ~$0.45.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
