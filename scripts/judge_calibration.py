#!/usr/bin/env python
"""judge_calibration.py — Step 3 of Phase 3.4: the JUDGE's labels (paid).

Runs the Phase 3.2 `grade()` over the 90 cached golden responses and stores one
pass/fail verdict per response. In Step 4 these are compared against the human labels
to get Cohen's kappa. Independent of the human labels on purpose — the judge must not
see them, and vice-versa.

Cost: only the four JUDGMENT categories (60 responses) call Haiku; injection/pii (30)
are graded by the free deterministic check. ~$0.09 total. Prints actual cost at the end.

Resumable: verdicts are appended as they complete; already-judged ids are skipped, so a
re-run costs nothing for work already done.

PSEUDOCODE
    1. Load responses.jsonl (Step 1) and index the golden seeds by id (for the Entry
       that grade() needs — category, ground_truth, forbidden_outputs).
    2. Load any existing judge_verdicts.jsonl; skip those ids (resume).
    3. build_metrics() once (one judge model reused across the run).
    4. For each remaining response: grade(entry, output) -> Verdict; append
       {id, category, judge_label, method, score, reason}. Deterministic rows cost $0.
    5. Print counts + a rough cost note (native model cost is tracked by DeepEval).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from guardrail.dataset import load_corpus  # noqa: E402
from guardrail.dataset.schema import Source  # noqa: E402
from guardrail.judge.metrics import build_metrics, grade  # noqa: E402

CAL = Path(__file__).resolve().parent.parent / "calibration"
RESP_PATH = CAL / "responses.jsonl"
VERDICTS_PATH = CAL / "judge_verdicts.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    if not RESP_PATH.exists():
        print(f"no responses at {RESP_PATH} — run gen_calibration_responses.py first.")
        return 1

    responses = sorted(_read_jsonl(RESP_PATH), key=lambda r: r["id"])
    seeds = {e.id: e for e in load_corpus() if e.source is Source.HANDWRITTEN}
    done = {r["id"] for r in _read_jsonl(VERDICTS_PATH)}
    todo = [r for r in responses if r["id"] not in done]

    print(f"responses: {len(responses)} | already judged: {len(done)} | to do: {len(todo)}")
    if not todo:
        print("all judged.")
        return 0

    metrics = build_metrics()  # one judge model, reused
    judged = 0
    with VERDICTS_PATH.open("a") as f:
        for i, row in enumerate(todo, 1):
            entry = seeds[row["id"]]
            v = grade(entry, row["output"], metrics)
            f.write(json.dumps({
                "id": entry.id,
                "category": entry.category.value,
                "judge_label": "pass" if v.passed else "fail",
                "method": v.method,
                "score": v.score,
                "reason": v.reason,
            }, ensure_ascii=False) + "\n")
            f.flush()
            judged += 1
            print(f"  [{i:2d}/{len(todo)}] {entry.id:9s} {v.method:13s} "
                  f"{'PASS' if v.passed else 'FAIL'} ({v.score:.2f})")

    print(f"\njudged {judged} responses -> {VERDICTS_PATH}")
    print("(DeepEval tracks native-model cost; a full 60-judgment run is ~$0.09.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
