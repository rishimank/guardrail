#!/usr/bin/env python
"""label_calibration.py — Step 2 of Phase 3.4: the human's blind labels.

Shows you, for each of the 90 golden responses, the prompt + the authored ground truth
(the correct-behavior description) + the model's actual output, and asks: did the model
behave correctly? pass / fail. Your answers become the ground truth for the JUDGE — in
Step 4 we measure how often the judge agrees with you (Cohen's kappa).

BLIND on purpose: this tool never shows a machine verdict. You grade against the same
ground_truth the judge sees (fair parity), but you must not see the judge's pass/fail —
if you did, you'd anchor to it and the agreement number would be meaningless.

Resumable: each label is appended immediately; already-labeled ids are skipped, so you
can label in several sittings. Order is by id, which groups categories together.

PSEUDOCODE
    1. Load responses.jsonl (Step 1 output). Refuse to run if it isn't there yet.
    2. Load any existing labels.jsonl; skip those ids (resume).
    3. For each remaining response: print category / prompt / correct-behavior /
       model output. Read p(ass) / f(ail) / s(kip) / q(uit). Re-prompt on bad input.
    4. Append {id, category, human_label} per answer, flushed immediately.
    5. On quit or finish: print counts labeled / remaining and the pass:fail split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CAL = Path(__file__).resolve().parent.parent / "calibration"
RESP_PATH = CAL / "responses.jsonl"
LABELS_PATH = CAL / "labels.jsonl"

RULE = "=" * 78


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _ask(index: int, total: int, row: dict) -> str | None:
    """Show one item and return 'pass'/'fail', None to skip, or raise on quit."""
    print(f"\n{RULE}\n[{index}/{total}]  id={row['id']}  category={row['category']}")
    print(f"\nPROMPT:\n  {row['prompt']}")
    print(f"\nCORRECT BEHAVIOR (ground truth):\n  {row['ground_truth']}")
    print(f"\nMODEL OUTPUT:\n  {row['output'].strip()}")
    print(f"\n{'-'*78}")
    while True:
        choice = input("Did the model behave correctly? [p]ass / [f]ail / [s]kip / [q]uit: ")
        c = choice.strip().lower()
        if c in ("p", "pass"):
            return "pass"
        if c in ("f", "fail"):
            return "fail"
        if c in ("s", "skip"):
            return None
        if c in ("q", "quit"):
            raise KeyboardInterrupt
        print("  please type p, f, s, or q.")


def main() -> int:
    if not RESP_PATH.exists():
        print(f"no responses at {RESP_PATH} — run gen_calibration_responses.py first.")
        return 1

    responses = sorted(_read_jsonl(RESP_PATH), key=lambda r: r["id"])
    labeled = {r["id"]: r["human_label"] for r in _read_jsonl(LABELS_PATH)}
    todo = [r for r in responses if r["id"] not in labeled]

    print(f"responses: {len(responses)} | already labeled: {len(labeled)} | "
          f"remaining: {len(todo)}")
    if not todo:
        print("all labeled. done.")
        return 0
    print("Grade whether the OUTPUT matches the CORRECT BEHAVIOR. p/f/s/q at each prompt.")

    added = 0
    try:
        with LABELS_PATH.open("a") as f:
            for i, row in enumerate(todo, 1):
                label = _ask(i, len(todo), row)
                if label is None:
                    continue  # skipped: stays in todo for a later sitting
                f.write(json.dumps(
                    {"id": row["id"], "category": row["category"], "human_label": label},
                    ensure_ascii=False,
                ) + "\n")
                f.flush()
                added += 1
    except (KeyboardInterrupt, EOFError):
        print("\n\nstopped.")

    total_labeled = len(labeled) + added
    all_labels = list(labeled.values()) + [
        r["human_label"] for r in _read_jsonl(LABELS_PATH) if r["id"] not in labeled
    ]
    npass = all_labels.count("pass")
    nfail = all_labels.count("fail")
    print(f"\nlabeled this session: {added} | total labeled: {total_labeled}/"
          f"{len(responses)} | remaining: {len(responses) - total_labeled}")
    print(f"label split so far: {npass} pass / {nfail} fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
