#!/usr/bin/env python
"""run_eval.py — drive the Phase 3.3 corpus runner from the CLI.

Generates SUT responses for the corpus (or a filtered slice), grades them with the
calibrated Haiku judge, and prints per-category pass/fail COUNTS. Artifacts land in
runs/<sut>/ (responses.jsonl + verdicts.jsonl), both resumable.

COST: injection/pii grade for free (deterministic). The other four categories call
Haiku (~$0.09 per 60 judgment rows). A full base-model run over all 512 prompts is
~24 min of local generation + roughly ~$0.50 of judging. This script prints how many
judgment rows it is about to grade before doing so.

Free offline smoke (no API): restrict to the deterministic categories against the mock:
    venv/bin/python scripts/run_eval.py --sut mock --category injection --category pii

PSEUDOCODE
    1. Parse --sut / --out-dir / --category (repeatable) / --limit / --no-resume.
    2. Load the corpus; apply category filter and limit.
    3. Count judgment rows (the ones that will cost money) and print a cost note.
    4. run_corpus(get_sut(sut), entries, out_dir) with a printing progress callback.
    5. Print the counts table + artifact paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from guardrail.dataset import load_corpus  # noqa: E402
from guardrail.dataset.schema import Category  # noqa: E402
from guardrail.judge.metrics import DETERMINISTIC_CATEGORIES  # noqa: E402
from guardrail.runner import run_corpus  # noqa: E402
from guardrail.sut import get_sut  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sut", default="mock", help="mock | mlx | lora (default mock)")
    parser.add_argument("--out-dir", default=None, help="default runs/<sut>/")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="restrict to a category (repeatable); default = all six",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap total prompts")
    parser.add_argument("--no-resume", action="store_true", help="ignore banked rows")
    args = parser.parse_args()

    entries = load_corpus()
    if args.category:
        wanted = {Category(c) for c in args.category}
        entries = [e for e in entries if e.category in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        print("no entries after filtering.")
        return 1

    judgment_n = sum(1 for e in entries if e.category not in DETERMINISTIC_CATEGORIES)
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "runs" / args.sut

    print(f"corpus: {len(entries)} prompts | sut={args.sut} | out={out_dir}")
    if judgment_n:
        print(
            f"⚠️  {judgment_n} judgment rows will call the Haiku judge "
            f"(~${judgment_n / 60 * 0.09:.2f}). {len(entries) - judgment_n} "
            f"deterministic rows are free."
        )
    else:
        print("all rows deterministic — no judge calls, $0.")

    summary = run_corpus(
        get_sut(args.sut),
        entries,
        out_dir,
        resume=not args.no_resume,
        progress=print,
    )
    print("\n" + summary.format_table())
    print(f"\nartifacts -> {out_dir}/responses.jsonl, verdicts.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
