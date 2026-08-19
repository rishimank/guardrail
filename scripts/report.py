#!/usr/bin/env python
"""report.py — render a banked run's verdicts as the BENCHMARKS.md rates table (Phase 5.5).

Reads runs/<sut>/verdicts.jsonl (produced by run_eval.py), rolls it up to per-category
pass/fail COUNTS via the runner's `aggregate`, then converts those to FAILURE RATES with
95% Wilson CIs via `report.summarize`, and prints the markdown table. This is a $0,
offline, repeatable step: it never calls the model or the judge — it only re-reads
verdicts already banked. Run it as many times as you like.

The commit SHA is printed alongside so the table can be pasted into BENCHMARKS.md with
its provenance (method + n + CI + SHA) intact, as the honesty note requires.

`--split test` filters the banked verdicts to the held-out prompts before rolling up —
the same partition write_benchmarks.py applies, exposed here for a quick look at one run
without regenerating BENCHMARKS.md. Filtering happens at REPORT time from ids, so a
full-corpus run can be re-read as a test-split report without re-running anything.

PSEUDOCODE
    1. Parse --sut (default mlx) / --verdicts (explicit path) / --split / --conf.
    2. Read verdicts.jsonl; apply the split filter; pull model_id from the rows.
    3. aggregate(verdicts) -> RunSummary (counts).
    4. summarize(summary, conf) -> RunReport (rates + Wilson CIs).
    5. Print format_markdown() + the current commit SHA.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from guardrail.report import summarize
from guardrail.runner import aggregate
from guardrail.split import Split, split_for_id

REPO = Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sut", default="mlx", help="reads runs/<sut>/verdicts.jsonl")
    parser.add_argument("--verdicts", default=None, help="explicit verdicts.jsonl path")
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="restrict the banked verdicts to a split before rolling up (default all)",
    )
    parser.add_argument("--conf", type=float, default=0.95, help="CI confidence (0-1)")
    args = parser.parse_args()

    path = Path(args.verdicts) if args.verdicts else REPO / "runs" / args.sut / "verdicts.jsonl"
    if not path.exists():
        print(f"no verdicts at {path} — run scripts/run_eval.py first.")
        return 1

    verdicts = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if not verdicts:
        print(f"{path} is empty.")
        return 1

    if args.split != "all":
        # Re-derived from sha256(id), never read from a stored field — see split.py.
        wanted = Split(args.split)
        before = len(verdicts)
        verdicts = [v for v in verdicts if split_for_id(v["id"]) is wanted]
        if not verdicts:
            print(f"no {args.split}-split rows among the {before} verdicts in {path}.")
            return 1

    model_id = next((v.get("model_id", "") for v in verdicts if v.get("model_id")), args.sut)
    summary = aggregate(verdicts, model_id=model_id)
    report = summarize(summary, conf=args.conf)

    src = path.resolve()
    try:
        src_label = src.relative_to(REPO)
    except ValueError:
        src_label = src  # verdicts outside the repo (e.g. an ad-hoc --verdicts path)
    print(report.format_markdown())
    print(f"\n_source: {src_label} · commit `{_git_sha()}`_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
