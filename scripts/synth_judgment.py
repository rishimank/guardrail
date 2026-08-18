#!/usr/bin/env python
"""synth_judgment.py — LLM-synthesize + verify the judgment categories (Phase 2.3, paid).

Generates hallucination/scope/overrefusal prompts via Haiku structured output, runs an
INDEPENDENT verification pass, keeps only survivors, dedups against seeds, and writes
each `<category>.generated.jsonl`. Idempotent per category (overwrites).

Run small first to measure and eyeball, then full:
    scripts/synth_judgment.py --n 6                              # measured probe (~cents)
    scripts/synth_judgment.py --n 60                             # full run, all 3 categories
    scripts/synth_judgment.py --category overrefusal --n 150 --append   # grow one category

⚠️ DEFAULT IS DESTRUCTIVE. Without --append this OVERWRITES each target category's
generated file with fresh prompts. Once a baseline has been measured against the corpus
(Phase 5), overwriting silently invalidates it: the banked verdicts in runs/ are keyed by
id, so replacing over-042's prompt while keeping its id makes the run file a lie. After a
baseline exists, ALWAYS use --append, which preserves existing rows and their ids and only
adds new ones numbered after the current maximum.

PSEUDOCODE
    1. For each target category (--category, default all three): load seeds; in --append
       mode also load the existing generated rows and keep them.
    2. Generate ~n items (passing existing prompts as `avoid`), verify, keep the ones the
       verifier passed AND whose prompt isn't a dup of a seed / existing row / this batch.
    3. Map survivors to Entry(source=SYNTHESIZED), ids continuing after the current max.
    4. Write the generated file; print cost, keep-rate, and samples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

from guardrail.dataset import Category, load_category  # noqa: E402
from guardrail.dataset.loader import generated_path  # noqa: E402
from guardrail.dataset.schema import (  # noqa: E402
    Entry,
    ExpectedBehavior,
    Severity,
    Source,
)
from guardrail.dataset.synthesis import CostMeter, generate, verify  # noqa: E402

JUDGMENT = [Category.HALLUCINATION, Category.SCOPE, Category.OVERREFUSAL]
PREFIX = {Category.HALLUCINATION: "hall", Category.SCOPE: "scope", Category.OVERREFUSAL: "over"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _existing_generated(category: Category) -> list[Entry]:
    """Rows already in <category>.generated.jsonl, or [] if the file is absent."""
    path = generated_path(category)
    if not path.exists() or not path.read_text().strip():
        return []
    return [e for e in load_category(category) if e.source is Source.SYNTHESIZED]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="items to generate per category")
    ap.add_argument(
        "--category",
        action="append",
        choices=[c.value for c in JUDGMENT],
        help="repeatable; default = all three judgment categories",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="keep existing generated rows and add after them (REQUIRED once a baseline "
             "has been measured — the default overwrites and invalidates runs/)",
    )
    args = ap.parse_args()

    targets = [Category(c) for c in args.category] if args.category else list(JUDGMENT)

    client = Anthropic()
    meter = CostMeter()
    mode = "APPEND" if args.append else "OVERWRITE"
    print(f"synthesizing {args.n}/category for {[c.value for c in targets]} "
          f"({mode}, paid)...\n")

    for category in targets:
        # Seeds only (hand-authored) drive the exemplars; existing generated rows are
        # carried forward untouched in append mode so their ids stay bound to their
        # prompts — that binding is what makes the banked baseline in runs/ still valid.
        seeds = [e for e in load_category(category) if e.source is not Source.SYNTHESIZED]
        carried = _existing_generated(category) if args.append else []
        seed_prompts = {_norm(e.prompt) for e in seeds} | {_norm(e.prompt) for e in carried}

        # Continue numbering after the highest id currently in use, so a new row can never
        # collide with (and silently overwrite) a measured one.
        idx = 16
        for e in carried:
            suffix = int(e.id.rsplit("-", 1)[1])
            idx = max(idx, suffix + 1)

        avoid = [e.prompt for e in seeds] + [e.prompt for e in carried]
        items = generate(client, category, seeds, args.n, meter, avoid=avoid)
        verdicts = {v.index: v for v in verify(client, category, items, meter)}

        kept: list[Entry] = list(carried)
        added = 0
        seen: set[str] = set()
        for i, it in enumerate(items):
            v = verdicts.get(i)
            norm = _norm(it.prompt)
            if v is None or not v.keep:
                continue
            if norm in seed_prompts or norm in seen:
                continue  # dedup vs seeds and within batch
            seen.add(norm)
            kept.append(Entry(
                id=f"{PREFIX[category]}-{idx:03d}",
                category=category,
                prompt=it.prompt,
                ground_truth=it.ground_truth,
                expected_behavior=ExpectedBehavior(it.expected_behavior),
                # Simple heuristic: refusals matter more than benign answers for weighting.
                severity=Severity.MEDIUM if it.expected_behavior == "refuse" else Severity.LOW,
                source=Source.SYNTHESIZED,
                tags=("synthesized",),
            ))
            idx += 1

        path = generated_path(category)
        lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False) for e in kept]
        path.write_text("\n".join(lines) + "\n" if lines else "")
        rejected = len(items) - len(kept)
        print(f"{category.value:14s} generated {len(items):3d} -> kept {len(kept):3d} "
              f"(rejected {rejected}) -> {path.name}")

    print(f"\ncost: ${meter.dollars:.4f} over {meter.calls} calls")
    print("\n=== sample kept prompts (eyeball) ===")
    for category in JUDGMENT:
        rows = [e for e in load_category(category) if e.source is Source.SYNTHESIZED]
        if rows:
            e = rows[0]
            print(f"\n[{category.value}] {e.id} ({e.expected_behavior.value})")
            print(f"  prompt: {e.prompt[:120]}")
            print(f"  truth : {e.ground_truth[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
