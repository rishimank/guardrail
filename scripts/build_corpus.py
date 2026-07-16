#!/usr/bin/env python
"""build_corpus.py — write the machine-generated half of the corpus (Phase 2.3).

Idempotent: overwrites each `<category>.generated.jsonl` from scratch, so re-running
never duplicates. Hand-authored seed files are never touched.

This script covers the FREE, templated half (injection / pii / toxicity). The paid
LLM-synthesis half (hallucination / scope / overrefusal) is a separate gated step —
see the 2.3 pilot finding: an LLM refuses to rewrite adversarial seeds, so those
categories are templated instead.

PSEUDOCODE
    1. For each deterministic category, run its expand_* generator (deterministic).
    2. Serialize each Entry to one JSON line (enums -> values, tuples -> lists).
    3. Write <category>.generated.jsonl. Report per-category counts.
"""

from __future__ import annotations

import json
import sys

from guardrail.dataset import Category
from guardrail.dataset.loader import generated_path
from guardrail.dataset.schema import Entry
from guardrail.dataset.templates import (
    expand_injection,
    expand_pii,
    expand_toxicity,
)

TEMPLATED = {
    Category.INJECTION: expand_injection,
    Category.PII: expand_pii,
    Category.TOXICITY: expand_toxicity,
}


def write_generated(category: Category, entries: list[Entry]) -> int:
    path = generated_path(category)
    lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False) for e in entries]
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main() -> int:
    print("templated generation (free):")
    total = 0
    for category, expand in TEMPLATED.items():
        entries = list(expand())
        n = write_generated(category, entries)
        total += n
        print(f"  {category.value:12s} {n:3d} -> {generated_path(category).name}")
    print(f"total templated: {total}")
    print("\nnote: hallucination/scope/overrefusal are the paid LLM step (gated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
