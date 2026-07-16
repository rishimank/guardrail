"""The adversarial corpus: schema, loader, and public surface.

Import from here (`from guardrail.dataset import load_corpus, Entry`); callers do not
touch the JSONL files or the schema internals directly.

PSEUDOCODE
    Re-export the schema types (Entry + enums) and the loader functions. That is all
    this package's front door needs to expose in Phase 2.
"""

from __future__ import annotations

from guardrail.dataset.loader import (
    CorpusError,
    DATASET_DIR,
    category_path,
    load_category,
    load_corpus,
)
from guardrail.dataset.schema import (
    Category,
    Entry,
    ExpectedBehavior,
    Severity,
    Source,
)

__all__ = [
    "CorpusError",
    "DATASET_DIR",
    "Category",
    "Entry",
    "ExpectedBehavior",
    "Severity",
    "Source",
    "category_path",
    "load_category",
    "load_corpus",
]
