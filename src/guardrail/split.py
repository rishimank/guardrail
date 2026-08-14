"""The held-out train/test split — the integrity crux of the Phase 6 reduction claim.

The second resume bullet is a *reduction* number ("N% fewer violations after fine-tuning").
That number is only honest if the model is measured on prompts it was never trained on.
If a single measured prompt leaks into training, the model can memorise it and the whole
reduction is meaningless. So this file is the one guardrail that has to be bulletproof.

The design decision (made back in the schema, see dataset/schema.py) is that the split is
**NOT a stored field**. A `split: "train"` column in the JSONL could be edited — by hand, by
a bug, by a well-meaning "let me just move this one" — and the leak would be silent. Instead
the split is a pure, deterministic FUNCTION of the prompt's `id`. There is nothing to edit.
Re-derive it anywhere (mining, training, reporting) and you get the identical partition, every
time, with no shared state. Un-storable is the point.

Why hashing the id (not, say, "first 70% of the file"): a hash is stable under reordering,
insertion, and deletion. Add 50 new injection prompts tomorrow and every existing prompt keeps
its side of the split — so a prompt that was measured in the baseline can never quietly cross
into training just because the corpus grew. The hash also spreads ids uniformly, so each
category lands ~TEST_FRACTION on the test side without any per-category bookkeeping (ids are
category-prefixed, and the hash is blind to the prefix's meaning — it just scatters).

TEST_FRACTION = 0.30. Rationale: training data comes only from TRAIN-split *failures*, and the
corpus has just 224 of them, so we favour the training side (70%) to give the fine-tune enough
signal to actually move the needle; the 30% test side (~154 prompts) still yields a defensible
overall Wilson CI. Per-category test CIs will be wide (~25 prompts each) and we report them
honestly rather than pretending they are tight.

PSEUDOCODE
    1. Split enum: TRAIN | TEST.
    2. _bucket(id): sha256(id) -> int in [0, 1000). Stable across processes/machines
       (hashlib, NOT Python's salted hash()).
    3. split_for_id(id): TEST if bucket < TEST_FRACTION*1000 else TRAIN.
    4. is_train / is_test: convenience predicates.
    5. partition(entries): split a list into (train, test) preserving order.
    6. counts(entries): {"train": .., "test": ..} for logging/sanity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum

# The measured (held-out) fraction. See the module docstring for why 0.30 and not 0.20/0.50.
# Changing this re-draws the boundary, which would invalidate any already-measured reduction —
# so it is a constant here, versioned in git, never a runtime knob.
TEST_FRACTION = 0.30

_RESOLUTION = 1000  # bucket granularity; 0.30 -> ids with bucket < 300 are TEST


class Split(StrEnum):
    TRAIN = "train"  # fine-tune on these
    TEST = "test"  # measure the reduction on these; NEVER trained on


def _bucket(entry_id: str) -> int:
    """Map an id to a stable integer in [0, _RESOLUTION).

    Uses hashlib (sha256), not the builtin hash(): hash() is salted per process
    (PYTHONHASHSEED), so it would give a DIFFERENT split every run — the one thing a
    leak-proof split must never do. sha256 is identical on every machine, forever.
    """
    digest = hashlib.sha256(entry_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _RESOLUTION


def split_for_id(entry_id: str) -> Split:
    """The whole contract: an id deterministically belongs to TRAIN or TEST."""
    return Split.TEST if _bucket(entry_id) < TEST_FRACTION * _RESOLUTION else Split.TRAIN


def is_train(entry_id: str) -> bool:
    return split_for_id(entry_id) is Split.TRAIN


def is_test(entry_id: str) -> bool:
    return split_for_id(entry_id) is Split.TEST


def partition(entries: Iterable[object]) -> tuple[list, list]:
    """Split any id-bearing objects (Entry, verdict dict, ...) into (train, test).

    Accepts either objects with an `.id` attribute or dicts with an `"id"` key, so the
    same function partitions corpus Entries and banked verdict rows identically — which
    is exactly what keeps mining and reporting on the same side of the line.
    """
    train: list = []
    test: list = []
    for e in entries:
        eid = e["id"] if isinstance(e, dict) else e.id  # type: ignore[attr-defined]
        (train if is_train(eid) else test).append(e)
    return train, test


def counts(entries: Iterable[object]) -> dict[str, int]:
    """Return {"train": n, "test": m} for a batch — for logging and sanity checks."""
    train, test = partition(entries)
    return {"train": len(train), "test": len(test)}


__all__ = [
    "TEST_FRACTION",
    "Split",
    "counts",
    "is_test",
    "is_train",
    "partition",
    "split_for_id",
]
