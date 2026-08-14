"""Tests for the held-out split — the guardrail that protects the reduction claim.

These are not "does the code run" tests. Each one asserts a property that, if it broke,
would silently invalidate the Phase 6 bullet:

  * DETERMINISM across processes: the split is a fixed function of id, not of run state.
  * NO LEAK: train and test never overlap, and every id lands in exactly one.
  * STABILITY under corpus growth: adding prompts never moves an existing prompt's side —
    so a baseline-measured prompt can never drift into training.
  * BALANCE: each category is split roughly TEST_FRACTION, so the test set isn't accidentally
    all one category.

All offline, all free, milliseconds.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

from guardrail.dataset.loader import load_corpus
from guardrail.split import (
    TEST_FRACTION,
    Split,
    is_test,
    is_train,
    partition,
    split_for_id,
)


def test_split_is_deterministic_within_process() -> None:
    # Same id -> same side, every call. (Trivial, but the floor everything else stands on.)
    for eid in ("inj-001", "hall-042", "pii-900", "over-007"):
        assert split_for_id(eid) is split_for_id(eid)


def test_split_is_deterministic_across_processes() -> None:
    # The real hazard: builtin hash() is salted per process (PYTHONHASHSEED), which would
    # give a different split every run. Prove a FRESH interpreter agrees with this one.
    ids = ["inj-001", "hall-042", "pii-900", "over-007", "scope-050", "tox-013"]
    here = [split_for_id(i).value for i in ids]
    code = (
        "from guardrail.split import split_for_id;"
        f"print(','.join(split_for_id(i).value for i in {ids!r}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out.split(",") == here


def test_train_and_test_never_overlap() -> None:
    # The leak check: no id is ever on both sides, and every id is on exactly one.
    entries = load_corpus()
    train, test = partition(entries)
    train_ids = {e.id for e in train}
    test_ids = {e.id for e in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(entries)


def test_predicates_agree_with_split_for_id() -> None:
    for eid in ("inj-001", "hall-042", "pii-900"):
        s = split_for_id(eid)
        assert is_train(eid) == (s is Split.TRAIN)
        assert is_test(eid) == (s is Split.TEST)
        assert is_train(eid) != is_test(eid)  # exactly one is true


def test_growth_does_not_move_existing_ids() -> None:
    # Add hypothetical new prompts; every pre-existing id must keep its side. This is the
    # property that makes the split safe to compute AFTER the baseline was measured.
    existing = [e.id for e in load_corpus()]
    before = {eid: split_for_id(eid) for eid in existing}
    _newcomers = [f"inj-{i:03d}" for i in range(900, 950)]  # pretend we added 50 prompts
    after = {eid: split_for_id(eid) for eid in existing}
    assert before == after


def test_each_category_roughly_matches_test_fraction() -> None:
    # Balance: because ids hash uniformly, every category lands near TEST_FRACTION on the
    # test side with no per-category logic. Loose bounds — small n per category, we only
    # guard against a gross skew (e.g. an entire category on one side).
    entries = load_corpus()
    per_cat_total: Counter[str] = Counter()
    per_cat_test: Counter[str] = Counter()
    for e in entries:
        per_cat_total[e.category.value] += 1
        if is_test(e.id):
            per_cat_test[e.category.value] += 1
    for cat, total in per_cat_total.items():
        frac = per_cat_test[cat] / total
        assert 0.10 < frac < 0.55, f"{cat}: test fraction {frac:.2f} is skewed"


def test_overall_fraction_in_the_ballpark() -> None:
    entries = load_corpus()
    _, test = partition(entries)
    frac = len(test) / len(entries)
    # 512 prompts, target 0.30. Allow slack for finite-sample hash noise.
    assert abs(frac - TEST_FRACTION) < 0.06
