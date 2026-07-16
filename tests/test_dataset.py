"""The dataset is code, so it gets tested like code.

These tests guard the corpus itself, not the software that reads it. A bad edit to a
JSONL file — a duplicate id, a blank ground truth, a canary on a judge-only category,
a category that quietly became refuse-only — is a data bug that would silently skew a
Phase 5 rate. Here it fails CI instead, offline and in milliseconds.

Two kinds of assertion, labelled below:
  * PERMANENT invariants: must hold in every phase, at any corpus size. These encode
    the project's design (deterministic vs judge-only categories, anti-gaming controls,
    the golden set's integrity).
  * SNAPSHOT facts: true of the 90 handwritten seeds right now. Phase 2.3 (Synthesizer
    -> 500+) will change these ON PURPOSE; update them in that same commit. They exist
    so an *accidental* change to the seed set is caught today.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from guardrail.dataset import Category, Entry, load_corpus
from guardrail.dataset.schema import ExpectedBehavior, Source

# Categories judged by a deterministic check (canary / regex), per CLAUDE.md.
DETERMINISTIC = {Category.INJECTION, Category.PII}
# Everything else is judged only by the LLM judge and must carry no forbidden_outputs.
JUDGE_ONLY = set(Category) - DETERMINISTIC

# --- SNAPSHOT: the 90 hand-written seeds as they stand at Phase 2.2 -----------
# Phase 2.3 will grow these numbers; change them here when it does.
SEEDS_PER_CATEGORY = 15
SEED_TOTAL = SEEDS_PER_CATEGORY * len(Category)


@pytest.fixture(scope="module")
def corpus() -> list[Entry]:
    # If the corpus is invalid this raises CorpusError with a file:line — the failure
    # is the test doing its job.
    return load_corpus()


# --- PERMANENT invariants ----------------------------------------------------


def test_corpus_loads_and_is_nonempty(corpus: list[Entry]) -> None:
    assert corpus, "corpus loaded empty"


def test_ids_are_globally_unique(corpus: list[Entry]) -> None:
    ids = [e.id for e in corpus]
    assert len(ids) == len(set(ids))


def test_no_duplicate_prompts(corpus: list[Entry]) -> None:
    # Exact-duplicate prompts inflate sample size without adding information, which
    # falsely narrows Phase 5's confidence intervals. (Near-dupes are 2.4's job.)
    norm = [re.sub(r"\s+", " ", e.prompt.strip().lower()) for e in corpus]
    dupes = {p for p in norm if norm.count(p) > 1}
    assert not dupes, f"duplicate prompt text: {dupes}"


def test_every_entry_has_prompt_and_ground_truth(corpus: list[Entry]) -> None:
    # The schema enforces this on load; asserted here as a dataset-level guarantee so
    # the contract is visible where the data is tested.
    for e in corpus:
        assert e.prompt.strip()
        assert e.ground_truth.strip()


def test_all_six_categories_present(corpus: list[Entry]) -> None:
    assert {e.category for e in corpus} == set(Category)


def test_every_category_has_an_answer_control(corpus: list[Entry]) -> None:
    # The anti-gaming rule: a category with no "should answer" prompt could be aced by
    # a model that refuses everything. Every category must contain at least one.
    for cat in Category:
        behaviors = {e.expected_behavior for e in corpus if e.category == cat}
        assert ExpectedBehavior.ANSWER in behaviors, f"{cat.value} has no answer-control"


def test_forbidden_outputs_only_on_deterministic_categories(
    corpus: list[Entry],
) -> None:
    for e in corpus:
        if e.category in JUDGE_ONLY:
            assert not e.forbidden_outputs, (
                f"{e.id}: {e.category.value} is judge-only but has forbidden_outputs"
            )


def test_deterministic_categories_actually_use_forbidden_outputs(
    corpus: list[Entry],
) -> None:
    # Injection and PII are supposed to be checkable without an LLM; at least some of
    # their entries must carry the machine-checkable marker or the design is hollow.
    for cat in DETERMINISTIC:
        with_marker = [e for e in corpus if e.category == cat and e.forbidden_outputs]
        assert with_marker, f"{cat.value} has no forbidden_outputs entries"


def test_forbidden_output_strings_are_nonblank(corpus: list[Entry]) -> None:
    for e in corpus:
        for s in e.forbidden_outputs:
            assert s.strip(), f"{e.id}: blank forbidden_output would match everything"


def test_pii_forbidden_values_are_synthetic(corpus: list[Entry]) -> None:
    # A leak-detection corpus must never carry real personal data. Embedded values
    # must be provably fake: 555 fiction phones, example.com, the 4111 test card,
    # placeholder SSNs, and the templated generators (SSN area 900+ is never assigned;
    # 555-01xx is fiction-reserved; userN@example.com can't be registered).
    synthetic = re.compile(
        r"(555-\d{4}|example\.com|4111 1111 1111 1111"
        r"|123-45-6789|987-65-4321|222-33-4444|555-01\d\d|MRN 55123"
        r"|900-\d\d-\d{4}|user\d+@example\.com)"
    )
    for e in corpus:
        if e.category is Category.PII:
            for s in e.forbidden_outputs:
                assert synthetic.search(s), f"{e.id}: forbidden value {s!r} not obviously synthetic"


# --- PERMANENT: the handwritten golden set is frozen -------------------------
# After 2.3 the corpus is a mix (handwritten seeds + templated/synthesized). The
# durable invariant is that the 90 hand-authored seeds — the Phase 3.4 calibration
# set — remain intact: exactly 15 per category, all source=handwritten. This must
# hold no matter how many generated rows are added later.


def test_golden_set_frozen(corpus: list[Entry]) -> None:
    handwritten = [e for e in corpus if e.source is Source.HANDWRITTEN]
    assert len(handwritten) == SEED_TOTAL, "the 90-seed golden set changed size"
    per_cat = Counter(e.category for e in handwritten)
    for cat in Category:
        assert per_cat[cat] == SEEDS_PER_CATEGORY, f"{cat.value} golden count changed"


def test_generated_rows_are_machine_sourced(corpus: list[Entry]) -> None:
    # Anything that isn't a seed must be explicitly marked templated or synthesized —
    # never silently handwritten. Keeps provenance honest for the golden-set split.
    for e in corpus:
        assert e.source in (Source.HANDWRITTEN, Source.TEMPLATED, Source.SYNTHESIZED)
