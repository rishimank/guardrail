"""Offline tests for the counts->rates report builder (Phase 5.2).

The load-bearing test here is DIRECTION: a "hallucination rate" must count FAILURES,
not passes. The runner stores `passed`; the report must divide `failed / total`. If
someone wires `passed` in by mistake the rate is 1 - the truth, so `test_rate_counts_
failures_not_passes` is the mutation guard for the entire report. Everything else
checks that the report defers to `wilson_interval` rather than re-doing any math.
"""

from __future__ import annotations

from guardrail.report import summarize
from guardrail.runner import CategorySummary, RunSummary
from guardrail.stats import wilson_interval


def _summary() -> RunSummary:
    cats = {
        "hallucination": CategorySummary("hallucination", total=88, passed=43),  # 45 fail
        "toxicity": CategorySummary("toxicity", total=70, passed=70),            # 0 fail
    }
    return RunSummary(model_id="demo", n=158, by_category=cats)


def test_rate_counts_failures_not_passes() -> None:
    rep = summarize(_summary())
    hall = rep.by_category["hallucination"]
    assert hall.failures == 45          # 88 - 43, NOT 43
    assert hall.rate == 45 / 88         # failure direction: higher = worse
    assert hall.passes == 43


def test_ci_defers_to_wilson_primitive() -> None:
    # the report must not re-implement stats: its CI == the primitive's, exactly.
    rep = summarize(_summary())
    hall = rep.by_category["hallucination"]
    assert hall.ci == wilson_interval(45, 88)


def test_overall_row_aggregates_all_categories() -> None:
    rep = summarize(_summary())
    assert rep.overall.n == 158
    assert rep.overall.failures == 45          # 45 + 0
    assert rep.overall.rate == 45 / 158
    assert rep.overall.ci == wilson_interval(45, 158)


def test_all_pass_category_has_zero_rate_but_nonzero_upper_ci() -> None:
    rep = summarize(_summary())
    tox = rep.by_category["toxicity"]
    assert tox.rate == 0.0
    assert tox.ci.lo == 0.0
    assert tox.ci.hi > 0.0  # all-pass is "unseen", not "impossible"


def test_confidence_level_flows_through() -> None:
    rep = summarize(_summary(), conf=0.99)
    assert rep.conf == 0.99
    assert rep.by_category["hallucination"].ci == wilson_interval(45, 88, 0.99)


def test_markdown_has_a_row_per_category_plus_overall() -> None:
    md = summarize(_summary()).format_markdown()
    assert "| hallucination |" in md
    assert "| toxicity |" in md
    assert "| **overall** |" in md      # overall bolded
    assert "51.1%" in md                # 45/88 failure rate rendered
    assert "95% Wilson CI" in md
