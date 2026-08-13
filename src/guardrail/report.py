"""Turn a run's COUNTS into a REPORT of bounded failure RATES (Phase 5.2).

This is the thin composition layer between two things that each do exactly one job:
the runner produces `RunSummary` (per-category pass/fail COUNTS, no division), and
`stats.wilson_interval` turns one (k, n) pair into a confidence interval. This module
walks the summary, calls the primitive once per category (plus once for the grand
total), and packages rate + counts + CI into a `RunReport` that can render itself as
the markdown table that lands in BENCHMARKS.md.

It contains ZERO statistics of its own — every interval comes from the one locked-down
`wilson_interval`. That is deliberate: 5.3 tests that primitive in isolation, and
because nothing here re-implements the math, those tests protect this whole report.

DIRECTION: every rate here is a FAILURE rate — k = the number of prompts the model got
WRONG, so higher is always worse, for all six categories. Overrefusal is no exception:
it is itself a failure category (refusing when it should have answered IS the failure),
so its "rate" is the rate of bad refusals, not of refusals in general.

PSEUDOCODE
    _rate(category, k, n, conf):
        rate = k/n; ci = wilson_interval(k, n, conf); bundle into CategoryRate.
    summarize(RunSummary, conf=0.95) -> RunReport:
        1. for each category: _rate(cat, summary.failed_in_cat, summary.total_in_cat).
        2. overall = _rate("overall", summary.failed, summary.n).
        3. return RunReport(model_id, n, overall, by_category, conf).
    RunReport.format_markdown():
        one row per category (sorted) + an overall row: n, fails, fail-rate %, 95% CI.
"""

from __future__ import annotations

from dataclasses import dataclass

from guardrail.runner import RunSummary
from guardrail.stats import Interval, wilson_interval


@dataclass(frozen=True, slots=True)
class CategoryRate:
    """One category's failure rate with its confidence interval.

    `rate` and `ci` are FAILURE-direction: `failures` out of `n`. Higher = worse.
    The counts are kept alongside the rate on purpose — a rate is not interpretable
    without the n that produced it (5.1's whole point), so the report never drops it.
    """

    category: str
    n: int
    failures: int
    rate: float
    ci: Interval

    @property
    def passes(self) -> int:
        return self.n - self.failures


def _rate(category: str, k: int, n: int, conf: float) -> CategoryRate:
    """Bundle a (k failures out of n) count into a rate + Wilson CI."""
    return CategoryRate(
        category=category,
        n=n,
        failures=k,
        rate=k / n,
        ci=wilson_interval(k, n, conf),
    )


@dataclass(frozen=True, slots=True)
class RunReport:
    """A whole run expressed as failure rates + CIs, ready to render for BENCHMARKS.md."""

    model_id: str
    n: int
    overall: CategoryRate
    by_category: dict[str, CategoryRate]
    conf: float

    def format_markdown(self) -> str:
        pct = int(round(self.conf * 100))
        lines = [
            f"model: `{self.model_id}` · n = {self.n} · {pct}% Wilson CI",
            "",
            f"| category | n | fails | fail rate | {pct}% CI |",
            "|---|---:|---:|---:|---|",
        ]
        for cat in sorted(self.by_category):
            lines.append(_row(self.by_category[cat]))
        lines.append(_row(self.overall, bold=True))
        return "\n".join(lines)


def _row(r: CategoryRate, *, bold: bool = False) -> str:
    name = f"**{r.category}**" if bold else r.category
    lo, hi = r.ci
    return (
        f"| {name} | {r.n} | {r.failures} | "
        f"{r.rate * 100:.1f}% | [{lo * 100:.1f}%, {hi * 100:.1f}%] |"
    )


def summarize(summary: RunSummary, *, conf: float = 0.95) -> RunReport:
    """Map per-category COUNTS to per-category failure RATES + Wilson CIs."""
    by_category = {
        cat: _rate(cat, cs.failed, cs.total, conf)
        for cat, cs in summary.by_category.items()
    }
    overall = _rate("overall", summary.failed, summary.n, conf)
    return RunReport(
        model_id=summary.model_id,
        n=summary.n,
        overall=overall,
        by_category=by_category,
        conf=conf,
    )


__all__ = ["CategoryRate", "RunReport", "summarize"]
