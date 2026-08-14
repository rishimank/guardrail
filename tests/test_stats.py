"""Offline tests for the Wilson CI primitive (Phase 5.1) — the ruler, calibrated.

Two kinds of check, on purpose:
  * KNOWN VALUES: hand-verifiable cases whose answers do not depend on our code being
    right (0/n, n/n, a coin, the realistic ~49% case). If someone breaks the formula
    or swaps the z-quantile, these move and fail. This is the mutation guard.
  * INVARIANTS: properties that must hold for EVERY (k, n) — the interval brackets the
    point estimate and never leaves [0, 1] — swept over a range so a boundary bug
    (the whole reason we rejected Wald) cannot hide.
"""

from __future__ import annotations

import pytest

from guardrail.stats import Interval, wilson_interval


def test_realistic_case_known_value() -> None:
    # 43/88 ~ 48.9%: the everyday category-sized measurement.
    lo, hi = wilson_interval(43, 88)
    assert lo == pytest.approx(0.3869, abs=1e-3)
    assert hi == pytest.approx(0.5913, abs=1e-3)


def test_all_pass_lower_bound_is_zero_upper_is_not() -> None:
    # 0/70: the toxicity case Wald gets fatally wrong ([0,0]). Wilson: [0, ~0.05].
    lo, hi = wilson_interval(0, 70)
    assert lo == 0.0
    assert 0.0 < hi < 0.10  # "unseen, not impossible"
    assert hi == pytest.approx(0.0519, abs=1e-3)


def test_all_fail_is_mirror_image() -> None:
    lo, hi = wilson_interval(70, 70)
    assert hi == 1.0
    assert 0.90 < lo < 1.0
    assert lo == pytest.approx(0.9481, abs=1e-3)


def test_coin_flip_interval_is_symmetric_and_wide() -> None:
    # 1/2: almost no information -> a huge, centered interval.
    lo, hi = wilson_interval(1, 2)
    assert lo == pytest.approx(1 - hi, abs=1e-9)  # symmetric about 0.5
    assert hi - lo > 0.7  # n=2 tells you almost nothing, and says so


def test_higher_confidence_is_wider() -> None:
    w95 = wilson_interval(43, 88, 0.95)
    w99 = wilson_interval(43, 88, 0.99)
    assert w99.lo < w95.lo and w99.hi > w95.hi


def test_more_samples_narrows_the_interval() -> None:
    # same rate (50%), 10x the data -> a much tighter bound.
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)


@pytest.mark.parametrize("n", [1, 2, 5, 13, 70, 88, 512])
def test_invariants_hold_for_every_k(n: int) -> None:
    for k in range(n + 1):
        lo, hi = wilson_interval(k, n)
        p = k / n
        assert 0.0 <= lo <= hi <= 1.0  # never leaves [0, 1]
        assert lo <= p <= hi           # always brackets the point estimate
        assert isinstance(lo, float) and isinstance(hi, float)  # no np.float64 leak


def test_returns_named_interval() -> None:
    r = wilson_interval(1, 10)
    assert isinstance(r, Interval)
    assert (r.lo, r.hi) == tuple(r)  # still a plain tuple underneath


@pytest.mark.parametrize("bad", [(0, 0), (5, 3), (-1, 10), (3, -2)])
def test_bad_counts_raise(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        wilson_interval(*bad)


@pytest.mark.parametrize("conf", [0.0, 1.0, -0.1, 1.5])
def test_bad_confidence_raises(conf: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 10, conf)
