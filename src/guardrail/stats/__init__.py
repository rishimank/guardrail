"""Statistics primitives — turn pass/fail COUNTS into bounded RATES (Phase 5).

The runner hands us counts on purpose (k failures out of n prompts) and refuses to
divide, because a bare fraction is not a claim. `43/88 = 48.9%` looks precise but is
a lie of precision: with only 88 samples the true rate could plausibly be anywhere in
a ~20-point band. This module computes that band. It is the difference between "the
hallucination rate is 48.9%" and "48.9%, 95% CI [38.5%, 59.4%], n=88" — the second is
the only one the honesty note lets us claim.

WHY WILSON, NOT THE TEXTBOOK (WALD) INTERVAL
    The interval everyone is taught is Wald:  p̂ ± z·√(p̂(1−p̂)/n). It is wrong for
    exactly our situation and we must not use it:
      * small n per category (~70–93 prompts) — its normal approximation needs large n;
      * rates near 0 or 1 — toxicity is ~all-pass, so k≈0. Wald on 0/70 gives the CI
        [0, 0]: a confident, false claim that the model NEVER emits toxicity. It can
        also hand back a lower bound below 0 or an upper bound above 1 — impossible
        probabilities.
    The Wilson score interval is derived by inverting the score test instead of
    assuming the estimate is symmetric and normal. Consequences we want:
      * it is ALWAYS inside [0, 1];
      * it stays sensible at the boundaries (0/70 -> [0, ~0.05], not [0, 0]);
      * it is ASYMMETRIC near the edges — which is correct: if you saw 0 failures the
        truth might be 3% but obviously not −3%, so the interval only reaches upward.

PSEUDOCODE
    wilson_interval(k, n, conf):
        1. reject n <= 0 (no observations -> no interval) and k outside [0, n].
        2. z = the normal quantile for the two-sided conf level (0.95 -> 1.96), via
           scipy so the number is honest, not a hardcoded 1.96.
        3. p̂ = k/n.
        4. center = (p̂ + z²/2n) / (1 + z²/n)          # pulled toward 1/2 by z²/n
           half   = (z / (1 + z²/n)) · √( p̂(1−p̂)/n + z²/4n² )
        5. clamp (center ± half) into [0, 1] to kill floating-point overshoot and
           return Interval(lo, hi).
"""

from __future__ import annotations

from typing import NamedTuple

from scipy.stats import norm


class Interval(NamedTuple):
    """A closed confidence interval on a proportion. Both ends live in [0, 1]."""

    lo: float
    hi: float


def wilson_interval(k: int, n: int, conf: float = 0.95) -> Interval:
    """95%-by-default Wilson score interval for `k` successes in `n` trials.

    "Success" here just means the counted event — a failure count and a pass count are
    both proportions, so this is direction-agnostic: feed it failures to get a failure
    rate's CI, passes to get a pass rate's CI.

    Raises ValueError on n <= 0 (an interval needs data) or k outside [0, n].
    """
    if n <= 0:
        raise ValueError(f"need at least one trial, got n={n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n]; got k={k}, n={n}")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1); got {conf}")

    z = norm.ppf(1 - (1 - conf) / 2)  # two-sided: 0.95 -> ppf(0.975) -> 1.9599...
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return Interval(float(lo), float(hi))  # scipy hands back np.float64; keep it plain


__all__ = ["Interval", "wilson_interval"]
