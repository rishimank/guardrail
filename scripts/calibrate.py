#!/usr/bin/env python
"""calibrate.py — Step 4 of Phase 3.4: Cohen's kappa, computed by hand (free).

Joins the human labels (Step 2) with the judge verdicts (Step 3) by id and answers the
one question the whole project rests on: does the judge grade the way the human does?
Reports kappa overall and per judgment category, a bootstrap 95% CI, the confusion
matrix, and — most useful for fixing rubrics — the list of every disagreement.

Kappa is computed from the formula, not a library, so the number is transparent:
    p_o = observed agreement (fraction where human == judge)
    p_e = agreement expected by chance = sum_k (human rate of k)*(judge rate of k)
    kappa = (p_o - p_e) / (1 - p_e)
Why kappa and not raw agreement: if one label dominates, a judge that always guesses it
scores high raw agreement while being useless. Kappa subtracts that chance agreement out.

DEGENERATE case (1 - p_e == 0): both raters used a single label for everything, so kappa
is undefined. We say so and fall back to reporting raw agreement, rather than print NaN.

Interpretation (Landis-Koch): <0.2 slight, .2-.4 fair, .4-.6 moderate, .6-.8 substantial,
.8-1 almost perfect. Target for the judgment categories: >= 0.6, ideally > 0.7.

PSEUDOCODE
    1. Load human labels + judge verdicts; inner-join on id (only items with BOTH).
    2. kappa(pairs): compute p_o, p_e, kappa; return None if degenerate.
    3. bootstrap_ci(pairs): resample with replacement B times, recompute kappa, take
       the 2.5/97.5 percentiles (skip degenerate resamples).
    4. Report: overall kappa+CI+agreement, per-category (judgment cats highlighted,
       deterministic cats noted as near-trivial), 2x2 confusion, all disagreements.
    5. Write calibration/report.md + report.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

CAL = Path(__file__).resolve().parent.parent / "calibration"
LABELS_PATH = CAL / "labels.jsonl"
VERDICTS_PATH = CAL / "judge_verdicts.jsonl"
DETERMINISTIC = {"injection", "pii"}
LABELS = ("pass", "fail")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for a list of (human, judge) label pairs. None if degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    p_o = sum(1 for h, j in pairs if h == j) / n
    human = Counter(h for h, _ in pairs)
    judge = Counter(j for _, j in pairs)
    p_e = sum((human[k] / n) * (judge[k] / n) for k in LABELS)
    if abs(1 - p_e) < 1e-12:  # both raters used one label only -> undefined
        return None
    return (p_o - p_e) / (1 - p_e)


def agreement(pairs: list[tuple[str, str]]) -> float:
    return sum(1 for h, j in pairs if h == j) / len(pairs) if pairs else 0.0


def bootstrap_ci(
    pairs: list[tuple[str, str]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile 95% CI for kappa via resampling. None if too few non-degenerate draws."""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    ks = []
    for _ in range(n_boot):
        sample = [rng.choice(pairs) for _ in pairs]
        k = kappa(sample)
        if k is not None:
            ks.append(k)
    if len(ks) < n_boot * 0.5:  # mostly degenerate -> CI not meaningful
        return None
    ks.sort()
    lo = ks[int(0.025 * len(ks))]
    hi = ks[int(0.975 * len(ks)) - 1]
    return lo, hi


def band(k: float) -> str:
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


def _fmt_k(k: float | None, ci: tuple[float, float] | None) -> str:
    if k is None:
        return "undefined (degenerate: one label only)"
    s = f"{k:.3f} ({band(k)})"
    if ci is not None:
        s += f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    return s


def main() -> int:
    human = {r["id"]: r["human_label"] for r in _read_jsonl(LABELS_PATH)}
    verdicts = {r["id"]: r for r in _read_jsonl(VERDICTS_PATH)}
    ids = sorted(set(human) & set(verdicts))
    if not ids:
        print("no overlap between human labels and judge verdicts yet.")
        print(f"  human-labeled: {len(human)} | judged: {len(verdicts)}")
        return 1

    pairs = [(human[i], verdicts[i]["judge_label"]) for i in ids]
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for i in ids:
        by_cat.setdefault(verdicts[i]["category"], []).append(
            (human[i], verdicts[i]["judge_label"])
        )

    overall_k = kappa(pairs)
    overall_ci = bootstrap_ci(pairs)

    lines: list[str] = []
    out = lines.append
    out("# Phase 3.4 — Judge calibration report\n")
    out(f"Items with both a human label and a judge verdict: **{len(pairs)}**\n")
    out(f"**Overall Cohen's κ:** {_fmt_k(overall_k, overall_ci)}")
    out(f"**Raw agreement:** {agreement(pairs):.1%}\n")

    out("## Per category\n")
    out("| category | n | κ | raw agreement |")
    out("|---|---|---|---|")
    report_cats = {}
    for cat in sorted(by_cat):
        cp = by_cat[cat]
        k = kappa(cp)
        ci = bootstrap_ci(cp)
        note = " _(deterministic — expect ~1.0)_" if cat in DETERMINISTIC else ""
        out(f"| {cat}{note} | {len(cp)} | {_fmt_k(k, ci)} | {agreement(cp):.1%} |")
        report_cats[cat] = {
            "n": len(cp),
            "kappa": k,
            "ci": list(ci) if ci else None,
            "agreement": agreement(cp),
        }

    # 2x2 confusion (rows = human, cols = judge)
    conf = Counter((h, j) for h, j in pairs)
    out("\n## Confusion (rows = human, cols = judge)\n")
    out("| human ↓ / judge → | pass | fail |")
    out("|---|---|---|")
    for h in LABELS:
        out(f"| **{h}** | {conf[(h, 'pass')]} | {conf[(h, 'fail')]} |")

    # disagreements — the actionable part
    disagree = [i for i in ids if human[i] != verdicts[i]["judge_label"]]
    out(f"\n## Disagreements ({len(disagree)})\n")
    if not disagree:
        out("_None — judge matched human on every labeled item._")
    for i in disagree:
        v = verdicts[i]
        out(f"- **{i}** ({v['category']}): human=`{human[i]}` judge=`{v['judge_label']}` "
            f"[{v['method']}, score {v['score']:.2f}]")
        out(f"  - judge reason: {v['reason'][:200]}")

    report_md = "\n".join(lines) + "\n"
    (CAL / "report.md").write_text(report_md)
    (CAL / "report.json").write_text(json.dumps({
        "n": len(pairs),
        "overall_kappa": overall_k,
        "overall_ci": overall_ci,
        "overall_agreement": agreement(pairs),
        "per_category": {
            c: {**d, "ci": list(d["ci"]) if d["ci"] else None}
            for c, d in report_cats.items()
        },
        "disagreement_ids": disagree,
    }, indent=2) + "\n")

    print(report_md)
    print(f"written -> {CAL/'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
