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

import argparse
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
    # Two raters, joined by id. Rater A is the "ground truth" being compared against
    # (human labels by default, or the reference judge's verdicts); rater B is always
    # the production Haiku judge. Defaults reproduce the original human-vs-judge run;
    # the reference-judge mode just repoints rater A and renames the columns.
    parser = argparse.ArgumentParser(description="Cohen's kappa between two raters.")
    parser.add_argument("--rater-a-path", default=str(LABELS_PATH))
    parser.add_argument("--rater-a-field", default="human_label")
    parser.add_argument("--rater-a-name", default="human")
    parser.add_argument("--rater-b-name", default="judge")
    parser.add_argument(
        "--out-prefix",
        default="report",
        help="write <prefix>.md and <prefix>.json in calibration/ (default: report)",
    )
    args = parser.parse_args()
    name_a, name_b = args.rater_a_name, args.rater_b_name

    rater_a = {r["id"]: r[args.rater_a_field] for r in _read_jsonl(Path(args.rater_a_path))}
    verdicts = {r["id"]: r for r in _read_jsonl(VERDICTS_PATH)}
    overlap = sorted(set(rater_a) & set(verdicts))
    # A rater can produce a non-verdict (e.g. the reference judge refuses to grade a
    # harmful prompt, recorded as "refused"). Those rows are missing data, not a
    # pass/fail, so kappa is computed only over rows both raters actually scored.
    ids = [
        i for i in overlap
        if rater_a[i] in LABELS and verdicts[i]["judge_label"] in LABELS
    ]
    excluded = [i for i in overlap if i not in set(ids)]
    if not ids:
        print(f"no scorable overlap between {name_a} and {name_b} yet.")
        print(f"  {name_a}: {len(rater_a)} | {name_b}: {len(verdicts)} | "
              f"overlap: {len(overlap)} | excluded (non-verdict): {len(excluded)}")
        return 1

    pairs = [(rater_a[i], verdicts[i]["judge_label"]) for i in ids]
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for i in ids:
        by_cat.setdefault(verdicts[i]["category"], []).append(
            (rater_a[i], verdicts[i]["judge_label"])
        )

    overall_k = kappa(pairs)
    overall_ci = bootstrap_ci(pairs)

    lines: list[str] = []
    out = lines.append
    out("# Phase 3.4 — Judge calibration report\n")
    out(f"Raters: **{name_a}** vs **{name_b}**")
    out(f"Items with a label from both: **{len(pairs)}**\n")
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

    # 2x2 confusion (rows = rater A, cols = rater B)
    conf = Counter((a, b) for a, b in pairs)
    out(f"\n## Confusion (rows = {name_a}, cols = {name_b})\n")
    out(f"| {name_a} ↓ / {name_b} → | pass | fail |")
    out("|---|---|---|")
    for a in LABELS:
        out(f"| **{a}** | {conf[(a, 'pass')]} | {conf[(a, 'fail')]} |")

    # disagreements — the actionable part
    disagree = [i for i in ids if rater_a[i] != verdicts[i]["judge_label"]]
    out(f"\n## Disagreements ({len(disagree)})\n")
    if not disagree:
        out(f"_None — {name_b} matched {name_a} on every item._")
    for i in disagree:
        v = verdicts[i]
        out(f"- **{i}** ({v['category']}): {name_a}=`{rater_a[i]}` "
            f"{name_b}=`{v['judge_label']}` [{v['method']}, score {v['score']:.2f}]")
        out(f"  - {name_b} reason: {v['reason'][:200]}")

    report_md = "\n".join(lines) + "\n"
    md_path = CAL / f"{args.out_prefix}.md"
    json_path = CAL / f"{args.out_prefix}.json"
    md_path.write_text(report_md)
    json_path.write_text(json.dumps({
        "rater_a": name_a,
        "rater_b": name_b,
        "n": len(pairs),
        "overall_kappa": overall_k,
        "overall_ci": overall_ci,
        "overall_agreement": agreement(pairs),
        "per_category": report_cats,
        "disagreement_ids": disagree,
    }, indent=2) + "\n")

    print(report_md)
    print(f"written -> {md_path} and {json_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
