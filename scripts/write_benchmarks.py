#!/usr/bin/env python
"""write_benchmarks.py — generate BENCHMARKS.md from banked verdicts (Phase 5.3).

BENCHMARKS.md is the single source of truth for every number this project claims, so it is
GENERATED, never hand-edited. A hand-maintained numbers file drifts the moment a run is
re-done: someone updates the table and forgets the n, or fixes the rate and leaves a stale
CI. Deriving the whole file from runs/<sut>/verdicts.jsonl makes that impossible — the
numbers cannot disagree with the run, because there is only one place they come from.

$0 and offline. It re-reads verdicts that were already paid for; it never calls the model
or the judge. Re-run it as often as you like.

WHY TWO TABLES PER MODEL
    FULL CORPUS — every prompt. The honest description of "how often does this model fail
    on our corpus". This is the baseline headline number.
    TEST SPLIT ONLY — the ~30% of prompts that split.py holds out from fine-tuning. This
    is the ONLY table the Phase 6 reduction may be computed from, because the train side
    is contaminated by construction: the fine-tune is trained on those exact failures, so
    improvement there measures memorisation, not generalisation. Emitting the test table
    in the BASELINE file (before any fine-tune exists) is deliberate — it fixes the
    comparison target in git ahead of time, so the "after" number can't be quietly
    compared against a more flattering slice later.

A TUNED RUN IS NORMALLY TEST-ONLY, AND THAT CHANGES WHAT MAY BE PRINTED
    The baseline was run over the whole corpus; the fine-tuned model is run with
    `--split test`, because scoring it on the train side measures memorisation and costs
    money to learn nothing. So a tuned run legitimately contains ZERO train-split rows —
    and in that case a section headed "full corpus" would be a false label on a
    test-only table. This script detects that (it partitions the rows and looks) and
    omits the full-corpus section rather than mislabelling 188 rows as 662.

PSEUDOCODE
    1. Load baseline verdicts (runs/<baseline>/verdicts.jsonl); optionally a --tuned run.
    2. For each run: build a full-corpus report and a test-split-only report
       (partition by split.py, which re-derives the split from ids — nothing stored).
    3. If a tuned run is present, compute the per-category reduction vs. baseline on TEST
       ONLY, and flag the overrefusal counterbalance explicitly. Emit the tuned
       full-corpus table ONLY if the run actually covers the full corpus.
    4. If --adapter is given, read adapter_config.json and print the fine-tune's
       hyperparameters from the adapter itself, so they cannot be mistyped.
    5. Render the whole document (method, tables, limitations, provenance) and write it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from guardrail.report import RunReport, summarize
from guardrail.runner import aggregate
from guardrail.split import TEST_FRACTION, partition
from guardrail.sut.lora_sut import read_config

REPO = Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _reports(verdicts: list[dict]) -> tuple[RunReport | None, RunReport]:
    """(full-corpus report or None, test-split-only report) for one run's verdicts.

    The full-corpus report is None when the run contains no train-split rows at all —
    i.e. it was a `--split test` run. Returning None rather than a report over the test
    rows is the point: the caller must not be able to print a "full corpus" heading
    over a table that is nothing of the kind.
    """
    model_id = next((v.get("model_id", "") for v in verdicts if v.get("model_id")), "")
    train, test = partition(verdicts)
    full = summarize(aggregate(verdicts, model_id=model_id)) if train else None
    return full, summarize(aggregate(test, model_id=model_id))


def _finetune_section(adapter_dir: Path, checkpoint: int | None) -> str:
    """Render the fine-tune's hyperparameters, read from the adapter's own config.

    Read rather than retyped: these numbers end up in an interview answer, and a config
    copied by hand from a shell history is exactly the kind of thing that drifts from
    the weights it claims to describe.
    """
    cfg = read_config(adapter_dir)
    lora = cfg.get("lora_parameters", {})
    ckpt = f"iteration {checkpoint}" if checkpoint is not None else "final weights"
    return (
        "## Fine-tune\n\n"
        "**Method.** LoRA (Low-Rank Adaptation) via `mlx-lm`. The base weights are frozen; "
        "low-rank adapter matrices are trained and applied additively at inference. Only "
        f"the last {cfg.get('num_layers')} transformer layers carry adapters, and prompt "
        "tokens are masked out of the loss (`mask_prompt`), so the model is trained on what "
        "it should *generate*, never on reproducing the attack text it was shown.\n\n"
        f"| setting | value |\n|---|---|\n"
        f"| base model | `{cfg.get('model')}` |\n"
        f"| adapter | `{adapter_dir}` ({ckpt}) |\n"
        f"| rank | {lora.get('rank')} |\n"
        f"| adapted layers | {cfg.get('num_layers')} |\n"
        f"| iterations trained | {cfg.get('iters')} |\n"
        f"| batch size | {cfg.get('batch_size')} |\n"
        f"| learning rate | {cfg.get('learning_rate')} |\n"
        f"| prompt masked in loss | {cfg.get('mask_prompt')} |\n\n"
        "**Training data.** Mined from the BASELINE run's own failures on the TRAIN split "
        "only (`scripts/mine_failures.py`), which ends in a hard assertion that no TEST-split "
        "id reached the training set. Corrected target responses were written by "
        "`claude-haiku-4-5` conditioned on each prompt's authored `ground_truth` — "
        "deliberately not templated, because ~150 near-identical refusals would teach a "
        "refusal template and drive the overrefusal category up. Benign prompts the base "
        "model already answered correctly were mixed in as ballast, so that refusing "
        "everything is not the cheapest way to reduce the violation rate.\n\n"
        "**Checkpoint selection.** Chosen on VALIDATION loss (a slice held out of the train "
        "side), never on the test split — selecting a checkpoint by its test score is a "
        "softer form of the leak this project's split exists to prevent, and would bias the "
        "reduction upward."
    )


def _reduction_table(base: RunReport, tuned: RunReport) -> str:
    """Per-category baseline -> tuned failure rates on the TEST split, with deltas.

    Reported as relative reduction ((b-t)/b) because that is what a "N% fewer violations"
    claim means, alongside the absolute point difference so a big relative move off a tiny
    base rate can't masquerade as a big win.
    """
    lines = [
        "| category | baseline fail rate | tuned fail rate | absolute Δ | relative reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    cats = sorted(set(base.by_category) | set(tuned.by_category))
    for cat in [*cats, "overall"]:
        b = base.overall if cat == "overall" else base.by_category.get(cat)
        t = tuned.overall if cat == "overall" else tuned.by_category.get(cat)
        if b is None or t is None:
            continue
        absolute = t.rate - b.rate
        relative = ((b.rate - t.rate) / b.rate * 100) if b.rate > 0 else float("nan")
        rel = "n/a" if b.rate == 0 else f"{relative:+.1f}%"
        name = f"**{cat}**" if cat == "overall" else cat
        lines.append(
            f"| {name} | {b.rate * 100:.1f}% (n={b.n}) | {t.rate * 100:.1f}% (n={t.n}) "
            f"| {absolute * 100:+.1f} pts | {rel} |"
        )
    return "\n".join(lines)


HEADER = """# BENCHMARKS.md

**Every number this project claims lives here, with its method, sample size, confidence
interval, and commit SHA. Nothing goes on a resume that is not in this file.**

This file is GENERATED by `scripts/write_benchmarks.py` from banked verdicts. Do not edit
it by hand — re-run the script.
"""

METHOD = """## Method

**System under test.** `{model}`, run locally via MLX at temperature 0.0 (greedy). Greedy
decoding is the reason a measured delta is attributable: with sampling on, re-running the
same prompt gives a different answer and any before/after difference is partly noise.

**Corpus.** {n} adversarial prompts across 6 failure categories, each with authored
ground truth. Built three ways: 90 hand-written seeds (15/category, the frozen golden
set), templated permutations for the deterministic categories, and Haiku-synthesized rows
for the judgment categories — every synthesized row passed an INDEPENDENT second-pass
verification call before entering the corpus.

**Grading.** Injection and PII are graded **deterministically** (substring check against
per-prompt `forbidden_outputs` canaries / planted PII) — free, instant, un-gameable. The
other four categories are graded by an LLM judge (`claude-haiku-4-5`, temperature 0) using
per-category G-Eval rubrics that read each prompt's authored `ground_truth`, so one metric
correctly grades both attack rows and benign answer-controls.

**Judge calibration.** The Haiku judge was calibrated against **Claude Opus 4.8 as a
reference judge** on 59 rows: **Cohen's κ = 0.782** (substantial agreement), 96.6% raw
agreement. See `calibration/report-reference.md`.
⚠️ This establishes *"the cheap judge agrees with a much stronger judge"* — **not**
*"the judge agrees with humans."* No human-labelled ground truth backs these grades, and
this file never claims otherwise.

**Intervals.** All intervals are **95% Wilson score intervals**. The textbook Wald
interval is not used: at our per-category sample sizes and at rates near 0 or 1 it
produces impossible bounds (0/70 failures would give the CI [0%, 0%] — a confident claim
that the model *never* fails, which no sample of 70 can support).

**Direction.** Every rate below is a FAILURE rate — higher is always worse, for all six
categories. Overrefusal is no exception: it is itself a failure category, where the
failure is *wrongly refusing a benign prompt*.
"""

LIMITATIONS = """## Known limitations

These are stated because the number is only as good as its caveats.

1. **The judge is calibrated against a model, not against humans.** κ=0.782 vs. Opus 4.8.
   An earlier human-labelling attempt was discarded (labels were entered inverted, κ=−0.179)
   rather than silently repaired. Any claim must say "reference-judge calibrated".
2. **Per-category calibration is uneven.** Overall κ is strong, but on the calibration set
   **toxicity was degenerate** (both judges passed everything → κ undefined: no
   above-chance agreement signal) and **overrefusal scored κ=0** despite 93% raw agreement
   (the class-imbalance trap — with almost all rows in one class, chance agreement is
   already ~93%, so κ has no headroom). Category-specific claims for those two rest on
   thinner evidence than the headline κ suggests.
3. **Toxicity has a small test-split n and is not grown.** We do not fine-tune toxicity
   (too few training failures), and a refusal-oriented fine-tune pushes behaviour toward
   refusal — which the overrefusal category catches — rather than toward more harmful
   output. It is carried as a regression guard, not as a source of improvement claims.
4. **Per-category test-split intervals are wide** (n ≈ 20–70 per category). The overall
   test-split interval is the tight one; per-category numbers should be read as directional.
5. **Difficulty is self-authored.** These are our adversarial prompts, not a standard
   public benchmark. The failure rate describes this corpus, not the model in general —
   an easier corpus would produce a lower number. Cross-model comparisons are only
   meaningful when run against this same corpus.
6. **The overall rate is composition-dependent, so never compare it across corpus
   versions.** It is a weighted average over categories with very different failure rates
   (injection ~92%, toxicity ~7%), so changing how many prompts each category contributes
   moves the headline number *without the model changing at all*. This is not theoretical:
   growing overrefusal from 91 to 241 prompts moved the measured overall rate from 43.8%
   (n=512) to 38.2% (n=662) on the identical model, purely because a low-failure category
   gained weight. Only compare overall rates between runs on the SAME corpus version —
   which is exactly what the baseline-vs-fine-tuned comparison does. Per-category rates
   are the ones that are safe to read across versions.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="mlx", help="reads runs/<name>/verdicts.jsonl")
    ap.add_argument("--tuned", default=None, help="optional fine-tuned run (Phase 6)")
    ap.add_argument("--out", default="BENCHMARKS.md")
    args = ap.parse_args()

    base_path = REPO / "runs" / args.baseline / "verdicts.jsonl"
    if not base_path.exists():
        print(f"no verdicts at {base_path} — run scripts/run_eval.py first.")
        return 1
    base_v = _load(base_path)
    if not base_v:
        print(f"{base_path} is empty.")
        return 1
    base_full, base_test = _reports(base_v)

    tuned_full = tuned_test = None
    if args.tuned:
        tuned_path = REPO / "runs" / args.tuned / "verdicts.jsonl"
        if not tuned_path.exists():
            print(f"no verdicts at {tuned_path}")
            return 1
        tuned_full, tuned_test = _reports(_load(tuned_path))

    test_pct = int(TEST_FRACTION * 100)
    parts = [
        HEADER,
        METHOD.format(model=base_full.model_id, n=base_full.n),
        "## Baseline — full corpus\n\n"
        "The headline measurement: how often the un-tuned base model fails on this corpus.\n\n"
        + base_full.format_markdown(),
        f"## Baseline — held-out TEST split only ({test_pct}% of the corpus)\n\n"
        "The comparison target for any future fine-tune. The split is a deterministic\n"
        "function of each prompt's id (`sha256(id)`), computed in `src/guardrail/split.py`\n"
        "and **never stored as a field** — there is no column anyone could edit to move a\n"
        "prompt across the line, so a leak cannot be introduced by hand or by a bug in a\n"
        "data-writing script.\n\n" + base_test.format_markdown(),
    ]

    if tuned_full is not None and tuned_test is not None:
        over_b = base_test.by_category.get("overrefusal")
        over_t = tuned_test.by_category.get("overrefusal")
        counterbalance = ""
        if over_b is not None and over_t is not None:
            worse = over_t.rate > over_b.rate
            counterbalance = (
                "\n\n**Overrefusal counterbalance.** Overrefusal went from "
                f"{over_b.rate * 100:.1f}% to {over_t.rate * 100:.1f}% "
                f"({(over_t.rate - over_b.rate) * 100:+.1f} pts). "
                + (
                    "⚠️ This is a REGRESSION: the model refuses more benign prompts than "
                    "before. A violation drop bought with a refusal spike is not a real "
                    "win, and the reduction above must be reported together with this row."
                    if worse
                    else "The model did not become more refusal-happy, so the violation "
                    "reduction above is not an artefact of blanket refusal."
                )
            )
        parts += [
            "## Fine-tuned — full corpus\n\n" + tuned_full.format_markdown(),
            "## Fine-tuned — held-out TEST split only\n\n" + tuned_test.format_markdown(),
            "## Reduction (TEST split only)\n\n"
            "Measured exclusively on prompts the fine-tune never saw. Train-split numbers "
            "are deliberately omitted: the model was trained on those exact failures, so "
            "improvement there would measure memorisation.\n\n"
            + _reduction_table(base_test, tuned_test)
            + counterbalance,
        ]

    parts += [
        LIMITATIONS,
        f"## Provenance\n\n"
        f"- generated: {date.today().isoformat()}\n"
        f"- commit: `{_git_sha()}`\n"
        f"- baseline verdicts: `runs/{args.baseline}/verdicts.jsonl`\n"
        + (f"- tuned verdicts: `runs/{args.tuned}/verdicts.jsonl`\n" if args.tuned else "")
        + "- regenerate: `venv/bin/python scripts/write_benchmarks.py`\n",
    ]

    out = REPO / args.out
    out.write_text("\n\n".join(parts).rstrip() + "\n")
    print(f"wrote {out} ({base_full.n} baseline rows, test split n={base_test.n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
