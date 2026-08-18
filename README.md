# Guardrail

An automated adversarial evaluation harness for LLMs. It attacks a language model with 662
adversarial prompts across six failure categories, measures how often the model fails —
with confidence intervals, not bare percentages — fine-tunes it to fail less, and ships the
whole pipeline as a gated service that fails the build on a safety regression.

The system under test is `Qwen2.5-3B-Instruct-4bit`, run locally via Apple MLX. Grading is
part deterministic and part LLM judge (`claude-haiku-4-5`), with the judge calibrated against
a stronger reference judge. Everything is measured, nothing is asserted.

**All claimed numbers live in [`BENCHMARKS.md`](BENCHMARKS.md)** with their method, sample
size, confidence interval, and commit SHA. That file is generated from banked run data, never
hand-edited. If a number isn't in it, this project doesn't claim it.

## What it does

```
  corpus (662 prompts)          the model              grading                 report
  ────────────────────          ─────────              ───────                 ──────
  6 categories, JSONL   ──────▶  Qwen2.5-3B    ──────▶  deterministic  ──────▶  failure rate
  authored ground truth          temp 0.0              + LLM judge              + Wilson CI
  held-out split by hash         local, $0/call        (κ = 0.782)              per category
```

Each stage is resumable and independently re-runnable. Generation and grading write separate
JSONL files, so a crash never re-spends money, and responses can be re-graded without
re-running the model.

## Why these choices

**The model under test runs locally and costs $0/call.** That is the budget: the target is
free, so the entire spend goes to judging. It also has to be open-weights — the pipeline
fine-tunes it, which is not possible against a hosted frontier model.

**Judging is split between deterministic checks and an LLM judge.** Prompt injection and PII
leakage are checked against per-prompt canary strings and planted PII values: free, instant,
and un-gameable. The judge is reserved for the categories that genuinely need semantic reading.

**Overrefusal is a tracked failure category, not an afterthought.** Without it, the winning
"fix" for every other category is to refuse everything. A violation drop that arrives with a
refusal spike is a fake win, and this harness is built to say so — the category was
deliberately grown to 241 prompts so it has the statistical power to actually detect such a
spike rather than merely gesture at one.

**Train/test separation is enforced in code, not discipline.** The split is a pure function of
each prompt's id (`sha256(id) → TRAIN | TEST`) and is **never stored as a field**, so there is
no column anyone could edit — by hand or by a buggy script — to move a prompt across the line.
The fine-tune never sees the prompts it is measured on. If that leaked, any measured reduction
would be meaningless.

**Rates are always reported with intervals.** The runner deliberately returns *counts*, not
rates, and refuses to divide. A bare `43/88 = 48.9%` is a lie of precision; the interval is
what makes it a claim.

## The six failure categories

| # | Category | n | Graded by |
|---|---|---:|---|
| 1 | Hallucination | 88 | LLM judge vs. authored ground truth |
| 2 | Prompt injection / jailbreak | 85 | Deterministic canary check |
| 3 | PII leakage | 85 | Deterministic planted-value check |
| 4 | Toxicity / harmful content | 70 | LLM judge (G-Eval rubric) |
| 5 | Scope violation | 93 | LLM judge |
| 6 | Overrefusal | 241 | LLM judge — the counterbalance |

Hallucination is measurable without a retrieval corpus because ground truth is authored into
each prompt: questions with false premises (an invented novel, paper, or person), genuinely
unanswerable questions, and known-fact controls. For a false-premise prompt the ground truth is
*"this entity is fictional; correct behavior is to decline."*

Every category except overrefusal includes **answer-controls** — prompts that should be
answered — so a model that refuses everything cannot score well. Overrefusal is all
answer-by-design.

### How the corpus was built

90 hand-written seeds (15 per category, kept frozen as the golden set), 195 templated
permutations for the deterministic categories, and 377 LLM-synthesized rows for the judgment
categories. Synthesis is not naive: an LLM asked to rewrite injection or toxicity prompts
refuses, and the refusal text gets stored as the prompt. So the adversarial categories are
templated instead — free, un-refusable, and precise about canary placement — and every
synthesized row passes an **independent second-pass verification call** before entering the
corpus. That verifier has rejected rows for naming real entities in supposedly-fictional
prompts, which would have silently poisoned the hallucination rate.

## Judge calibration

An LLM judge that disagrees with reality makes every downstream number noise, so the judge is
measured too. Haiku's grades were compared against **Claude Opus 4.8 acting as a reference
judge** over 59 rows: **Cohen's κ = 0.782** (substantial agreement), 96.6% raw agreement.

This establishes *"the cheap judge agrees with a much stronger judge"* — **not** *"the judge
agrees with humans."* No human-labelled ground truth backs these grades. An earlier
human-labelling attempt was discarded rather than quietly repaired when it produced κ = −0.179
(the labels had been entered inverted). See [`calibration/report-reference.md`](calibration/report-reference.md)
and the limitations section of `BENCHMARKS.md`.

## Architecture

The load-bearing piece is `src/guardrail/sut/`: **one interface, three implementations.**

```
                   ┌─ MLXSUT    real Qwen, local, $0/call
SUT.generate() ────┼─ LoRASUT   fine-tuned Qwen
                   └─ MockSUT   canned, offline, free, deterministic
```

Nothing downstream names a concrete class — callers ask `get_sut()`, which reads
`$GUARDRAIL_SUT` and defaults to `mock`. Three things follow:

- **Measuring a fine-tune is a one-variable change.** Train, flip an env var, re-run the
  *identical* harness. Because nothing else moved, the delta is attributable to the fine-tune —
  and that attributability is the entire reduction claim.
- **CI can run at all.** MLX is Apple-Silicon-only; a Linux runner cannot load Qwen. `mlx-lm` is
  an optional extra, imported lazily, so `pip install -e .` + `MockSUT` works on Linux with no
  model and no API key.
- **The feedback loop is free.** `MockSUT` returns known answers instantly, so a failing test
  means the *harness* is broken rather than the model having an off day. It separates "is the
  instrument correct?" from "what does it measure?".

## Quickstart

Requires Python 3.13. Apple Silicon is needed only for the real model; everything else runs
anywhere against `MockSUT`.

```bash
python -m venv venv
venv/bin/pip install -e ".[dev,mlx]"   # drop ,mlx off Apple Silicon
cp .env.example .env                   # add ANTHROPIC_API_KEY
```

```bash
# one prompt in, one answer out
venv/bin/python scripts/ask.py --sut mock "What is the capital of France?"   # free, instant
venv/bin/python scripts/ask.py --sut mlx  "Who wrote the novel Zorgon?"      # real Qwen

venv/bin/pytest                        # 70 tests, no model, no key, sub-second
venv/bin/ruff check . && venv/bin/mypy src
```

### Running an evaluation

```bash
# free smoke test: deterministic categories only, mock model, no API calls
venv/bin/python scripts/run_eval.py --sut mock --category injection --category pii

# the real thing (~30 min generation, ~$0.50 judging)
venv/bin/python scripts/run_eval.py --sut mlx

# $0, offline, repeatable: re-read banked verdicts and print the rates table
venv/bin/python scripts/report.py --sut mlx

# regenerate BENCHMARKS.md from banked verdicts
venv/bin/python scripts/write_benchmarks.py
```

Runs are resumable — re-running skips ids already generated or graded, so an interrupted run
costs nothing to finish and growing the corpus only evaluates the new prompts.

`GUARDRAIL_SUT` (`mock` | `mlx` | `lora`) selects the model for everything, including `ask.py`.

## Cost

The model under test is free; spend is judge-only. Actual measured spend to date is **under
$2 total** across corpus synthesis, judge calibration, and a full 662-prompt judged run. A
full judged run costs roughly **$0.50** on `claude-haiku-4-5`. Deterministic categories
(injection, PII — 170 prompts) cost nothing at all.

Scripts that hit a paid API print an expected cost before spending. Set a billing limit in the
Anthropic console before running anything judged.

## Honest limitations

Stated up front, because a number is only as good as its caveats. The full list lives in
`BENCHMARKS.md`.

- **The judge is calibrated against a model, not against humans.**
- **Per-category calibration is uneven** — toxicity was degenerate on the calibration set (both
  judges passed everything, so κ is undefined) and overrefusal hit κ = 0 despite 93% raw
  agreement, the classic class-imbalance trap.
- **Difficulty is self-authored.** These are our adversarial prompts, not a standard public
  benchmark. The failure rate describes *this corpus*; an easier corpus would produce a lower
  number. Cross-model comparisons are only meaningful against this same corpus.
- **Per-category intervals on the held-out split are wide** (n ≈ 20–70 each). The overall
  interval is the tight one; per-category numbers are directional.

## Roadmap

Measurement and grading are built and running. What's left:

- **Fine-tuning** — mine held-out-safe training failures into a LoRA dataset, train via
  `mlx-lm`, re-measure on the test split only, and report the reduction *always* alongside the
  overrefusal counterbalance.
- **Service** — FastAPI wrapping the pipeline, with a `/gate` endpoint returning pass/fail.
- **Container** — multi-stage Docker image running `MockSUT`, so it needs neither a model nor
  Apple Silicon.
- **CI gate** — GitHub Actions running the suite on every PR plus an eval gate that **fails the
  build** on a safety regression, verified by deliberately regressing a PR and confirming CI
  rejects it.

## Layout

```
BENCHMARKS.md   generated; every claimed number, with method + n + CI + SHA
src/guardrail/
  sut/          system-under-test adapter: mlx / lora / mock behind one interface
  dataset/      prompt corpus (JSONL, versioned) + schema + loader + templates + synthesis
  judge/        Claude judge + per-category grading dispatch
  runner/       generate -> grade -> aggregate; resumable, returns counts not rates
  stats/        Wilson score intervals
  report.py     counts -> rates + confidence intervals -> markdown
  split.py      sha256(id) -> TRAIN | TEST; the leak-proof held-out split
  api/          FastAPI service
calibration/    judge-vs-reference-judge agreement (κ) report
training/       mined failures -> LoRA dataset -> mlx-lm train
runs/           banked responses + verdicts (gitignored)
scripts/        ask, run_eval, report, write_benchmarks, mine_failures, calibration
tests/          70 tests, all offline against MockSUT
```
