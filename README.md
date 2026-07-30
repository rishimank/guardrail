# Guardrail

An automated adversarial evaluation harness for LLMs. It attacks a real language model with
many adversarial prompts across six failure categories, measures how often the model
fails, fine-tunes it to fail less, and ships the whole pipeline as a gated CI/CD service that
fails the build on a safety regression.

The system under test is `Qwen2.5-3B-Instruct-4bit`, run locally via Apple MLX. Grading is
done by an LLM judge (`claude-haiku-4-5`) plus deterministic checks. Everything is measured,
nothing is asserted.

## Why these choices

**The model under test runs locally and costs $0/call.** That is the budget: the target is
free, so the entire spend goes to judging. It also has to be open-weights — the pipeline
fine-tunes it in Phase 6, and that is not possible against a hosted frontier model.

**Judging is split between deterministic checks and an LLM judge.** Prompt injection and PII
leakage are checked with canary strings and regexes: free, instant, and un-gameable. The
judge is reserved for the categories that genuinely need semantic reading.

**Overrefusal is a tracked failure category, not an afterthought.** Without it, the winning
"fix" for every other category is to refuse everything. A violation drop that arrives with a
refusal spike is a fake win, and this harness is built to say so.

**Train/test separation is enforced in code, not discipline.** The fine-tune never sees the
prompts it is measured on. If that leaks, any measured reduction is meaningless.

## The six failure categories

| # | Category | Judged by |
|---|---|---|
| 1 | Hallucination | LLM judge vs. authored ground truth |
| 2 | Prompt injection / jailbreak | Deterministic canary check + judge |
| 3 | PII leakage | Regex/deterministic first, judge as backstop |
| 4 | Toxicity / harmful content | LLM judge (G-Eval rubric) |
| 5 | Scope violation | LLM judge |
| 6 | Overrefusal | LLM judge — the counterbalance |

Hallucination is measurable without a retrieval corpus because ground truth is authored into
each prompt: questions with false premises (an invented novel, paper, or person), genuinely
unanswerable questions, and known-fact questions. For a false-premise prompt the ground truth
is *"this entity is fictional; correct behavior is to decline."*

## Architecture

The load-bearing piece is `src/guardrail/sut/`: **one interface, three implementations.**

```
                   ┌─ MLXSUT    real Qwen, local, $0/call
SUT.generate() ────┼─ LoRASUT   fine-tuned Qwen              (Phase 6)
                   └─ MockSUT   canned, offline, free, deterministic
```

Nothing downstream names a concrete class — callers ask `get_sut()`, which reads
`$GUARDRAIL_SUT`. Three things follow:

- **Phase 6 is a one-variable change.** Fine-tune, flip an env var, re-run the *identical*
  harness. Because nothing else moved, the delta is attributable to the fine-tune — and that
  attributability is the entire reduction claim.
- **CI can run at all.** MLX is Apple-Silicon-only; a Linux runner cannot load Qwen. `mlx-lm`
  is an optional extra and is imported lazily, so `pip install -e .` + `MockSUT` works on
  Linux with no model and no API key.
- **The feedback loop is free.** `MockSUT` returns known answers instantly, so a failing test
  means the *harness* is broken rather than the model having an off day. It separates "is the
  instrument correct?" from "what does it measure?".

## Status

Phases 0 and 1 are complete. Phases 2–9 are **not built yet** — they are the plan, described
below in the future tense.

| Phase | Tech | Adds | State |
|------:|------|------|-------|
| 0 | Python, git | Skeleton, hooks, keys | **Done** |
| 1 | MLX | System under test + the SUT seam | **Done** |
| 2 | DeepEval Synthesizer | 500+ adversarial prompts, 6 categories | Planned |
| 3 | DeepEval + Claude | Metrics, judge, judge calibration (κ) | Planned |
| 4 | LangSmith | Tracing, datasets, experiments | Planned |
| 5 | scipy | Baseline measurement + Wilson CIs | Planned |
| 6 | mlx-lm LoRA | Fine-tune on mined failures | Planned |
| 7 | FastAPI | Eval pipeline as a service + `/gate` | Planned |
| 8 | Docker | Containerize | Planned |
| 9 | GitHub Actions | CI + eval gate that fails the build | Planned |

### Phase 0 — Foundation (done)

Repo scaffolded on Python 3.13.7 with a venv; core and dev dependencies installed and imports
verified (deepeval 4.1.0, langsmith 0.10.5, anthropic 0.116.0, fastapi 0.139.0, scipy 1.18.0).
An auto-commit hook records every edit locally and never pushes. Anthropic and LangSmith keys
are in a gitignored `.env` and were verified live against the real APIs — the judge model id
returned a real completion, which confirms it before Phase 3 depends on it.

Also fixed here: `[tool.mypy] python_version` was `3.11` against a 3.13 venv. mypy failed
parsing a numpy stub and exited *before reaching `src/`* — reporting success while checking
**zero files**. Set to 3.13, it checks 15 files and immediately caught a real bug.

### Phase 1 — System under test (done)

`Qwen2.5-3B-Instruct-4bit` (1.6 GB) runs locally through `mlx-lm`. 3B/4-bit is deliberate:
the host is an M1 Pro with 16 GB unified memory, where 3B + LoRA fits comfortably and 7B
thrashes.

**Measured on M1 Pro / 16 GB:** model load ~1.2 s; **~71 tok/s** steady-state (3 runs,
69.6–72.0, warm-up discarded). That implies a 500-prompt run takes **~24 minutes** serially —
the number that makes the whole feedback loop affordable.

Built in this phase:

| File | Role |
|---|---|
| `sut/base.py` | The contract: `SUT` protocol (structural, no inheritance) + frozen `Response` carrying text, `model_id`, latency, and token counts |
| `sut/mlx_sut.py` | `MLXSUT` — real Qwen; loads once and reuses; lazy `mlx` import; already accepts `adapter_path` |
| `sut/mock.py` | `MockSUT` — canned, offline, deterministic; what CI runs |
| `sut/__init__.py` | `get_sut()` — reads `$GUARDRAIL_SUT`, defaults to `mock` |
| `scripts/ask.py` | One prompt in, one answer out; the project's feedback loop |
| `tests/test_sut.py` | 19 tests, **0.03 s**, no model, no key, no network |

Decisions worth naming:

- **Greedy decoding (temperature 0.0) is the documented default.** Reproducibility is what
  makes a measured delta attributable to a change rather than to sampling noise.
- **`Response` is frozen and carries `model_id`.** A measurement should not be editable
  before the judge reads it, and a LoRA result must never be mistakable for a baseline result
  by filename alone.
- **`get_sut()` defaults to `mock`.** A wrong default should cost $0 and download nothing.
- **`ask.py` prints tok/s only above 20 output tokens.** Below that, tokens ÷ latency measures
  startup overhead rather than speed — a 1-token reply reported "2.0 tok/s" on a model that
  sustains ~71. Printing no number beats printing a wrong one.

### Phases 2–9 (planned)

**2 — Corpus.** 500+ adversarial prompts across the six categories, as versioned JSONL with a
schema (`id`, `category`, `prompt`, `ground_truth`, `expected_behavior`, `severity`, `source`,
`tags`) and authored ground truth. Held-out split enforced in code.

**3 — Metrics + judge.** One DeepEval metric per category; `claude-haiku-4-5` wrapped as a
DeepEval custom model (DeepEval defaults to OpenAI; this overrides it). Includes **judge
calibration**: hand-grade a golden set, compare to the judge, report agreement (κ). If κ is
low, the metric is wrong and every number downstream is noise.

**4 — Tracing.** LangSmith traces, datasets, and experiments wrapped around `generate()`.

**5 — Baseline.** The first real numbers: per-category failure rates with Wilson score
confidence intervals, written to `BENCHMARKS.md` with method, sample size, and commit SHA.

**6 — Fine-tune.** Mine failures from the baseline, build a LoRA training set from the
**training split only**, train via `mlx-lm`, and re-measure. The reduction is reported with
CIs and **always alongside overrefusal**.

**7 — Service.** FastAPI wrapping the pipeline, including a `/gate` endpoint returning
pass/fail.

**8 — Container.** Multi-stage Docker image; runs `MockSUT`, so it needs no model and no
Apple Silicon.

**9 — CI gate.** GitHub Actions running the suite on every PR, with an eval gate that
**fails the build** on a safety regression. Verified by deliberately regressing a PR and
confirming CI rejects it.

## Claimed numbers

`BENCHMARKS.md` is the single source of truth for every claimed number, and it does not exist
yet because no failure rate has been measured yet. Nothing is claimed that is not in that file
with its method, sample size, confidence interval, and commit SHA.

The only measurements that exist today are the engineering ones on this page: ~71 tok/s, ~1.2 s
load, 19 tests in 0.03 s. **No hallucination rate or reduction figure has been measured**, and
none is claimed.

## Quickstart

Requires Python 3.13. Apple Silicon is needed only for the real model; everything else runs
anywhere against `MockSUT`.

```bash
python -m venv venv
venv/bin/pip install -e ".[dev,mlx]"   # drop ,mlx off Apple Silicon
cp .env.example .env                   # add ANTHROPIC_API_KEY / LANGSMITH_API_KEY
```

```bash
# one prompt in, one answer out
venv/bin/python scripts/ask.py --sut mock "What is the capital of France?"   # free, instant
venv/bin/python scripts/ask.py --sut mlx  "Who wrote the novel Zorgon?"      # real Qwen

venv/bin/pytest                        # 19 tests, no model, no key
venv/bin/ruff check . && venv/bin/mypy src
```

`GUARDRAIL_SUT` (`mock` | `mlx` | `lora`) selects the model for everything, including
`ask.py`. It defaults to `mock`.

## Cost

The model under test is free. Spend is judge-only, and a full 500-prompt judged run is
*estimated* at roughly $1.35 on `claude-haiku-4-5` against a $2 ceiling — an estimate, not a
measurement, until Phase 5 runs one and prints the actual cost. Set a billing limit in the
Anthropic console before running anything judged.

## Layout

```
src/guardrail/
  sut/          system-under-test adapter: base / lora / mock behind one interface
  dataset/      prompt corpus (JSONL, versioned) + loader + schema
  judge/        Claude judge wrapped as a DeepEval custom model
  metrics/      one DeepEval metric per failure category
  runner/       orchestration, response cache, concurrency, cost accounting
  stats/        Wilson intervals, bootstrap CIs, regression tests
  tracing/      LangSmith client + run/experiment plumbing
  api/          FastAPI service
training/       failure mining -> LoRA dataset -> mlx-lm train -> fuse
scripts/        ask.py and friends
tests/          all against MockSUT
```

Only `sut/`, `scripts/`, and `tests/` are populated today.
