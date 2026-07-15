# CLAUDE.md — Guardrail

Project context auto-loaded each session. Keep it short and current.

## What this is & why

A **learning project**: build an automated framework that attacks a real LLM with 500+
adversarial prompts across 6 failure categories, measures how often it fails, fine-tunes it
to fail less, and ships the whole pipeline as a gated CI/CD service.

It mirrors a set of LLM-safety-engineering resume bullets (DeepEval + LangSmith evals →
fine-tuning → FastAPI + Docker + GitHub Actions).

**The goal is deep understanding, not shipping speed.** Success = the owner can explain every
piece end-to-end, including what a claim like "23% hallucination rate" actually depends on
(judge calibration, sample size, prompt difficulty, confidence intervals).

## Who I'm working with (working style)

- The owner is **new to this stack** (DeepEval, LangSmith, LoRA/MLX, FastAPI, Docker, CI/CD).
- **Teach as we go:** build → explain *why* it works → suggest a small experiment.
  **Line-by-line walkthroughs, not code dumps. Check understanding before continuing.**
- **Cost safety:** flag expected cost **before** anything that hits a paid API.
- Finish and understand one phase before starting the next.

## The honesty note (read this before touching any number)

The resume bullets this project mirrors say **23%** hallucination and **61%** reduction.
**Those are not targets.** We measure, and whatever we measure is what gets claimed —
same precedent as LeaseLight, where the only number claimed was one actually benchmarked.

`BENCHMARKS.md` is the single source of truth for every claimed number. Nothing goes on a
resume that isn't in that file with its method, sample size, confidence interval, and commit SHA.

Two things protect the reduction claim specifically:
- **Overrefusal is a tracked category (#6).** Without it, the winning "fix" is to refuse
  everything. A violation drop that comes with a refusal spike is a fake win, and we say so.
- **Strict held-out split.** We never train on prompts we measure on. Enforced in code, not
  discipline. If this leaks, the reduction number is meaningless.

## Status

**Phase 0 — in progress.** 0.1 done (2026-07-14): repo scaffolded, venv on Python 3.13.7,
core+dev deps installed and imports verified (deepeval 4.1.0, langsmith 0.10.4,
anthropic 0.116.0, fastapi 0.139.0, scipy 1.18.0). 0.3 done: auto-commit hook ported from
`fleet-report-rag`. **Next:** 0.4 (Anthropic key + billing limit), 0.5 (LangSmith key),
0.6 (create private repo).

⚠️ **DeepEval is 4.x, not 1.x.** Nearly every tutorial online is 1.x. Verify the custom-judge
API against the installed package in Phase 3.1 — do not trust recalled/blog syntax.

| Phase | Tech | Adds |
|------:|------|------|
| 0 | Python, git | Skeleton, hooks, keys |
| 1 | MLX | The system under test + the SUT seam |
| 2 | DeepEval Synthesizer | 500+ adversarial prompts, 6 categories, authored ground truth |
| 3 | DeepEval + Claude | Metrics, judge, **judge calibration (κ)** |
| 4 | LangSmith | Tracing, datasets, experiments |
| 5 | scipy | Baseline measurement + Wilson CIs → `BENCHMARKS.md` |
| 6 | mlx-lm LoRA | Fine-tune on mined failures → measured reduction |
| 7 | FastAPI | Eval pipeline as a service + `/gate` |
| 8 | Docker | Containerize |
| 9 | GitHub Actions | CI + eval gate that **fails the build** on regression |

## Tech stack & key decisions

- **Language:** Python 3.13 (venv at `venv/`).
- **System under test:** `mlx-community/Qwen2.5-3B-Instruct-4bit`, run locally via MLX.
  Chosen because (a) it's fine-tunable — Anthropic offers no public fine-tuning, (b) it's a
  general-purpose assistant, (c) it costs **$0/call**, so the whole budget goes to judging.
- **Why 3B/4-bit:** the machine is an M1 Pro with **16 GB** unified memory. 3B+LoRA fits
  comfortably; 7B thrashes. **Do not upsize without re-checking memory.**
- **Fine-tuning:** LoRA via `mlx-lm`. Unsloth is CUDA-only and cannot run here.
- **Judge:** `claude-haiku-4-5` via a DeepEval custom-model wrapper (DeepEval defaults to
  OpenAI; we override). Chosen to keep a full run ≈ **$1.35**, under the $2 ceiling.
- **`mlx-lm` is an optional extra, never a core dep.** It is Apple-Silicon-only; putting it in
  `dependencies` would break the linux Docker image and GitHub Actions at `pip install`.
  Mac: `pip install -e ".[mlx]"`. CI/Docker: `pip install -e .` + `MockSUT`.

### The six failure categories

| # | Category | Judged by |
|---|---|---|
| 1 | Hallucination | LLM judge vs. authored ground truth |
| 2 | Prompt injection / jailbreak | Deterministic canary check + judge |
| 3 | PII leakage | Regex/deterministic first, judge as backstop |
| 4 | Toxicity / harmful content | LLM judge (G-Eval rubric) |
| 5 | Scope violation | LLM judge |
| 6 | **Overrefusal** | LLM judge — **the counterbalance; see honesty note** |

2 and 3 are deterministic on purpose: free, instant, un-gameable.

### How hallucination is measurable without RAG

A general assistant has no corpus, so we **author ground truth into each prompt**: false-premise
questions (an invented novel/paper/person), unanswerable questions, and known-fact questions.
Ground truth = *"this entity is fictional; correct behavior is to decline."* This is what real
hallucination benchmarks do.

## Architecture — the one seam that matters

`src/guardrail/sut/` defines **one interface** satisfied by three implementations:
`MLXSUT` (real base model), `LoRASUT` (fine-tuned), `MockSUT` (deterministic, offline, free).

This is why Phase 6 is a one-variable change, and why CI can run the full test suite with no
model download and no API calls. Same trick as the Chroma→Pinecone swap in `fleet-report-rag`:
build the seam *before* you need it.

## Repo & workflow

- Repo: **`rishimank/guardrail`** (private).
- **Auto-commit hook** (`.claude/settings.json`): commits after every Write/Edit, but
  **never pushes**. Pushing is manual — typically once per completed phase, and only when
  the owner OKs it.
- Full phased plan: `~/.claude/plans/i-am-wanting-to-federated-sun.md`.

## Guardrails

- **Never commit secrets.** API keys go in `.env` (gitignored). Don't print full keys.
- **Don't push without asking.**
- **Never commit model weights** (`models/`, `adapters/`, `*.safetensors`) — ~1.7 GB.
- Keep `MockSUT` working throughout as the fast, free, offline feedback loop.
- Set a **billing limit** in the Anthropic console before the first paid call.

## Code conventions

- **Top-of-file summary on every file.** Start each code file with a docstring that includes a
  **PSEUDOCODE** block: a short, step-by-step plain-English summary of what the file does and
  why it exists. (Owner preference, applies project-wide.)

## Commands

Run with the venv's Python (no activation needed): `venv/bin/python <script>`.
- Recreate env: `venv/bin/pip install -e ".[dev,mlx]"`  (drop `mlx` off Apple Silicon)
- Tests: `venv/bin/pytest`
- Lint/type: `venv/bin/ruff check .` · `venv/bin/mypy src`
