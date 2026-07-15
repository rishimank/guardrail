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
`fleet-report-rag`. 0.6 done (2026-07-15): private repo `rishimank/guardrail` created and
`main` pushed. 0.4 + 0.5 done (2026-07-15): `.env` populated; both keys verified live —
`claude-haiku-4-5` returned a real completion (confirming the judge model id) and LangSmith
authenticated. **Phase 0 complete.**

**Phase 1 — in progress.** 1.1 done (2026-07-15): `mlx-lm` 0.31.3 / `mlx` 0.32.0 installed,
Qwen2.5-3B-Instruct-4bit downloaded (1.6 GB, HF cache), generation confirmed. Measured
**~71 tok/s** steady-state (3 runs, 69.6–72.0, warm-up discarded) → a 500-prompt run is
**~24 min** serially. Output appeared deterministic across runs (identical token counts) —
`mlx-lm` likely defaults to greedy decoding; verify and make temperature explicit in 1.2.
1.2 done: `sut/base.py` — `SUT` Protocol (structural, not inheritance) + frozen `Response`
(text, model_id, latency_s, prompt_tokens, completion_tokens). Greedy (temp 0.0) is the
documented default: reproducibility is what makes a measured delta attributable.
1.3 done: `sut/mock.py` (`MockSUT`, canned/offline/free) + `sut/mlx_sut.py` (`MLXSUT`, real
Qwen, lazy mlx import so linux CI can still import the package). Both pass
`isinstance(x, SUT)`; both return "Paris".
Also fixed: `[tool.mypy] python_version` was **3.11** on a 3.13 venv — mypy died on a numpy
stub and checked **0 files**. Now 3.13, checks 12, and immediately caught a real unpack bug.
1.4 done: `sut/__init__.py` exposes `get_sut()` (reads `$GUARDRAIL_SUT`, defaults **mock** —
a wrong default must cost $0 and download nothing) + `scripts/ask.py`. **Phase 1 gate passes:**
`scripts/ask.py "..."` returns a real generation; `--sut mock|mlx` swaps the model with no
code change; `lora` fails with an actionable message + exit 1.
`ask.py` prints tok/s **only** above 20 output tokens — below that, tokens/latency measures
startup overhead, not speed (a 1-token reply reported "2.0 tok/s" on a ~71 tok/s model).
**Next:** tests for the SUT seam (all against `MockSUT`), then Phase 2.

⚠️ **`mlx-lm` is 0.31.x, not 0.20.x.** Verify API against the installed package, not blogs.
Confirmed by inspection: `load(path, adapter_path=...)` — `adapter_path` is the Phase 6 seam;
`generate(model, tokenizer, prompt, **kw) -> str` is stateless.

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
