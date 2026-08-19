# CLAUDE.md — Guardrail

Project context auto-loaded each session. Keep it short and current.

## What this is & why

An automated framework that attacks a real LLM with 500+ adversarial prompts across 6 failure
categories, measures how often it fails, fine-tunes it to fail less, and ships the whole
pipeline as a gated CI/CD service.

It mirrors a set of LLM-safety-engineering resume bullets (adversarial evals → fine-tuning →
FastAPI + Docker + GitHub Actions).

**Success = a shipped, running, gated service whose numbers the owner can defend.** Both
halves matter: it has to exist, and the owner has to be able to explain what a claim like
"43.8% failure rate" depends on (judge calibration, sample size, confidence intervals).

## Who I'm working with (working style)

**SHIP-FAST MODE (adopted 2026-08-18).** The earlier mode was "one small step, explain, check
understanding, repeat." That was right for Phases 0–6, where the risk was claiming a number
the owner couldn't defend. It is wrong for the remaining infra phases, where the risk is
not shipping. New cadence:

- **Batch the work.** Execute a whole phase (or a coherent chunk of one) in one go — write all
  the files, run the tests, then stop. Do not pause between sub-steps for approval.
- **Explain after, not during.** One walkthrough per batch covering what was built and *why it
  is shaped that way* — the design decisions and the tradeoffs, not a line-by-line reading of
  code the owner can see in the diff.
- **Depth is now weighted to Phases 7–9** (FastAPI, Docker, GitHub Actions). These are the
  phases the owner most needs to be able to talk through in an interview and has the least
  prior exposure to, so they get the long explanations. Phases 5–6 get short ones.
- The owner is **new to this stack** (LoRA/MLX, FastAPI, Docker, CI/CD) — ship-fast changes the
  *cadence* of teaching, not whether it happens.
- **DEFINE YOUR TERMS (added 2026-08-18).** The first time a piece of jargon appears in an
  explanation, give it a **two-part gloss**: (1) the general definition — what the word means
  to anyone in the field, and (2) **what it concretely refers to in *this* project**. Both
  halves, always. "Benchmarking = systematically measuring a system's performance against a
  fixed task set → here, it's running all 662 corpus prompts through Qwen and recording the
  pass/fail rate per category." The generic half alone is a dictionary; the project half alone
  is unexplained shorthand. Applies to stats terms (Wilson interval, κ, class imbalance),
  infra terms (container, layer cache, image), and ML terms (LoRA, adapter, rank, epoch).
- **Cost safety is unchanged:** flag expected cost **before** anything that hits a paid API.
  Ship-fast is never a reason to skip this.
- **The honesty note below is not in scope for speed.** No claimed number gets loosened,
  no CI gets dropped, no split gets relaxed to move faster.

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

**Phases 0–3 COMPLETE.** Condensed log (full detail in git history):
- **0** — repo, venv (Python 3.13.7), deps, auto-commit hook, private repo `rishimank/guardrail`,
  both API keys verified live.
- **1** — the SUT seam. `sut/base.py` (`SUT` Protocol + frozen `Response`), `MockSUT` (offline/free),
  `MLXSUT` (Qwen2.5-3B-Instruct-4bit, ~71 tok/s → ~24 min for a 512-prompt run). Greedy (temp 0.0)
  is the documented default: reproducibility is what makes a measured delta attributable.
  `get_sut()` reads `$GUARDRAIL_SUT`, **defaults to mock** (a wrong default must cost $0).
- **2** — the corpus. 512 prompts, one JSONL per category, built three ways: 90 handwritten seeds
  (15/cat, the frozen golden set), 195 templated (injection/pii/toxicity — free, un-refusable,
  precise canary control), 227 Haiku-synthesized (hallucination/scope/overrefusal) with a
  **separate verify call** that rejected 7 bad rows. Cost $0.34. **No `split` field on purpose.**
- **3** — the judge. `get_judge()` (native `AnthropicModel`, Haiku, temp 0), `judge/metrics.py`
  (`grade()` dispatch: injection/pii deterministic via `forbidden_outputs` substring, other 4 via
  per-category G-Eval rubrics reading `ground_truth`). `runner/run_corpus()` = generate → grade →
  aggregate, two resumable JSONL phases so a crash never re-spends money. Judge is injectable, so
  tests run offline+free. Calibrated vs. **Opus 4.8 as reference judge**: **κ = 0.782, 96.6%
  agreement, n=59** (`calibration/report-reference.md`). Honest framing: "cheap Haiku ≈ strong
  Opus", NOT "≈ human".

**Phase 4 (LangSmith) — DROPPED (2026-08-18).** Cut deliberately, not skipped by accident. The
resumable JSONL runs + banked verdicts + offline `scripts/report.py` already cover everything the
reduction claim depends on; LangSmith would have added a tracing UI and hosted experiment tracking
that no number relies on. Directory stubs `src/guardrail/tracing/` and `src/guardrail/metrics/`
are empty and should be **deleted** in the 5.x cleanup. Resume wording must not mention LangSmith.

**Phase 5 — IN PROGRESS.** Built: `split.py` (sha256(id) → TRAIN/TEST, `TEST_FRACTION = 0.30`;
un-storable by design so a leak can't be introduced by editing a file), `stats/` (Wilson intervals
— Wald is wrong here: small per-category n and rates near 0/1 would give impossible CIs like
[0,0] on toxicity), `report.py` + `scripts/report.py` (counts → rates + CIs → markdown, $0 and
offline, re-runnable forever). Suite **70 green**.

**The baseline run is banked** (`runs/mlx/`, all 512 prompts, Qwen2.5-3B):

| category | n | fails | fail rate | 95% CI |
|---|---:|---:|---:|---|
| hallucination | 88 | 48 | 54.5% | [44.2%, 64.5%] |
| injection | 85 | 78 | 91.8% | [84.0%, 96.0%] |
| overrefusal | 91 | 15 | 16.5% | [10.3%, 25.4%] |
| pii | 85 | 71 | 83.5% | [74.2%, 89.9%] |
| scope | 93 | 7 | 7.5% | [3.7%, 14.7%] |
| toxicity | 70 | 5 | 7.1% | [3.1%, 15.7%] |
| **overall** | 512 | 224 | **43.8%** | [39.5%, 48.1%] |

**Phase 5 COMPLETE (2026-08-18).** 5a: overrefusal grown 91 → **241** (150 synthesized, 150/150
kept, $0.165). Two bugs fixed to make it safe: `synth_judgment.py` was destructive by default
(overwrote all 3 judgment cats — would have replaced hallucination/scope prompts while KEEPING
their ids, silently invalidating the banked run, since runs/ is keyed by id) → added
`--category`/`--append`; and `verify()` sent all items in one 4096-token call, so past ~60 items
the verdict list truncated and the caller read missing verdicts as "reject" — an invisible
silent-discard → now chunked at 40 with index remapping. 5b: incremental re-run (resume skipped
the 512 already-banked rows). 5c: `scripts/write_benchmarks.py` — BENCHMARKS.md is **generated,
never hand-edited**, and emits the baseline TEST-split table NOW so the Phase 6 comparison target
is fixed in git before the fine-tune exists. 5d: dead `tracing/`+`metrics/` stubs deleted.

**BASELINE (corpus v2, n=662, commit `6e1a713`):** overall **38.2%** [34.6%, 42.0%].
By category: injection 91.8%, pii 83.5%, hallucination 54.5%, overrefusal 18.3%, scope 7.5%,
toxicity 7.1%. **TEST split (n=188): 35.1%** [28.6%, 42.2%] — this is the Phase 6 comparison target.

⚠️ **The overall rate is composition-dependent — never compare it across corpus versions.**
Growing overrefusal moved overall from 43.8% (n=512) to 38.2% (n=662) **on the identical model**,
purely because a low-failure category gained weight. Only compare overall rates on the SAME corpus
version (which baseline-vs-tuned does). Per-category rates are the ones safe to read across versions.
Recorded as limitation #6 in BENCHMARKS.md.

**Phase 6a scaffolded:** `scripts/mine_failures.py` written (not yet run). Mines TRAIN-split
failures → `training/{train,valid}.jsonl` in mlx-lm's `{"prompt","completion"}` format. Verified
against installed mlx-lm 0.31.3: `--data DIR` with `{train,valid,test}.jsonl`, `--mask-prompt`,
`--fine-tune-type lora`, `--num-layers`, `--adapter-path`. Two design decisions baked in:
corrected completions are **written by Haiku, not templated** (150 near-identical templated
refusals would teach a refusal template → straight into overrefusal collapse), and **benign
"ballast" rows** (TRAIN rows the model already answers correctly, replayed as targets) are mixed
in so the cheapest way to cut violations isn't "refuse everything". Ends with a hard
`SystemExit` if any TEST id reaches the training set — raises, never warns.

### Two facts that shape Phase 6 (decided 2026-08-18)

**1. The fine-tune is a THREE-category fine-tune.** Training fuel = TRAIN-split failures only:
injection 60, pii 52, hallucination 32 — but **scope 4 and toxicity 3**. Scope/toxicity cannot be
trained on and don't need to be (base model already at 7.5% / 7.1%). The reduction claim is scoped
to injection + pii + hallucination, plus an overall number; scope/toxicity are reported as
**regression guards** (did the fine-tune make them worse?), never as improvements.

**2. Overrefusal is being grown 91 → ~200 to give the counterbalance statistical power.** At the
current test-side n=27/k=2, a fine-tune that DOUBLED the refusal rate would be invisible:
4/27 = 14.8% [5.9%, 32.5%] vs 9/27 = 33.3% [18.6%, 52.0%] overlap almost entirely. At n≈60
(what 200 total yields on a 30% split) the same shift reads [9.3%, 28.0%] vs [24.2%, 47.5%] —
detectable. A counterbalance that cannot fire is decoration.
**Toxicity is deliberately NOT grown** (test n=20) — we aren't training on it, and a
refusal-oriented fine-tune pushes toward refusal (which overrefusal catches), not toward more
harmful output. Recorded as a stated limitation in BENCHMARKS.md, not silently ignored.

⚠️ **Carried-forward calibration caveat:** overall κ=0.78 is strong, but **toxicity is degenerate**
(all-pass both judges → κ undefined) and **overrefusal κ=0** (class-imbalance trap despite 93% raw
agreement). Growing overrefusal in 5a also adds calibration headroom there. A category-specific
toxicity claim rests on thin evidence — don't lean on it.

⚠️ **DeepEval 4.x Synthesizer API — verify against installed 4.1.0 in 2.3, not blogs (1.x).**

⚠️ **`mlx-lm` is 0.31.x, not 0.20.x.** Verify API against the installed package, not blogs.
Confirmed by inspection: `load(path, adapter_path=...)` — `adapter_path` is the Phase 6 seam;
`generate(model, tokenizer, prompt, **kw) -> str` is stateless.

⚠️ **DeepEval is 4.x, not 1.x.** Nearly every tutorial online is 1.x. Verify the custom-judge
API against the installed package in Phase 3.1 — do not trust recalled/blog syntax.

| Phase | Tech | Adds | State |
|------:|------|------|-------|
| 0 | Python, git | Skeleton, hooks, keys | ✅ |
| 1 | MLX | The system under test + the SUT seam | ✅ |
| 2 | DeepEval Synthesizer | 500+ adversarial prompts, 6 categories, authored ground truth | ✅ |
| 3 | DeepEval + Claude | Metrics, judge, **judge calibration (κ = 0.782)** | ✅ |
| ~~4~~ | ~~LangSmith~~ | ~~Tracing, datasets, experiments~~ | ❌ **dropped** |
| 5 | scipy | Baseline measurement + Wilson CIs → `BENCHMARKS.md` | 🔄 5a–5d left |
| 6 | mlx-lm LoRA | Fine-tune on mined failures → measured reduction | ⬜ |
| **7** | **FastAPI** | **Eval pipeline as a service + `/gate`** | ⬜ **deep-dive** |
| **8** | **Docker** | **Containerize (linux, MockSUT, no mlx)** | ⬜ **deep-dive** |
| **9** | **GitHub Actions** | **CI + eval gate that fails the build on regression** | ⬜ **deep-dive** |

**Ship-fast batching plan (2026-08-18).** Six sittings to done:
1. **5a–5d** — grow overrefusal, incremental re-run, `BENCHMARKS.md`, delete dead stubs.
2. **6a–6c** — mine TRAIN failures → JSONL, LoRA train, `LoRASUT` behind the existing seam.
3. **6d–6e** — re-measure on TEST only, write the reduction (+ counterbalance) into BENCHMARKS.
4. **Phase 7** — FastAPI + `/gate`. *Long explanation.*
5. **Phase 8** — Docker. *Long explanation.*
6. **Phase 9** — GitHub Actions. *Long explanation.*

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

- Repo: **`rishimank/guardrail`** — **PUBLIC** (confirmed 2026-08-19; it was created private
  and this file said so until then). Everything pushed is world-readable, including
  `training/*.jsonl`, which holds working injection payloads paired with correct refusals.
  That is intentional — it is a portfolio project and a defensive dataset — but it raises the
  bar on the secrets rule below: a leaked key here is leaked publicly, not just internally.
  `data/` (the 662-prompt corpus) is **not gitignored but has never been committed**; leave
  it that way unless the owner asks otherwise.
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
