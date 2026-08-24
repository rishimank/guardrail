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
- ⚠️ **CORRECTED 2026-08-21, after Phase 7.** "Explain after" was applied too literally on a
  deep-dive phase: Phase 7 was built end-to-end and then summarised, and the owner's feedback was
  *"I want shipping speed AND understanding of the phase — you just built everything with no
  explanation."* A post-hoc summary of decisions already made is not teaching; the owner cannot
  defend a design they only saw justified once it was final. **On Phases 8 and 9 the order is:**
  **(1)** concepts first, before any file is written — what the technology is, the handful of
  terms, what the phase must accomplish (a few minutes' read, not a tutorial); **(2)** the design
  and its forks, with recommendations, so the choice is made knowingly; **(3)** build the whole
  batch in one uninterrupted go — batching does NOT relax; **(4)** walk the actual diff, tying
  each piece back to (1). Speed is unaffected — the teaching *brackets* the build rather than
  replacing it.
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
- **0** — repo, venv (Python 3.13.7), deps, auto-commit hook, repo `rishimank/guardrail` (now public),
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

**Phase 6 COMPLETE (2026-08-19).** `scripts/mine_failures.py` mined TRAIN-split failures →
`training/{train,valid}.jsonl`; LoRA trained via mlx-lm (rank 8, last 8 layers, `--mask-prompt`,
150 iters, **checkpoint 125 selected on VALIDATION loss, never on test**); `LoRASUT` slotted in
behind the existing seam; re-measured on the held-out TEST split only.

**THE REDUCTION (TEST split, n=188, `adapters/v2`@125):** overall **35.1% → 9.0%**,
−26.1 pts, **74.2% relative**. Exact McNemar on 65 discordant pairs **p = 3.16e-10**;
reduction 95% CI **[52.8%, 95.7%]** — always quote the interval, the *size* is loosely pinned.
Per category: hallucination 55.2→3.4, injection 85.7→4.8, pii 90.5→14.3, scope 10.7→7.1,
toxicity 10.0→0.0, **overrefusal 11.6→14.5 (a REGRESSION — the counterbalance fired)**.
Limitation #6 in BENCHMARKS.md records that the overrefusal rubric conflates "refused" with
"answered unhelpfully"; refusal-specific change is 3/69 → 6/69. Deliberately NOT measured on the
full corpus — its train rows are what the training targets were mined from.

**Phase 7 COMPLETE (2026-08-21).** FastAPI service in `src/guardrail/api/`. Endpoints:
`/health`, `/benchmarks`, `/evaluate`, `/gate`, `/runs` (+`/runs/{id}`). Design decisions:
- **The service is a decision + inspection surface, not "run the eval over HTTP"** — a real run
  is ~24 min, so `/runs` returns **202 + an id to poll** via an in-process registry (single-flight:
  one local model, so two concurrent runs would contend for memory and interleave JSONL writes).
- **`/gate` and `/benchmarks` need NO model and NO API key.** That is the load-bearing property
  for Phases 8–9: the container and CI can block a regression without mlx or 1.6 GB of weights.
- **The gate is a pure function** (`api/gate.py`, zero FastAPI imports) so Phase 9 calls it with
  no server: `scripts/gate.py --run runs/<name> [--split test]`, **exit 0 / 1 / 2** (pass /
  regression / gate-broken — collapsing 1 and 2 makes a broken gate look like a caught one).
- **Four rules:** coverage (a category that didn't really run can't be scored — 0/0 is absence of
  evidence, not a perfect score) → regression vs a measured baseline → absolute ceiling (stops
  drift accumulating one tolerated step at a time) → same on the aggregate. **NOTHING NETS**:
  a run that halves injection and doubles overrefusal FAILS. Tested as
  `test_counterbalance_regression_fails_despite_huge_improvement`.
- **Tolerances are sized to what ONE prompt is worth** at each category's test n (toxicity n=20 →
  1 prompt = 5 pts → tolerance 6.0). Tight tolerances are only affordable because Phase 1 chose
  greedy decoding — a sampling harness would need slack wide enough to hide real regressions.
- **`benchmarks/baselines.json`** (counts, generated by `write_benchmarks.py`) vs
  **`benchmarks/gate_policy.json`** (thresholds, hand-edited). MEASURED and CHOSEN are separate
  files so a regenerating script can never quietly move a threshold. `runs/` is gitignored, which
  is why the committed counts file exists at all.
- **Paid grading needs TWO keys turned:** request `use_judge` AND deployment `GUARDRAIL_ALLOW_JUDGE`.
  An ungraded response returns `verdict: null`, **never a default pass**.
- Bug found by its own test and fixed in `gate.py`: the aggregate now compares only categories
  the run and baseline SHARE (`overall_over`), or a run with an extra category would compare two
  different prompt sets — limitation #8's composition trap, live.
- Also: dead `langsmith` dep removed from `pyproject.toml`; free offline `runs/mock` banked
  (170 deterministic rows, 0 failures) as the CI-side baseline profile.
Suite **133 green**, ruff + mypy clean.

**Phase 8 COMPLETE (2026-08-23).** `Dockerfile` (3 stages), `.dockerignore`, `docker-compose.yml`,
`scripts/docker_smoke.sh`, `tests/test_container.py`. Concepts + design forks were presented
BEFORE any file was written (the corrected cadence). Design decisions:
- **One image, two jobs.** `ENTRYPOINT ["python"]` + `CMD` uvicorn, so
  `docker run guardrail:local scripts/gate.py ...` swaps the job with no `--entrypoint`. They
  share ~100% of their dependency closure; two images would let the gate CI runs against drift
  from the image that gets demoed.
- **`python:3.13-slim`.** NOT alpine (musl → no manylinux wheels → scipy compiles from source),
  NOT distroless (no shell/interpreter kills the two-jobs design).
- **`.dockerignore` is an ALLOWLIST** (`*` then `!pyproject.toml !src !scripts !benchmarks
  !tests`). A denylist has to predict every future file; the repo root holds `.env` (live key),
  `venv/`, `models/`, `adapters/`, and the repo is PUBLIC. Docker does not read `.gitignore`.
- **Layer order is the CI budget:** `COPY pyproject.toml` → `pip install` → `COPY src/` →
  `pip install --no-deps --force-reinstall .`. Reversed, every one-char edit re-downloads scipy.
  A stub `src/guardrail/__init__.py` exists only so hatchling can build in the deps layer.
- **`--target test` runs pytest + ruff + mypy INSIDE the image**, against its own resolved
  dependency set. `runtime` does not depend on `test`, so a plain build stays fast; CI invokes
  the test stage explicitly. **136 passed in-image**, ruff + mypy clean.
- Non-root uid 1000; `HEALTHCHECK` on `/health` (free + model-free by Phase 7 design);
  `GUARDRAIL_SUT=mock` / `GUARDRAIL_ALLOW_JUDGE=false` restated as `ENV` so "cannot spend money"
  is inspectable via `docker inspect`, not a claim about the source.
- **Local build is `linux/arm64` (native, fast).** Phase 9 builds `linux/amd64` on the runner —
  that is the real cross-arch parity test, deliberately deferred rather than emulated here.

⚠️ **THREE REAL BUGS THE CONTAINER EXPOSED** (all fixed, all with regression tests):
1. `runner.grade_responses` called `build_metrics()` unconditionally; DeepEval's `AnthropicModel`
   resolves the API key **in its constructor**, so an injection/pii-only run *required* a key it
   would never use. Invisible locally because DeepEval caches a key in `.deepeval/`. This had
   silently broken the "free offline smoke" path that CI depends on. Now built lazily, only if a
   judgment row is actually present. Test: `test_deterministic_only_run_never_builds_the_judge`.
2. `api/settings.REPO_ROOT` walked up from `__file__` — right for editable installs, wrong for a
   real wheel (`site-packages` → `/usr/local/lib/python3.13/benchmarks`). `/benchmarks` 404'd and
   12 tests went red in-image. Now `_repo_root()` **verifies** the guess and falls back to `cwd`.
   Tests: `test_default_benchmarks_dir_resolves_to_a_real_directory`, `..._falls_back_to_cwd_...`.
3. Unpinned dev tools: the image resolved ruff 0.16.4 vs the venv's 0.15.21, and the newer
   default rule set produced **79 findings in unchanged code**. `[dev]` is now pinned `==`
   exactly. **A red build must mean the code regressed, never that a tool released.**

`scripts/docker_smoke.sh` (8 steps, all green) is the actual proof: absence checks
(`.env`/`venv`/`adapters`/`models`/`.git`/`mlx_lm`/`ANTHROPIC_API_KEY`), corpus loads from
site-packages (662), a real 170-prompt MockSUT eval, gate exit **0**, **seeded regression → exit
1**, missing run → exit **2**, then `/health` + `/benchmarks` + `POST /gate` over HTTP.
Suite **141 green** (136 + 5 container tests that skip without a built image).

**Phase 9 COMPLETE (2026-08-23).** `.github/workflows/ci.yml` (3 jobs), `scripts/gate_selftest.py`,
`--run-profile` on `scripts/gate.py`, `.env.example` de-LangSmith'd, `.gitkeep` removed.
Concepts + forks presented before any file was written. Design decisions:
- **Three parallel jobs.** `checks` (pytest/ruff/mypy natively, ~1 min — a syntax error must not
  wait on a Docker build) · `image` (build **linux/amd64**, run the suite INSIDE the image, then
  `docker_smoke.sh --no-build`) · `gate`. Parallel because a lint error and a safety regression
  are independent facts and you want both in one run.
- **$0 and NO SECRETS — a design consequence, not a saving.** Phase 7 made `/gate` model-free and
  Phase 8 proved deterministic grading needs no key, so there is no `ANTHROPIC_API_KEY` in the
  workflow to leak. On a public repo where any fork can open a PR, that removes a whole class of
  exfiltration risk. `GUARDRAIL_ALLOW_JUDGE: "false"` is pinned in `env:` regardless.
- **`permissions: contents: read`** (the default token can write; nothing here needs to) and
  `concurrency` with `cancel-in-progress`.
- **The `image` job is the cross-arch test Phase 8 deferred.** M1 builds arm64; the runner builds
  amd64 from the same Dockerfile, **not emulated** — emulation would prove QEMU works. Docker
  layer cache via `type=gha`, which is what makes Phase 8's `COPY pyproject.toml`-first ordering
  pay off *between* runs.
- **NEW: `gate.py --run-profile`** gates one COMMITTED profile against another, with no run dir.
  `runs/` is gitignored so a checkout has nothing banked — but `baselines.json` is committed,
  which is why it exists. CI runs `--run-profile lora-v2-ck125 --profile mlx-test`: does the
  fine-tune still beat the base model within tolerance and meet every ceiling? Goes red if
  someone loosens a threshold or re-banks a worse run. Comparing a profile to itself is **exit 2**
  — a check that cannot fail is broken, not passing.
- **(a) vs (b) are never conflated.** (a) committed baselines = REAL Qwen numbers, TEST split,
  n=188. (b) live MockSUT eval over 170 deterministic prompts = the PIPELINE works on a clean
  machine, NOT a real-model result. Step names, logs and the `$GITHUB_STEP_SUMMARY` table all say
  which is which.
- **(c) the negative control** (`gate_selftest.py`): four cases — clean→0, seeded regression→1,
  missing run→2, self-comparison→2. Shells out to `gate.py` rather than importing `evaluate_gate`,
  because the **exit code** is the contract CI consumes. Mirrored in the suite by
  `test_base_model_fails_the_ship_criteria` — the un-tuned model must be REJECTED by the same
  policy the tuned model passes, or the ceilings are vacuous.

⚠️ **THE 0.1-POINT FACT — the most quotable thing in the project.** Check (a) prints
`overrefusal regression 14.5 observed vs 14.6 limit` on every build. The shipped fine-tune
passes the counterbalance check by **one tenth of a percentage point**. The category that exists
so "refuse everything" can't win the violation metric very nearly blocked the model we ship.

⚠️ **A red X is not a closed door.** The workflow makes checks RUN; it does not make them
REQUIRED. Until `checks`/`image`/`gate` are added as required status checks in Settings →
Branches → branch protection for `main`, a human can merge past a failing gate. Mechanism vs
enforcement — say it that way in an interview.

Suite **146 green**, ruff + mypy clean.

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
| 5 | scipy | Baseline measurement + Wilson CIs → `BENCHMARKS.md` | ✅ |
| 6 | mlx-lm LoRA | Fine-tune on mined failures → **74.2% reduction, p≈1e-10** | ✅ |
| **7** | **FastAPI** | **Eval pipeline as a service + `/gate`** | ✅ **deep-dive** |
| **8** | **Docker** | **Containerize (linux, MockSUT, no mlx)** | ✅ **deep-dive** |
| **9** | **GitHub Actions** | **CI + eval gate that fails the build on regression** | ✅ **deep-dive** |

**Ship-fast batching plan (2026-08-18).** Six sittings to done — ~~1~~ ~~2~~ ~~3~~ ~~4~~ done:
1. ~~**5a–5d**~~ ✅ 2. ~~**6a–6c**~~ ✅ 3. ~~**6d–6e**~~ ✅ 4. ~~**Phase 7** — FastAPI + `/gate`~~ ✅
5. ~~**Phase 8** — Docker~~ ✅ 6. ~~**Phase 9** — GitHub Actions~~ ✅ — **all six sittings done.**

**REMAINING (not phases, but the project isn't "shipped" without them):**
- **Push.** 25+ commits were unpushed as of Phase 9; CI that has never run is not evidence.
- **Branch protection** on `main` — see the red-X warning above.
- **The regressing-PR demo** — open a PR that genuinely regresses the corpus, watch CI reject it,
  record the PR URL in README, close it. That is what turns "verified by deliberately regressing
  a PR" into a fact with a link. **Note: the link goes in README, NOT BENCHMARKS.md** — that file
  is generated by `write_benchmarks.py` and must never be hand-edited.

**Open decision for Phase 8/9 (raised 2026-08-21).** The corpus DOES ship in the wheel (see the
corpus-location correction below), so CI can run a real, free, offline eval — but only over the
**deterministic categories** (injection + pii, 170 prompts, graded by substring, no API key). The
four judgment categories need paid Haiku. So the CI gate has two possible shapes: (a) gate the
committed `baselines.json` counts, which are real Qwen numbers but only change when someone
re-banks a run, or (b) run MockSUT over the deterministic categories live each build, which is a
genuine end-to-end pipeline proof but on a mock model. **Both are honest; neither alone is the
whole claim.** Recommend doing both and saying which is which — never describing (b) as if a real
model ran in CI.
✅ **(b) is de-risked as of Phase 8:** the live MockSUT run over the 170 deterministic prompts is
proven to work in a container with no key and no weights (`docker compose run --rm eval` → 170
verdicts, all `method: deterministic`), so Phase 9 can do both. Wording discipline stands: (b)
proves the PIPELINE is portable, not that a real model was measured on Linux.

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
  ⚠️ **Corpus location — corrected 2026-08-21.** This file used to say the corpus was at a
  repo-root `data/` that "has never been committed". Both halves were wrong. The corpus lives
  **inside the package** at `src/guardrail/dataset/data/` (`loader.DATASET_DIR`), and all 662
  rows **are tracked in git** and are therefore public. Consequence worth knowing: the corpus
  ships in the wheel, so the Docker image and CI runner get all 662 prompts with no data volume
  and no download — which is what makes a free offline eval possible in CI at all.
- **Auto-commit hook** (`.claude/settings.json`): commits after every Write/Edit, but
  **never pushes**. Pushing is manual — typically once per completed phase, and only when
  the owner OKs it.
- ⚠️ The phased-plan file `~/.claude/plans/i-am-wanting-to-federated-sun.md` **no longer exists**
  (the whole `~/.claude/plans/` dir is empty as of 2026-08-21). This file plus `BENCHMARKS.md`
  are now the plan of record.

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
- **Serve** (free/offline by default; open `/docs`): `venv/bin/python scripts/serve.py`
- **Gate a run** (free, offline, no server — the Phase 9 command; exit 0/1/2):
  `venv/bin/python scripts/gate.py --run runs/lora-v2-ck125 --profile lora-v2-ck125`
- **Container** (Phase 8; all free + offline):
  `docker build -t guardrail:local .` · `docker compose up api` ·
  `docker compose run --rm eval` · `docker compose run --rm gate` ·
  `scripts/docker_smoke.sh` (build + 8 assertions) · `docker build --target test .`
- **Regenerate the numbers** (BENCHMARKS.md *and* `benchmarks/baselines.json`, one command —
  they must never drift):
  `venv/bin/python scripts/write_benchmarks.py --baseline mlx --tuned lora-v2-ck125 --adapter adapters/v2 --checkpoint 125`
