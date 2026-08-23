# syntax=docker/dockerfile:1.7
#
# Dockerfile — Guardrail as a portable, model-free, key-free image (Phase 8).
#
# WHAT THIS IMAGE IS FOR
# Phase 7 made a claim true in code: the gate that blocks a safety regression is a pure
# function over counts, so it needs no model, no Apple Silicon, and no API key. This
# image is the PROOF of that claim. It is a Linux environment with no `mlx-lm`, no 1.6 GB
# of Qwen weights, no `.env`, and (in CI) a different CPU architecture — and the gate
# still returns the right PASS/FAIL. If the container can do it, the Phase 9 runner can.
#
# ONE IMAGE, TWO JOBS. `ENTRYPOINT ["python"]` with a `CMD` that defaults to uvicorn:
#     docker run -p 8000:8000 guardrail:local                      # the service
#     docker run guardrail:local scripts/gate.py --run ... --profile mock   # the gate
# Two images would double build time and let the gate CI runs against drift from the
# image that gets demoed. They share ~100% of their dependency closure; keep them one.
#
# THREE STAGES
#     builder   resolves and installs everything into a relocatable /install prefix
#     test      `--target test` runs pytest + ruff + mypy against the IMAGE's resolved
#               dependency set, not a developer's venv. This is what catches
#               "works on my machine" before CI does.
#     runtime   the default target: /install + the two host directories the CLI reads,
#               running as a non-root user. Does not depend on `test`, so a plain
#               `docker build` stays fast; CI invokes the test stage explicitly.
#
# WHY python:3.13-slim
#   * NOT alpine — musl libc, and numpy/scipy publish manylinux wheels, not musl ones,
#     so pip would fall back to compiling scipy from source (needs gfortran, ~10 min).
#   * NOT distroless — no shell and no interpreter to invoke ad hoc, which would kill
#     the "same image also runs the gate as a command" design above.
#
# WHAT IS DELIBERATELY ABSENT
#   * mlx-lm: Apple-Silicon-only, an optional extra in pyproject for exactly this reason.
#     `pip install .` here installs the core deps and nothing else, which is why this
#     builds on linux/amd64 at all.
#   * ANTHROPIC_API_KEY and .env: the image must not be able to spend money. See the
#     GUARDRAIL_ALLOW_JUDGE default below.
#
# PSEUDOCODE
#     1. builder: COPY pyproject.toml -> pip install (deps layer, cached across code
#        edits) -> COPY src/ -> reinstall the package alone with --no-deps.
#     2. test:    copy /install, add [dev] extras and the source tree, run the suite.
#     3. runtime: copy /install, copy benchmarks/ + scripts/, point GUARDRAIL_*_DIR at
#        them, create an unprivileged user, HEALTHCHECK /health, CMD uvicorn.

# =============================================================================
# STAGE 1 — builder
# =============================================================================
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

# --- dependency layer ---------------------------------------------------------
# Only pyproject.toml is copied here, so this ~90-second layer is reused for every
# build in which the dependency set did not change. Copying src/ first instead would
# re-resolve and re-download scipy on every one-character edit to app.py — the single
# biggest determinant of how long the Phase 9 CI job takes.
#
# hatchling needs the package directory to exist to build a wheel at all, so a stub is
# created purely to satisfy the build backend. It is overwritten in the next layer.
COPY pyproject.toml ./
RUN mkdir -p src/guardrail \
 && : > src/guardrail/__init__.py \
 && pip install --prefix=/install .

# --- package layer ------------------------------------------------------------
# --no-deps: the dependency layer above already installed them, and re-resolving here
# would defeat the whole point of the split. --force-reinstall replaces the stub.
COPY src/ ./src/
RUN pip install --prefix=/install --no-deps --force-reinstall .

# =============================================================================
# STAGE 2 — test  (docker build --target test .)
# =============================================================================
# Runs the suite inside the image's own dependency resolution. A green run here means
# something a green run on the Mac does not: that the code works with no mlx installed,
# on Linux, with the corpus loaded from the installed wheel rather than the repo tree.
FROM python:3.13-slim AS test

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.13/site-packages \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    ERROR_REPORTING=NO \
    GUARDRAIL_SUT=mock \
    GUARDRAIL_ALLOW_JUDGE=false

COPY --from=builder /install /install

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY benchmarks/ ./benchmarks/

RUN pip install --prefix=/install \
      "pytest>=8.0" "pytest-asyncio>=0.24" "ruff>=0.6" "mypy>=1.11"

# One RUN so a failure in any of the three fails the build. The suite runs against the
# INSTALLED guardrail on PYTHONPATH, not ./src — `pytest` would otherwise import the
# local tree and the wheel would never be exercised.
RUN python -m pytest -q \
 && ruff check . \
 && mypy src

# =============================================================================
# STAGE 3 — runtime  (the default target)
# =============================================================================
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="guardrail" \
      org.opencontainers.image.description="Adversarial LLM eval harness + regression gate" \
      org.opencontainers.image.source="https://github.com/rishimank/guardrail"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.13/site-packages

# DeepEval writes telemetry/state files next to the working directory on import. Under
# a non-root user that is a permission error at import time — a container that dies
# before reaching any of our code. Opting out is both the fix and the right default for
# a build agent that nobody is watching.
ENV DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    ERROR_REPORTING=NO

# COST SAFETY, RESTATED IN THE IMAGE. Settings already defaults to these, but a default
# inherited three layers down is easy to lose in a refactor and impossible to see in
# `docker inspect`. Written explicitly so that "this image cannot spend money" is an
# inspectable property of the artifact, not a claim about its source.
ENV GUARDRAIL_SUT=mock \
    GUARDRAIL_ALLOW_JUDGE=false

# api/settings.py derives REPO_ROOT from __file__, which is correct for an editable
# install and WRONG for a real wheel: here __file__ lives in site-packages, so REPO_ROOT
# would resolve to /install/lib/python3.13 and /benchmarks would 404. These two
# variables are the supported fix — pydantic-settings reads every field from GUARDRAIL_*,
# which is exactly what settings.py's docstring says the container should do: configure
# through the environment, not a baked config file.
ENV GUARDRAIL_BENCHMARKS_DIR=/app/benchmarks \
    GUARDRAIL_RUNS_DIR=/app/runs

COPY --from=builder /install /install

WORKDIR /app

# The corpus is NOT copied — all 662 prompts are inside the installed wheel at
# /install/lib/python3.13/site-packages/guardrail/dataset/data/. That is what makes a
# free, offline, end-to-end eval possible in CI with no data volume and no download.
# Only the two things that legitimately live outside the package come in:
COPY benchmarks/ /app/benchmarks/
COPY scripts/ /app/scripts/

# Non-root. Root in a container is root on any bind-mounted host directory, and
# `docker-compose.yml` mounts ./runs. /app/runs is created and owned here so a mounted
# or anonymous volume is writable without the container ever needing to chown at start.
RUN useradd --create-home --uid 1000 guardrail \
 && mkdir -p /app/runs \
 && chown -R guardrail:guardrail /app
USER guardrail

EXPOSE 8000

# /health was built in Phase 7 to load nothing — no model, no judge, no corpus scan —
# specifically so it is safe to poll every 30s. A healthcheck that pulls 1.6 GB of
# weights is a liability, not a healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

# ENTRYPOINT is the interpreter, not the server, so `docker run <img> scripts/gate.py`
# swaps the whole job without --entrypoint. 0.0.0.0 (not 127.0.0.1) because the loopback
# inside a container is not reachable from a published port.
ENTRYPOINT ["python"]
CMD ["-m", "uvicorn", "guardrail.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
