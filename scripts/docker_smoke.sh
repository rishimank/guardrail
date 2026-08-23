#!/usr/bin/env bash
# docker_smoke.sh — prove the image actually does the thing it was built to do.
#
# A Dockerfile that builds is not evidence. The claim Phase 8 makes is specific:
#
#     with no mlx, no model weights, no API key, and no repo checkout, this image can
#     run a real eval and BLOCK A REGRESSION — and it distinguishes a caught regression
#     from a broken gate.
#
# So this script asserts exactly that, including the two negative cases that matter far
# more than the happy path: a seeded regression must exit 1, and a missing run must exit
# 2. A gate that only ever returns 0 is indistinguishable from no gate at all, and the
# only way to know which one you have is to make it say no on purpose.
#
# It also asserts what is ABSENT from the image, because the leak modes here are silent:
# a `.env` baked into a layer of a public image is not visible from `docker run`.
#
#     scripts/docker_smoke.sh              # build + all checks
#     scripts/docker_smoke.sh --no-build   # reuse the existing guardrail:local
#
# Everything below is free and offline. No check in this file can spend money.
#
# PSEUDOCODE
#      1. Build the runtime stage as guardrail:local.
#      2. ABSENCE: no .env, no venv/, no adapters/, no mlx in the image.
#      3. PRESENCE: all 662 corpus prompts load from the installed wheel.
#      4. Run a real 170-prompt MockSUT eval into a named volume.
#      5. Gate it against the committed `mock` baseline -> expect exit 0.
#      6. Seed a regression into a copy of those verdicts -> expect exit 1.
#      7. Point the gate at a run that does not exist -> expect exit 2.
#      8. Start the server; assert /health, /benchmarks and POST /gate over HTTP.

set -euo pipefail

IMAGE="guardrail:local"
VOLUME="guardrail-smoke-runs"
CONTAINER="guardrail-smoke-api"
PORT="${PORT:-8123}"
BUILD=1
[[ "${1:-}" == "--no-build" ]] && BUILD=0

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

cleanup() {
  docker rm -f "$CONTAINER"  >/dev/null 2>&1 || true
  docker volume rm "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Run a one-shot container with the runs volume attached. "$@" lands after ENTRYPOINT
# (python), so the first argument is a script path.
drun() { docker run --rm -v "$VOLUME:/app/runs" "$IMAGE" "$@"; }

# ---------------------------------------------------------------- 1. build
if [[ $BUILD -eq 1 ]]; then
  step "1. build (runtime stage)"
  docker build --target runtime -t "$IMAGE" . >/dev/null
  pass "built $IMAGE"
else
  step "1. build (skipped, --no-build)"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "$IMAGE does not exist; drop --no-build"
  pass "reusing $IMAGE"
fi
cleanup
docker volume create "$VOLUME" >/dev/null

# ---------------------------------------------------------------- 2. absence
# These are the checks that a passing build cannot give you. .dockerignore is an
# allowlist precisely so these hold; this is where that is verified rather than trusted.
step "2. what must NOT be in the image"

for forbidden in /app/.env /app/venv /app/adapters /app/models /app/.git; do
  if docker run --rm --entrypoint sh "$IMAGE" -c "[ -e '$forbidden' ]"; then
    fail "$forbidden is present in the image"
  fi
  pass "absent: $forbidden"
done

# mlx-lm is Apple-Silicon-only and an optional extra for exactly this reason. If it ever
# migrates into [dependencies], the image stops building on linux — better to assert the
# absence here, where the message explains why, than to read a pip resolver backtrace.
if docker run --rm --entrypoint sh "$IMAGE" -c "python -c 'import mlx_lm' 2>/dev/null"; then
  fail "mlx_lm is installed — it must stay an optional [mlx] extra"
fi
pass "absent: mlx_lm (image is architecture-portable)"

if docker run --rm --entrypoint sh "$IMAGE" -c "printenv ANTHROPIC_API_KEY >/dev/null 2>&1"; then
  fail "ANTHROPIC_API_KEY is baked into the image"
fi
pass "absent: ANTHROPIC_API_KEY"

# ---------------------------------------------------------------- 3. presence
step "3. the corpus ships inside the wheel"
drun -c "
from guardrail.dataset import load_corpus
from guardrail.dataset.loader import DATASET_DIR
c = load_corpus()
assert len(c) == 662, f'expected 662 prompts, got {len(c)}'
assert 'site-packages' in str(DATASET_DIR), f'corpus not loaded from the wheel: {DATASET_DIR}'
print(f'  corpus: {len(c)} prompts from {DATASET_DIR}')
"
pass "662 prompts, loaded from site-packages (no data volume, no download)"

# ---------------------------------------------------------------- 4. a real run
step "4. a real eval, \$0, offline"
drun scripts/run_eval.py --sut=mock --category=injection --category=pii \
     --out-dir=/app/runs/mock-container >/dev/null
drun -c "
import json, pathlib
rows = [json.loads(l) for l in
        pathlib.Path('/app/runs/mock-container/verdicts.jsonl').read_text().splitlines() if l.strip()]
assert len(rows) == 170, f'expected 170 verdicts, got {len(rows)}'
assert all(r['method'] == 'deterministic' for r in rows), 'a judge was called — this must be free'
print(f'  {len(rows)} verdicts, all deterministic')
"
pass "170 prompts generated and graded, no judge calls"

# ---------------------------------------------------------------- 5. gate PASSES
step "5. gate a clean run -> exit 0"
set +e
drun scripts/gate.py --run=/app/runs/mock-container --profile=mock
code=$?
set -e
[[ $code -eq 0 ]] || fail "expected exit 0, got $code"
pass "clean run passes the gate"

# ---------------------------------------------------------------- 6. gate FAILS
# The load-bearing check. Every injection verdict is flipped to a failure, which is a
# +100 pt move against a baseline of 0.0 — far outside the 5.0 pt tolerance. If this
# returns 0, the gate is decoration and every green build downstream means nothing.
step "6. seed a regression -> exit 1"
drun -c "
import json, pathlib
src = pathlib.Path('/app/runs/mock-container/verdicts.jsonl')
dst = pathlib.Path('/app/runs/regressed'); dst.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
for r in rows:
    if r['category'] == 'injection':
        r['passed'] = False
(dst / 'verdicts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
print(f'  seeded: {sum(1 for r in rows if not r[\"passed\"])} injection failures')
"
set +e
drun scripts/gate.py --run=/app/runs/regressed --profile=mock >/dev/null
code=$?
set -e
[[ $code -eq 1 ]] || fail "expected exit 1 (regression), got $code"
pass "seeded regression is caught, exit 1"

# ---------------------------------------------------------------- 7. gate BROKEN
# 1 and 2 must not collapse. A missing run is an operational failure, and the reflex fix
# for a red build — relax the threshold — is the wrong response to it.
step "7. point the gate at nothing -> exit 2"
set +e
drun scripts/gate.py --run=/app/runs/does-not-exist --profile=mock >/dev/null 2>&1
code=$?
set -e
[[ $code -eq 2 ]] || fail "expected exit 2 (gate broken), got $code"
pass "missing run is exit 2, distinct from a caught regression"

# ---------------------------------------------------------------- 8. the service
step "8. the HTTP service"
docker run -d --name "$CONTAINER" -p "$PORT:8000" -v "$VOLUME:/app/runs" "$IMAGE" >/dev/null

for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

health=$(curl -sf "http://127.0.0.1:$PORT/health") || fail "server never became healthy"
echo "$health" | grep -q '"corpus_size":662' || fail "/health does not report 662 prompts: $health"
echo "$health" | grep -q '"judge_enabled":false' || fail "/health says judging is ENABLED: $health"
pass "/health: 662 prompts, judging disabled, no model loaded"

# The REPO_ROOT trap: with a real wheel install this 404s unless GUARDRAIL_BENCHMARKS_DIR
# is set. It is the one endpoint that proves the container found its committed baselines.
curl -sf "http://127.0.0.1:$PORT/benchmarks" | grep -q 'lora-v2-ck125' \
  || fail "/benchmarks is missing the tuned profile — check GUARDRAIL_BENCHMARKS_DIR"
pass "/benchmarks serves the committed baselines with their Wilson intervals"

# A decision is a successful response: the gate says no with HTTP 200 and passed=false.
# 4xx would mean the REQUEST was wrong, which is a different thing entirely.
gate=$(curl -sf -X POST "http://127.0.0.1:$PORT/gate" \
  -H 'content-type: application/json' \
  -d '{"baseline_profile":"mock","counts":{"injection":{"n":85,"failures":85},"pii":{"n":85,"failures":0}}}') \
  || fail "POST /gate did not return 200"
echo "$gate" | grep -q '"passed":false' || fail "POST /gate passed a 100% injection failure rate: $gate"
pass "POST /gate returns 200 with passed=false on a regression"

printf '\n\033[32mall checks passed\033[0m — the image evaluates, gates, and serves with no model, no key, no repo.\n'
