#!/usr/bin/env python
"""gate.py — the CI gate as a command. Exit code 0 = ship, 1 = regression, 2 = broken.

This is the command Phase 9's GitHub Actions workflow runs, and it deliberately does NOT
go through HTTP. The decision logic in `guardrail.api.gate` is a pure function, so CI can
call it with no server to start, no port to wait on, and no FastAPI in the dependency
path — one less moving part between a regression and a red build.

    # gate a banked run against the committed baseline for that model
    venv/bin/python scripts/gate.py --run runs/lora-v2-ck125

    # gate only the held-out rows, against a named profile
    venv/bin/python scripts/gate.py --run runs/mlx --split test --profile mlx-test

    # no run directory at all: gate one COMMITTED profile against another. This is the
    # form CI uses, because runs/ is gitignored and a checkout has no banked run — but
    # baselines.json is committed, so the shipped model's real Qwen numbers travel with
    # the repo and can be re-checked against the ship criteria on every PR.
    venv/bin/python scripts/gate.py --run-profile lora-v2-ck125 --profile mlx-test

THREE EXIT CODES, NOT TWO
    0  every check passed
    1  the gate says no — a real regression, block the merge
    2  the gate could not run at all — missing file, unknown profile, corrupt counts
Collapsing 1 and 2 into "non-zero" is the classic CI mistake: a broken gate would then
look exactly like a caught regression, and the reflex fix for a red build ("relax the
threshold") is the wrong response to a missing baselines.json. One blocks a merge, the
other pages whoever owns this — the codes have to distinguish them.

FREE AND OFFLINE. Reads two files and does arithmetic: no model, no judge, no network.

PSEUDOCODE
    1. Parse the run source — exactly one of --run (a run dir or verdicts.jsonl) or
       --run-profile (a committed profile) — plus --profile, --split, --policy, --json.
    2. Load the run's verdicts; optionally keep only the held-out rows; roll up to
       per-category {n, failures}. With --run-profile the counts are read straight from
       baselines.json instead, and comparing a profile to itself is refused (exit 2):
       a check that cannot fail is broken, not passing.
    3. Resolve the baseline profile from benchmarks/baselines.json (explicit name, or
       the sut_defaults entry for $GUARDRAIL_SUT).
    4. evaluate_gate(run, baseline, policy) and print the table.
    5. Exit 0 / 1 / 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from guardrail.api.gate import Counts, GatePolicy, RunCounts, evaluate_gate
from guardrail.split import is_test

REPO = Path(__file__).resolve().parent.parent

EXIT_PASS, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2


def _fail(message: str) -> int:
    print(f"gate: {message}", file=sys.stderr)
    return EXIT_ERROR


def _load_verdicts(target: Path) -> list[dict]:
    path = target / "verdicts.jsonl" if target.is_dir() else target
    if not path.exists():
        raise FileNotFoundError(f"no verdicts at {path}")
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"{path} is empty — an empty run must not be gradeable as a pass")
    return rows


def _to_counts(verdicts: list[dict]) -> RunCounts:
    tally: dict[str, list[int]] = {}
    for v in verdicts:
        row = tally.setdefault(v["category"], [0, 0])
        row[0] += 1
        if not v["passed"]:
            row[1] += 1
    model_id = next((v.get("model_id", "") for v in verdicts if v.get("model_id")), "")
    return RunCounts(
        model_id=model_id,
        by_category={c: Counts(n=n, failures=f) for c, (n, f) in sorted(tally.items())},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="run directory, or a verdicts.jsonl path")
    source.add_argument(
        "--run-profile",
        default=None,
        help=(
            "gate one COMMITTED profile against another, with no run directory at all. "
            "`--run-profile lora-v2-ck125 --profile mlx-test` asks the question the whole "
            "project exists to answer — does the shipped model still beat the base model "
            "and still meet the ship criteria? — from files in git, for $0, offline."
        ),
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="baseline profile in benchmarks/baselines.json; default: the one mapped to $GUARDRAIL_SUT",
    )
    ap.add_argument(
        "--split",
        choices=("all", "test"),
        default="all",
        help="'test' gates only held-out rows — use it whenever the run spans both sides",
    )
    ap.add_argument("--baselines", default="benchmarks/baselines.json")
    ap.add_argument("--policy", default="benchmarks/gate_policy.json")
    ap.add_argument("--json", action="store_true", help="emit the decision as JSON")
    args = ap.parse_args()

    verdicts: list[dict] = []
    if args.run:
        try:
            verdicts = _load_verdicts(Path(args.run))
        except (FileNotFoundError, ValueError) as exc:
            return _fail(str(exc))

        if args.split == "test":
            verdicts = [v for v in verdicts if is_test(v["id"])]
            if not verdicts:
                return _fail("no held-out rows in this run")

    baselines_path = REPO / args.baselines
    if not baselines_path.exists():
        return _fail(
            f"{baselines_path} is missing. Generate it with "
            "`venv/bin/python scripts/write_benchmarks.py`."
        )
    baselines = json.loads(baselines_path.read_text())
    profiles = baselines.get("profiles", {})

    profile = args.profile
    if not profile:
        sut = os.getenv("GUARDRAIL_SUT", "mock")
        profile = baselines.get("sut_defaults", {}).get(sut)
        if not profile:
            return _fail(
                f"no default baseline for GUARDRAIL_SUT={sut!r}. Pass --profile. "
                f"Available: {sorted(profiles)}"
            )
    if profile not in profiles:
        return _fail(f"unknown profile {profile!r}. Available: {sorted(profiles)}")

    if args.run_profile and args.run_profile not in profiles:
        return _fail(f"unknown run profile {args.run_profile!r}. Available: {sorted(profiles)}")
    if args.run_profile and args.run_profile == profile:
        # Gating a profile against itself is arithmetically guaranteed to pass, so a
        # green result would carry no information at all. Refusing is exit 2, not exit
        # 0: a check that cannot fail is a broken check, not a passing one.
        return _fail(
            f"--run-profile and --profile are both {profile!r}; comparing a profile to "
            "itself always passes and therefore proves nothing"
        )

    policy_path = REPO / args.policy
    policy = GatePolicy.load(policy_path) if policy_path.exists() else GatePolicy()

    try:
        run = (
            RunCounts.from_dict(profiles[args.run_profile])
            if args.run_profile
            else _to_counts(verdicts)
        )
        baseline = RunCounts.from_dict(profiles[profile])
        decision = evaluate_gate(run, baseline, policy)
    except ValueError as exc:
        return _fail(f"corrupt counts: {exc}")

    if args.json:
        print(
            json.dumps(
                {
                    "passed": decision.passed,
                    "summary": decision.summary,
                    "baseline_profile": profile,
                    "checks": [
                        {
                            "kind": c.kind,
                            "category": c.category,
                            "passed": c.passed,
                            "observed": c.observed,
                            "limit": None if c.limit != c.limit else c.limit,
                            "detail": c.detail,
                        }
                        for c in decision.checks
                    ],
                },
                indent=2,
            )
        )
    else:
        print(decision.format_table())
        print(f"\nbaseline profile: {profile}  (from {baselines_path.name}, "
              f"commit {baselines.get('commit', '?')})")

    return EXIT_PASS if decision.passed else EXIT_REGRESSION


if __name__ == "__main__":
    sys.exit(main())
