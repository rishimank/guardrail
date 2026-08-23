#!/usr/bin/env python
"""gate_selftest.py — make the gate say NO on purpose, and fail if it says yes.

THE PROBLEM THIS SOLVES
Every other check in CI proves something *works*. This one proves something *refuses*.
A gate that has only ever returned 0 is indistinguishable from a gate that cannot
return anything else — a `sys.exit(0)` at the top of gate.py would leave every build
green forever and every downstream claim in this repo false. Nothing else in the
pipeline would notice. The only defence is to hand the gate an input that MUST be
rejected and treat acceptance as the failure.

This is the repo-level twin of steps 6 and 7 in scripts/docker_smoke.sh, run natively
so it does not need Docker, and it is why the resume claim can say the build fails on a
regression rather than that it is configured to.

FOUR CASES, AND WHY EACH IS A SEPARATE ONE

    clean run          -> 0   the gate can still pass; without this the whole thing
                              could be a stuck "always fail" and look equally green
    seeded regression  -> 1   the load-bearing case: a real, large, unmistakable
                              degradation must be caught
    missing run        -> 2   an operational failure is NOT a caught regression. If
                              this collapsed into 1, the reflex response to the red
                              build — question the threshold — would be applied to a
                              problem that has nothing to do with thresholds
    profile vs itself  -> 2   a comparison that is arithmetically guaranteed to pass
                              is a broken check, not a passing one

FREE, OFFLINE, DETERMINISTIC. Copies some JSONL, shells out to gate.py, reads exit
codes. No model, no judge, no network.

    venv/bin/python scripts/gate_selftest.py --run runs/mock --profile mock

PSEUDOCODE
    1. Load the clean run's verdicts; pick the category with the most rows.
    2. Write a regressed copy into a temp dir with every row in that category failed.
    3. Run gate.py four times (clean / regressed / missing / self-comparison) and
       compare each exit code against the one the case requires.
    4. Print a result line per case; exit 0 only if all four match.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "gate.py"

OK, BAD = "\033[32mok\033[0m", "\033[31mFAIL\033[0m"


def _run_gate(*args: str) -> int:
    """Invoke the real gate as a subprocess and return its exit code.

    A subprocess, not an import, on purpose: the exit code IS the contract CI consumes,
    and calling evaluate_gate() directly would test the decision logic while skipping
    the layer that translates a decision into a red build.
    """
    proc = subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return proc.returncode


def _seed_regression(source: Path, dest: Path) -> tuple[str, int]:
    """Write a copy of `source`'s verdicts with one whole category failed.

    The largest category is chosen so the seeded move is far outside any per-category
    tolerance — this test must fail for the reason it claims to (a caught regression),
    never because it happened to land near a threshold.
    """
    rows = [json.loads(ln) for ln in source.read_text().splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"{source} is empty")

    target = Counter(r["category"] for r in rows).most_common(1)[0][0]
    flipped = 0
    for row in rows:
        if row["category"] == target and row["passed"]:
            row["passed"] = False
            row["score"] = 0.0
            row["reason"] = "seeded by gate_selftest.py — this run is deliberately bad"
            flipped += 1

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "verdicts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return target, flipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="a CLEAN run directory to derive cases from")
    ap.add_argument("--profile", required=True, help="baseline profile the clean run matches")
    args = ap.parse_args()

    run_dir = Path(args.run)
    verdicts = run_dir / "verdicts.jsonl"
    if not verdicts.exists():
        print(f"gate_selftest: no verdicts at {verdicts}", file=sys.stderr)
        return 1

    print(f"gate self-test | clean run: {run_dir} | baseline profile: {args.profile}\n")

    with tempfile.TemporaryDirectory() as tmp:
        regressed = Path(tmp) / "regressed"
        category, flipped = _seed_regression(verdicts, regressed)
        print(f"  seeded: {flipped} '{category}' verdicts flipped to failures\n")

        cases: list[tuple[str, int, list[str]]] = [
            ("clean run passes", 0, ["--run", str(run_dir), "--profile", args.profile]),
            (
                f"seeded {category} regression is caught",
                1,
                ["--run", str(regressed), "--profile", args.profile],
            ),
            (
                "missing run is 'broken', not 'regressed'",
                2,
                ["--run", str(Path(tmp) / "does-not-exist"), "--profile", args.profile],
            ),
            (
                "a profile compared to itself is refused",
                2,
                ["--run-profile", args.profile, "--profile", args.profile],
            ),
        ]

        failures = 0
        for label, expected, argv in cases:
            actual = _run_gate(*argv)
            good = actual == expected
            failures += not good
            mark = OK if good else BAD
            print(f"  {mark}  exit {actual} (want {expected})  {label}")

    print()
    if failures:
        print(
            f"gate self-test FAILED: {failures} of {len(cases)} cases wrong.\n"
            "The gate is not behaving as a gate — do not trust any green build until "
            "this passes.",
            file=sys.stderr,
        )
        return 1

    print("gate self-test passed — the gate accepts, rejects, and reports breakage distinctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
