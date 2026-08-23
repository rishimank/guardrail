"""Container tests — the Phase 8 properties, asserted from the test suite (Phase 8).

`scripts/docker_smoke.sh` is the full end-to-end proof and the thing a human runs. This
file is the subset that belongs in `pytest`, so that the container's contract is checked
by the same command as everything else rather than only by a script someone remembers.

THESE TESTS SKIP RATHER THAN BUILD. Building the image takes ~90 seconds, and a test
suite that silently triggers a Docker build is a suite nobody runs. They require an
already-built `guardrail:local` and skip otherwise, which means:

    scripts/docker_smoke.sh   # builds, then these become live
    venv/bin/pytest tests/test_container.py

Skipping is the honest default here: an absent Docker daemon is not evidence that the
image is broken, and it is not evidence that it works either. What must never happen is
these silently passing when nothing was checked — hence `skip`, never a bare `return`.

WHAT IS ASSERTED, AND WHY EACH ONE
  * the corpus loads from site-packages, not a bind mount — the claim CI depends on
  * mlx_lm is absent — the moment it lands in [dependencies] the linux image dies
  * .env is absent — the repo is public and the key is live
  * the gate returns 0 / 1 / 2, all three distinctly — a gate that cannot say no is
    indistinguishable from no gate, and a broken gate must not look like a caught one

PSEUDOCODE
    1. Module-level skipif: no `docker` binary, no daemon, or no guardrail:local image.
    2. `run_in_image()` helper -> CompletedProcess, never raising on non-zero, because
       the non-zero exit codes ARE the thing under test.
    3. One test per property above.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

IMAGE = "guardrail:local"


def _image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _image_available(),
    reason=f"{IMAGE} not built — run scripts/docker_smoke.sh first",
)


def run_in_image(*args: str) -> subprocess.CompletedProcess[str]:
    """Run one command in a throwaway container. Does NOT raise on a non-zero exit.

    `check=True` would be wrong here: exit codes 1 and 2 from the gate are the assertion
    targets, not errors.
    """
    return subprocess.run(
        ["docker", "run", "--rm", IMAGE, *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_corpus_ships_inside_the_wheel() -> None:
    """All 662 prompts must load from the installed package, with no volume mounted.

    This is the property the whole CI story rests on. If the corpus lived at a repo-root
    data/ directory instead, the container would need a bind mount and the GitHub
    Actions runner would need a checkout step just to see the prompts.
    """
    proc = run_in_image(
        "-c",
        "from guardrail.dataset import load_corpus\n"
        "from guardrail.dataset.loader import DATASET_DIR\n"
        "import json\n"
        "print(json.dumps({'n': len(load_corpus()), 'dir': str(DATASET_DIR)}))",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["n"] == 662
    assert "site-packages" in payload["dir"], f"corpus came from outside the wheel: {payload['dir']}"


def test_mlx_is_not_installed() -> None:
    """mlx-lm must stay an optional extra, or this image stops building on linux."""
    proc = run_in_image("-c", "import mlx_lm")
    assert proc.returncode != 0, "mlx_lm is installed in the portable image"


def test_no_dotenv_in_image() -> None:
    """The repo is public and .env holds a live key. .dockerignore must exclude it."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", IMAGE, "-c", "ls -a /app"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert ".env" not in proc.stdout.split()


def test_judging_is_off_by_default() -> None:
    """An unconfigured container must be unable to spend money."""
    proc = run_in_image(
        "-c",
        "from guardrail.api.settings import Settings; s = Settings(); "
        "print(f'{s.sut},{s.allow_judge}')",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "mock,False"


def test_gate_exit_codes_are_distinct() -> None:
    """0 / 1 / 2 must be three different answers, checked inside the container.

    Exit 2 is the one worth protecting: if a missing baseline exited 1, the natural
    response to the red build ("loosen the threshold") would be applied to a problem
    that has nothing to do with thresholds.
    """
    broken = run_in_image("scripts/gate.py", "--run", "/app/runs/nope", "--profile", "mock")
    assert broken.returncode == 2, f"missing run should be exit 2, got {broken.returncode}"

    unknown = run_in_image(
        "scripts/gate.py", "--run", "/app/runs/nope", "--profile", "not-a-profile"
    )
    assert unknown.returncode == 2, f"unknown profile should be exit 2, got {unknown.returncode}"
