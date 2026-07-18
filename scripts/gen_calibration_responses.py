#!/usr/bin/env python
"""gen_calibration_responses.py — Step 1 of Phase 3.4 calibration.

Run the BASE Qwen (the real MLXSUT) over the 90 handwritten golden seeds and cache one
response per seed. These responses are what a human (Step 2) and the judge (Step 3)
each label, so Cohen's kappa can measure judge-vs-human agreement.

Why the real model, not the mock: calibration needs a natural SPREAD of good and bad
behavior to have anything to agree/disagree about. The base model on adversarial prompts
gives exactly that — some correct refusals, some hallucinations, some overrefusals.

Local + free, but slow (~71 tok/s => ~10-15 min for 90). Resumable: each response is
appended as it completes and already-done ids are skipped, so an interrupted run picks
up where it left off instead of paying the whole cost again.

PSEUDOCODE
    1. Load the 90 golden seeds (source=handwritten), sorted by id for stable order.
    2. Read any existing responses.jsonl; collect ids already done (resume support).
    3. sut = get_sut("mlx")  — the real base Qwen (greedy, temp 0).
    4. For each not-yet-done seed: generate, append one JSON line with the seed's
       grading fields + the model output + latency/tokens. Flush per line.
    5. Print progress; on finish, report count + total wall time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from guardrail.dataset import load_corpus
from guardrail.dataset.schema import Source
from guardrail.sut import get_sut

OUT_PATH = Path(__file__).resolve().parent.parent / "calibration" / "responses.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(json.loads(line)["id"])
    return ids


def main() -> int:
    seeds = sorted(
        (e for e in load_corpus() if e.source is Source.HANDWRITTEN),
        key=lambda e: e.id,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(OUT_PATH)
    todo = [e for e in seeds if e.id not in done]
    print(f"golden seeds: {len(seeds)} | already done: {len(done)} | to do: {len(todo)}")
    if not todo:
        print("nothing to do — all golden responses cached.")
        return 0

    sut = get_sut("mlx")  # real base Qwen; downloads nothing (cached in Phase 1)
    print(f"SUT: {sut.model_id}\n")

    start = time.monotonic()
    with OUT_PATH.open("a") as f:
        for i, e in enumerate(todo, 1):
            r = sut.generate(e.prompt)
            row = {
                "id": e.id,
                "category": e.category.value,
                "prompt": e.prompt,
                "ground_truth": e.ground_truth,
                "expected_behavior": e.expected_behavior.value,
                "forbidden_outputs": list(e.forbidden_outputs),
                "output": r.text,
                "model_id": r.model_id,
                "latency_s": r.latency_s,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i:2d}/{len(todo)}] {e.id:9s} {r.completion_tokens:4d} tok "
                  f"{r.latency_s:5.1f}s")

    dur = time.monotonic() - start
    print(f"\ndone: {len(todo)} responses in {dur/60:.1f} min -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
