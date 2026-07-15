#!/usr/bin/env python
"""ask.py — one prompt in, one answer out. The project's fast feedback loop.

This is the Phase 1 verification gate (`scripts/ask.py "hello"` must return a real
generation) and the tool you poke the model with for every phase after: try an
adversarial prompt, see what Qwen actually does, before writing any metric for it.

It names no concrete SUT. It asks the factory, which reads $GUARDRAIL_SUT — so the
same command hits the mock (free, instant) or real Qwen (~71 tok/s) or a Phase 6
LoRA, with no code change. That is the seam, used in anger for the first time.

    scripts/ask.py "What is the capital of France?"            # $GUARDRAIL_SUT
    scripts/ask.py --sut mlx "Who wrote the novel Zorgon?"     # real Qwen
    scripts/ask.py --sut mock "anything"                       # canned, offline

Answer goes to stdout, stats to stderr — so `ask.py ... > out.txt` captures only
the model's words, and the timings stay visible in the terminal.

PSEUDOCODE
    1. load_dotenv() so .env's GUARDRAIL_SUT / SUT_MODEL apply, like every other entrypoint.
    2. Parse args: prompt (required), --sut, --max-tokens, --temperature.
    3. sut = get_sut(args.sut)  -> the ONLY line that decides which model answers.
    4. r = sut.generate(prompt, ...); time the build separately from the call, so
       a slow model load is not misreported as slow generation (the 0.9 tok/s trap).
    5. Print the text to stdout; print model_id / latency / tokens / tok-per-sec to stderr.
    6. Exit 1 with a readable message on a known failure (bad --sut, missing adapter,
       mlx not installed) instead of dumping a traceback at the user.
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from guardrail.sut import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, VALID_SUTS, get_sut

# Below this many output tokens, tokens/latency measures startup overhead rather than
# generation speed, so we print no rate at all. See the note in main().
MIN_TOKENS_FOR_RATE = 20


def main() -> int:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Send one prompt to the system under test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", help="the prompt to send")
    p.add_argument(
        "--sut",
        choices=VALID_SUTS,
        default=None,
        help="override $GUARDRAIL_SUT (default: env, else mock)",
    )
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="0.0 = greedy = reproducible (default)",
    )
    args = p.parse_args()

    try:
        t0 = time.perf_counter()
        sut = get_sut(args.sut)
        build_s = time.perf_counter() - t0

        r = sut.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(r.text)

    stats = (
        f"\n[{r.model_id}] load {build_s:.2f}s · gen {r.latency_s:.2f}s "
        f"· {r.prompt_tokens} in / {r.completion_tokens} out"
    )
    # Only report throughput when there is enough output for it to MEAN throughput.
    # On a short answer the latency is nearly all fixed overhead (prompt processing,
    # first forward pass), so tokens/latency measures startup, not speed: a 1-token
    # reply prints "2 tok/s" on a model that sustains ~71. Printing no number beats
    # printing a wrong one.
    if r.completion_tokens >= MIN_TOKENS_FOR_RATE and r.latency_s > 0:
        stats += f" · {r.completion_tokens / r.latency_s:.1f} tok/s"
    print(stats, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
