#!/usr/bin/env python
"""serve.py — run the Guardrail API locally.

    venv/bin/python scripts/serve.py                  # mock SUT, free, offline
    GUARDRAIL_SUT=lora venv/bin/python scripts/serve.py --reload

Then open http://127.0.0.1:8000/docs — FastAPI generates that page from the type hints
in `schemas.py`, so it is always in sync with what the service actually accepts. It is
the fastest way to see the whole API and to fire a request without writing a client.

WHY A SCRIPT RATHER THAN A BARE `uvicorn` COMMAND
It loads `.env` first. Without that, `GUARDRAIL_ALLOW_JUDGE` and `ANTHROPIC_API_KEY` are
absent and every judgment-category request comes back ungraded — which looks like a bug
and is really a missing environment. The container (Phase 8) has no .env and calls
uvicorn directly, which is correct there: its configuration comes from the environment.

COST NOTE: with the defaults this server cannot spend money. MockSUT is free and the
judge is off unless GUARDRAIL_ALLOW_JUDGE=true AND a request opts in with use_judge.

PSEUDOCODE
    1. load_dotenv() so the local environment is present before Settings is constructed.
    2. Parse --host/--port/--reload/--sut.
    3. Print what is configured (which SUT, whether judging can happen) so an unexpected
       $0.00 or an unexpected bill is never a surprise.
    4. uvicorn.run("guardrail.api.app:app", ...).
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import uvicorn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--reload", action="store_true", help="restart on code changes (development only)"
    )
    ap.add_argument(
        "--sut",
        default=None,
        choices=("mock", "mlx", "lora"),
        help="overrides $GUARDRAIL_SUT for this process",
    )
    args = ap.parse_args()

    if args.sut:
        os.environ["GUARDRAIL_SUT"] = args.sut

    sut = os.getenv("GUARDRAIL_SUT", "mock")
    judge = os.getenv("GUARDRAIL_ALLOW_JUDGE", "false").lower() in {"1", "true", "yes"}
    print(f"guardrail api | sut={sut} | paid judging {'ENABLED' if judge else 'disabled'}")
    if sut != "mock":
        print(f"  note: {sut} loads the model on the FIRST /evaluate, not at startup.")
    print(f"  docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "guardrail.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
