"""The service package: the eval pipeline exposed over HTTP, plus the CI gate (Phase 7).

Import the gate logic from here (`from guardrail.api import evaluate_gate`) — it is a
pure function over dataclasses with no FastAPI anywhere in its import path, so Phase 9's
workflow and its tests can use it without starting a server.

`create_app` is imported lazily via `guardrail.api.app` rather than re-exported here, so
that importing the gate does not drag in FastAPI, Starlette, and the whole ASGI stack.
The gate is the piece that has to run in the leanest possible environment.

PSEUDOCODE
    1. Re-export the gate's public surface (the pure decision logic).
    2. Re-export Settings/get_settings (cheap, no web deps).
    3. Do NOT re-export create_app — `from guardrail.api.app import create_app`.
"""

from __future__ import annotations

from guardrail.api.gate import (
    Counts,
    GateCheck,
    GateDecision,
    GatePolicy,
    RunCounts,
    evaluate_gate,
)
from guardrail.api.settings import Settings, get_settings

__all__ = [
    "Counts",
    "GateCheck",
    "GateDecision",
    "GatePolicy",
    "RunCounts",
    "Settings",
    "evaluate_gate",
    "get_settings",
]
