"""The job registry — what makes POST /runs possible without a 24-minute HTTP request.

A full corpus run is ~24 minutes of local generation. No HTTP client will wait for that,
so the service cannot answer "run the corpus" synchronously. The standard shape is used
here: accept the work, return 202 with an id immediately, and let the caller poll.

WHY IN-MEMORY, AND WHY THAT IS THE RIGHT CALL HERE
State lives in a dict and dies with the process. The alternative — Redis/Celery, or a
job table on disk — buys durability across restarts, which matters when jobs are
customer work that must not be lost. Here a job is a local eval whose real output is
already durable (the runner banks resumable JSONL to runs/ as it goes) and whose real
invocation is the CLI. So a restart loses the *status record* of a run whose *results*
are on disk and resumable. Paying for a broker to protect a status record would be
architecture for its own sake — and the resumability that makes it safe was built in
Phase 3, not bolted on here.

SINGLE-FLIGHT, ENFORCED
Only one run may be active at a time. This is not tidiness: the SUT is ONE local model
on a 16 GB machine, and two concurrent runs would contend for memory and interleave
writes into the same JSONL. The runner is deliberately serial for the same reason. A
second request while one is active gets a clear 409 rather than a corrupted run.

PSEUDOCODE
    1. RunState: queued -> running -> succeeded | failed.
    2. RunRecord: id, state, params, timestamps, a capped progress ring, summary, error.
    3. RunRegistry (thread-safe):
       - create(...)          -> a queued record, refusing if one is already active
       - execute(id, work)    -> mark running, call work(progress_cb), bank the summary,
                                 and convert ANY exception into a failed record + message
       - get / list           -> read-only views for the endpoints
    4. Progress lines are capped: a 662-prompt run emits ~1300 of them, and an unbounded
       list in a long-lived process is a slow memory leak.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Enough lines to see what a run is doing now; not enough to grow without bound.
MAX_PROGRESS_LINES = 100

ACTIVE_STATES = frozenset({"queued", "running"})


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class RunRecord:
    """One launched run. Mutable by design — it is a status record, not a measurement.

    (Contrast `sut.Response` and `judge.Verdict`, which are frozen: those ARE
    measurements, and measurements do not get edited. This is bookkeeping.)
    """

    id: str
    sut: str
    split: str
    requested: int
    state: RunState = RunState.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: deque[str] = field(
        default_factory=lambda: deque(maxlen=MAX_PROGRESS_LINES)
    )
    summary: dict[str, Any] | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "sut": self.sut,
            "split": self.split,
            "requested": self.requested,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": list(self.progress),
            "summary": self.summary,
            "error": self.error,
        }


class RunBusyError(RuntimeError):
    """Raised when a run is requested while another is active. Surfaces as HTTP 409."""


class RunRegistry:
    """Thread-safe store of run records, with single-flight admission control."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, *, sut: str, split: str, requested: int) -> RunRecord:
        """Register a queued run, or refuse if one is already in flight."""
        with self._lock:
            active = [r for r in self._runs.values() if r.active]
            if active:
                raise RunBusyError(
                    f"run {active[0].id} is already {active[0].state.value}. The SUT is a "
                    "single local model, so runs are serial — wait for it to finish, or "
                    "poll GET /runs/{id}."
                )
            record = RunRecord(
                id=uuid.uuid4().hex[:12], sut=sut, split=split, requested=requested
            )
            self._runs[record.id] = record
            return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[RunRecord]:
        """Newest first — the one you just launched is the one you want to see."""
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def append_progress(self, run_id: str, line: str) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.progress.append(line)

    def execute(
        self,
        run_id: str,
        work: Callable[[Callable[[str], None]], dict[str, Any]],
    ) -> None:
        """Run `work` to completion, recording the outcome on the record.

        `work` receives a progress callback and returns the run summary as a dict.

        EVERY exception is caught and stored. This runs in a background thread with no
        client attached, so an escaping exception would vanish into a log and leave the
        record stuck in 'running' forever — a status endpoint that lies is worse than
        one that reports a failure.
        """
        record = self.get(run_id)
        if record is None:
            return
        with self._lock:
            record.state = RunState.RUNNING
            record.started_at = time.time()

        try:
            summary = work(lambda line: self.append_progress(run_id, line))
        except Exception as exc:  # noqa: BLE001 — see docstring: nothing may escape
            with self._lock:
                record.state = RunState.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                record.finished_at = time.time()
            return

        with self._lock:
            record.state = RunState.SUCCEEDED
            record.summary = summary
            record.finished_at = time.time()


__all__ = [
    "ACTIVE_STATES",
    "MAX_PROGRESS_LINES",
    "RunBusyError",
    "RunRecord",
    "RunRegistry",
    "RunState",
]
