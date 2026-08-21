"""The service — the eval pipeline exposed over HTTP, and the /gate CI calls (Phase 7).

WHAT THIS SERVICE IS, AND WHAT IT DELIBERATELY IS NOT
A real corpus run is ~24 minutes of local generation plus judging spend. No HTTP request
can block on that, so this is NOT "run the eval over HTTP and wait". It is a DECISION AND
INSPECTION surface over artifacts the pipeline produces, plus a fast single-prompt path:

    GET  /health      what is configured, what is loaded, what is available   (free)
    GET  /benchmarks  the banked rates WITH their Wilson intervals            (free)
    POST /evaluate    one prompt -> generate -> grade -> verdict              (cheap)
    POST /gate        counts + baseline -> PASS/FAIL + every check            (free)
    POST /runs        launch a corpus run, poll GET /runs/{id}                (varies)

/gate and /benchmarks NEED NO MODEL AT ALL. That is the load-bearing property of the
whole design: it is what lets the Phase 8 container and the Phase 9 CI job do the thing
that actually matters — block a regression — without mlx, without 1.6 GB of weights,
and without an API key. The seam from Phase 1 is what makes it true; this file just
declines to break it.

THREE THINGS THE ENDPOINT CODE IS CAREFUL ABOUT

  1. THE MODEL LOADS LAZILY. Nothing at startup touches a SUT, so /health and /gate stay
     instant and free forever. The first /evaluate pays the load cost, once, under a lock.
     A health check that pulls weights is a liability, not a health check.

  2. PAID GRADING NEEDS TWO KEYS TURNED. The request must set use_judge, AND the
     deployment must set GUARDRAIL_ALLOW_JUDGE. Either alone does nothing. A demo left
     running on an open port should not be able to spend money.

  3. `def`, NOT `async def`. Our work is CPU/GPU-bound local generation, not network
     waiting, so async buys nothing — and an `async def` around a blocking model call
     would freeze the whole event loop for every other request. FastAPI runs plain `def`
     endpoints in a threadpool, which is exactly what blocking work wants.

PSEUDOCODE
    1. create_app(settings) -> FastAPI, with a RunRegistry and lazy caches on app.state.
    2. Dependencies: settings, corpus (cached, degrades to empty), sut (lazy+locked),
       metrics (built once, only when judging is permitted).
    3. /health     -> configured state; never loads anything.
    4. /benchmarks -> baselines.json counts -> report.summarize -> rates + Wilson CIs.
    5. /evaluate   -> resolve entry (by id, or ad hoc) -> sut.generate -> grade -> verdict,
                      refusing to fake a grade when the judge is not available.
    6. /gate       -> request counts + chosen baseline + policy -> evaluate_gate -> decision,
                      returning 200 with passed=false (a decision is a successful response;
                      HTTP status describes the REQUEST, not the verdict).
    7. /runs       -> select entries, admit one run at a time, execute in the background.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status

from guardrail.api import gate as gate_mod
from guardrail.api.runs import RunBusyError, RunRegistry
from guardrail.api.schemas import (
    BenchmarkProfile,
    BenchmarksResponse,
    EvaluateRequest,
    EvaluateResponse,
    GateCheckBlock,
    GateRequest,
    GateResponse,
    HealthResponse,
    RateBlock,
    RunListResponse,
    RunRecordBlock,
    RunRequest,
    VerdictBlock,
)
from guardrail.api.settings import Settings, get_settings
from guardrail.dataset import Entry, load_corpus
from guardrail.dataset.schema import Category, ExpectedBehavior, ID_PREFIX, Severity, Source
from guardrail.judge.metrics import DETERMINISTIC_CATEGORIES, build_metrics, grade
from guardrail.report import RunReport, summarize
from guardrail.runner import CategorySummary, RunSummary, aggregate, run_corpus
from guardrail.split import Split, split_for_id
from guardrail.sut import get_sut

API_VERSION = "0.1.0"


# --------------------------------------------------------------------------- helpers


def _load_baselines(path: Path) -> dict[str, Any]:
    """Read the committed baselines file, tolerating its absence.

    A missing baselines file must not stop the service from booting: /health and
    /evaluate are still useful, and the operator needs a running service to tell them
    what is wrong. /gate is the endpoint that hard-fails, and it says exactly which
    file to generate.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc


def _counts_to_summary(model_id: str, categories: dict[str, Any]) -> RunSummary:
    """Adapt stored {category: {n, failures}} to the runner's summary type.

    Written out rather than stored as rates so that /benchmarks reuses `report.summarize`
    — the ONE place Wilson intervals are computed. If this endpoint did its own division
    there would be two implementations of the project's headline statistic, and the tests
    that protect the primitive would no longer protect what is served.
    """
    by_category = {
        cat: CategorySummary(
            category=cat, total=int(v["n"]), passed=int(v["n"]) - int(v["failures"])
        )
        for cat, v in categories.items()
    }
    return RunSummary(
        model_id=model_id,
        n=sum(c.total for c in by_category.values()),
        by_category=by_category,
    )


def _rate_block(cat_rate: Any) -> RateBlock:
    lo, hi = cat_rate.ci
    return RateBlock(
        category=cat_rate.category,
        n=cat_rate.n,
        failures=cat_rate.failures,
        rate_pts=round(cat_rate.rate * 100, 4),
        ci_low_pts=round(lo * 100, 4),
        ci_high_pts=round(hi * 100, 4),
    )


def _report_to_profile(name: str, meta: dict[str, Any], report: RunReport) -> BenchmarkProfile:
    return BenchmarkProfile(
        profile=name,
        model_id=report.model_id,
        split=str(meta.get("split", "all")),
        n=report.n,
        overall=_rate_block(report.overall),
        by_category=[_rate_block(report.by_category[c]) for c in sorted(report.by_category)],
    )


def _adhoc_entry(req: EvaluateRequest) -> Entry:
    """Build a throwaway corpus Entry from an ad-hoc request.

    The id is prefixed to match the category because Entry validates that cross-field
    agreement on construction — a misfiled prompt is a corpus bug this project catches
    at the type level, and an API convenience path does not get to bypass it.
    """
    category = req.category
    assert category is not None  # guaranteed by EvaluateRequest's validator
    return Entry(
        id=f"{ID_PREFIX[category]}-adhoc",
        category=category,
        prompt=req.prompt or "",
        ground_truth=req.ground_truth or "",
        # Not consulted by grade() — the rubrics read ground_truth — but Entry requires
        # it, so it is set to whatever is true for the category rather than a filler.
        expected_behavior=(
            ExpectedBehavior.ANSWER
            if category is Category.OVERREFUSAL
            else ExpectedBehavior.REFUSE
        ),
        severity=Severity.MEDIUM,
        source=Source.HANDWRITTEN,
        forbidden_outputs=tuple(req.forbidden_outputs),
    )


# ---------------------------------------------------------------------- app factory


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. A factory, not a module-level app, so tests get a fresh
    instance with its own registry and caches instead of sharing global state."""

    app = FastAPI(
        title="Guardrail",
        version=API_VERSION,
        description=(
            "Adversarial eval pipeline for LLM failure modes: run the corpus, grade it, "
            "and gate a build on a safety regression."
        ),
    )
    app.state.settings = settings or get_settings()
    app.state.registry = RunRegistry()
    app.state.sut = None
    app.state.metrics = None
    app.state.corpus = None
    app.state.load_lock = threading.Lock()

    # ------------------------------------------------------------- dependencies
    # A dependency is just a function FastAPI calls to build an argument. Declaring them
    # this way (rather than reaching for a global inside each endpoint) is what lets a
    # test swap in MockSUT or a fake judge with app.dependency_overrides[...] — the same
    # injectable-seam idea the runner already uses for `metrics`.

    def settings_dep(request: Request) -> Settings:
        return request.app.state.settings  # type: ignore[no-any-return]

    def corpus_dep(request: Request) -> list[Entry]:
        """The corpus, loaded once. An unreadable corpus yields [], not a crash: the
        gate does not need it, and a service that refuses to start because of a missing
        optional asset cannot tell anyone what is missing."""
        state = request.app.state
        if state.corpus is None:
            try:
                state.corpus = load_corpus()
            except Exception:  # noqa: BLE001 — reported via /health.corpus_available
                state.corpus = []
        return state.corpus  # type: ignore[no-any-return]

    def sut_dep(request: Request) -> Any:
        """The SUT, loaded lazily and at most once.

        Under a lock because two concurrent first-requests would otherwise both load the
        model — on a 16 GB machine, loading Qwen twice at once is how you get an OOM
        instead of a response.
        """
        state = request.app.state
        cfg: Settings = state.settings
        if state.sut is None:
            with state.load_lock:
                if state.sut is None:
                    try:
                        state.sut = get_sut(cfg.sut)
                    except Exception as exc:  # noqa: BLE001
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"could not load SUT {cfg.sut!r}: {exc}",
                        ) from exc
        return state.sut

    def ensure_metrics() -> dict[Category, Any]:
        """The judge, built at most once per process and only when actually needed.

        Deliberately NOT a `Depends(...)` dependency: a declared dependency runs on every
        request to the endpoint, so it would construct the judge even for the free
        deterministic categories that never use it. Building the judge is the moment an
        API key becomes required, so it must happen only on the paths that truly grade.
        """
        if app.state.metrics is None:
            with app.state.load_lock:
                if app.state.metrics is None:
                    app.state.metrics = build_metrics()
        return app.state.metrics  # type: ignore[no-any-return]

    def baselines_dep(request: Request) -> dict[str, Any]:
        cfg: Settings = request.app.state.settings
        return _load_baselines(cfg.baselines_path)

    # ------------------------------------------------------------------ /health

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(
        request: Request,
        cfg: Settings = Depends(settings_dep),
        corpus: list[Entry] = Depends(corpus_dep),
        baselines: dict[str, Any] = Depends(baselines_dep),
    ) -> HealthResponse:
        """Liveness plus configuration. Loads nothing; safe to poll every few seconds."""
        return HealthResponse(
            sut=cfg.sut,
            sut_loaded=request.app.state.sut is not None,
            corpus_size=len(corpus),
            corpus_available=bool(corpus),
            baseline_profiles=sorted(baselines.get("profiles", {})),
            judge_enabled=cfg.allow_judge,
            version=API_VERSION,
        )

    # -------------------------------------------------------------- /benchmarks

    @app.get("/benchmarks", response_model=BenchmarksResponse, tags=["meta"])
    def benchmarks(baselines: dict[str, Any] = Depends(baselines_dep)) -> BenchmarksResponse:
        """The banked measurements as rates WITH their Wilson intervals.

        Counts are what is stored; rates and intervals are computed here, on every
        request, by the same `report.summarize` that writes BENCHMARKS.md. Storing the
        counts and deriving the rates means the served numbers and the committed
        markdown cannot drift — there is only one arithmetic path.
        """
        if not baselines:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "no baselines file. Generate it with "
                    "`venv/bin/python scripts/write_benchmarks.py`."
                ),
            )
        conf = float(baselines.get("conf", 0.95))
        profiles = []
        for name, meta in sorted(baselines.get("profiles", {}).items()):
            report = summarize(
                _counts_to_summary(meta.get("model_id", ""), meta["categories"]), conf=conf
            )
            profiles.append(_report_to_profile(name, meta, report))
        return BenchmarksResponse(
            commit=str(baselines.get("commit", "")), conf=conf, profiles=profiles
        )

    # ---------------------------------------------------------------- /evaluate

    @app.post("/evaluate", response_model=EvaluateResponse, tags=["eval"])
    def evaluate(
        req: EvaluateRequest,
        cfg: Settings = Depends(settings_dep),
        corpus: list[Entry] = Depends(corpus_dep),
        sut: Any = Depends(sut_dep),
    ) -> EvaluateResponse:
        """Generate one response and grade it.

        Injection and PII are graded deterministically (a substring check against the
        planted canaries) and are therefore always free and always available. The four
        judgment categories need the LLM judge, which requires BOTH the request to opt
        in and the deployment to permit it; when either is off, the response is returned
        UNGRADED with a note. It is never graded as a pass — inventing a verdict because
        grading was unavailable is precisely the dishonesty this project is built to avoid.
        """
        if req.entry_id is not None:
            match = next((e for e in corpus if e.id == req.entry_id), None)
            if match is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"no corpus entry with id {req.entry_id!r}",
                )
            entry = match
        else:
            if len(req.prompt or "") > cfg.max_prompt_chars:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"prompt exceeds max_prompt_chars ({cfg.max_prompt_chars})",
                )
            entry = _adhoc_entry(req)

        response = sut.generate(entry.prompt, max_tokens=req.max_tokens)

        deterministic = entry.category in DETERMINISTIC_CATEGORIES
        may_judge = req.use_judge and cfg.allow_judge
        if not deterministic and not may_judge:
            note = (
                f"{entry.category.value} needs the LLM judge. Set use_judge=true on the "
                "request AND GUARDRAIL_ALLOW_JUDGE=true on the deployment. Returning the "
                "generation ungraded rather than assuming a verdict."
            )
            return EvaluateResponse(
                id=entry.id,
                category=entry.category,
                output=response.text,
                model_id=response.model_id,
                latency_s=response.latency_s,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                verdict=None,
                graded=False,
                note=note,
            )

        # Deterministic categories pass metrics=None and never touch the judge.
        metrics = None if deterministic else ensure_metrics()
        verdict = grade(entry, response.text, metrics)
        return EvaluateResponse(
            id=entry.id,
            category=entry.category,
            output=response.text,
            model_id=response.model_id,
            latency_s=response.latency_s,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            verdict=VerdictBlock(
                passed=verdict.passed,
                score=verdict.score,
                method=verdict.method,  # type: ignore[arg-type]
                reason=verdict.reason,
            ),
            graded=True,
            note=None,
        )

    # -------------------------------------------------------------------- /gate

    @app.post("/gate", response_model=GateResponse, tags=["gate"])
    def run_gate(
        req: GateRequest,
        cfg: Settings = Depends(settings_dep),
        baselines: dict[str, Any] = Depends(baselines_dep),
    ) -> GateResponse:
        """Decide whether a run may ship. The endpoint Phase 9 calls.

        Returns HTTP 200 even when the gate FAILS. An HTTP status describes what happened
        to the request, and "I successfully computed that your model regressed" is a
        successful request. Encoding the verdict in the status code would conflate
        "the gate said no" with "the gate is broken" — and a CI job must be able to tell
        those apart, because one blocks a merge and the other pages whoever owns this.
        """
        # Counts() rejects impossible input (failures > n, negatives). That is a bad
        # REQUEST, not a server fault, so it must surface as 422 — a 500 here would page
        # whoever owns the service for what is actually a caller's malformed payload.
        def to_counts(blocks: dict[str, Any]) -> dict[str, gate_mod.Counts]:
            try:
                return {
                    cat: gate_mod.Counts(n=c.n, failures=c.failures)
                    for cat, c in blocks.items()
                }
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
                ) from exc

        run = gate_mod.RunCounts(model_id=req.model_id, by_category=to_counts(req.categories))

        profile_name = req.baseline_profile
        if req.baseline is not None:
            baseline = gate_mod.RunCounts(
                model_id="(inline)", by_category=to_counts(req.baseline)
            )
            profile_name = "(inline)"
        else:
            profiles = baselines.get("profiles", {})
            if not profile_name:
                profile_name = str(baselines.get("sut_defaults", {}).get(cfg.sut, ""))
            if profile_name not in profiles:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"unknown baseline profile {profile_name!r}. "
                        f"Available: {sorted(profiles)}. Regenerate with "
                        "`venv/bin/python scripts/write_benchmarks.py`."
                    ),
                )
            meta = profiles[profile_name]
            baseline = gate_mod.RunCounts.from_dict(meta)

        # Committed policy first, request overrides on top — and the response echoes the
        # merged result, so a decision can never be quoted without its thresholds.
        policy_data: dict[str, Any] = {}
        if cfg.gate_policy_path.exists():
            policy_data = json.loads(cfg.gate_policy_path.read_text())
        if req.policy is not None:
            policy_data.update(
                {k: v for k, v in req.policy.model_dump().items() if v is not None}
            )
        policy = gate_mod.GatePolicy.from_dict(policy_data)

        try:
            decision = gate_mod.evaluate_gate(run, baseline, policy)
        except ValueError as exc:  # corrupt counts, e.g. failures > n
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

        def block(c: gate_mod.GateCheck) -> GateCheckBlock:
            limit = None if c.limit != c.limit else c.limit  # NaN -> null
            return GateCheckBlock(
                kind=c.kind,
                category=c.category,
                passed=c.passed,
                observed=round(c.observed, 4),
                limit=None if limit is None else round(limit, 4),
                detail=c.detail,
            )

        return GateResponse(
            passed=decision.passed,
            summary=decision.summary,
            run_model_id=decision.run_model_id,
            baseline_model_id=decision.baseline_model_id,
            baseline_profile=profile_name,
            checks=[block(c) for c in decision.checks],
            violations=[block(c) for c in decision.failures],
            policy_applied={
                "tolerance_pts": policy.tolerance_pts,
                "per_category_tolerance_pts": policy.per_category_tolerance_pts,
                "ceiling_pts": policy.ceiling_pts,
                "overall_ceiling_pts": policy.overall_ceiling_pts,
                "min_coverage": policy.min_coverage,
            },
            table=decision.format_table(),
        )

    # -------------------------------------------------------------------- /runs

    @app.post(
        "/runs",
        response_model=RunRecordBlock,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["runs"],
    )
    def launch_run(
        req: RunRequest,
        background: BackgroundTasks,
        cfg: Settings = Depends(settings_dep),
        corpus: list[Entry] = Depends(corpus_dep),
        sut: Any = Depends(sut_dep),
    ) -> RunRecordBlock:
        """Accept a corpus run and return 202 immediately with an id to poll.

        202 Accepted, not 200 OK: the work has been admitted, not completed. Encoding
        that in the status code is what tells a client it must poll rather than treat
        the response as the result.
        """
        if not corpus:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no corpus available to run",
            )

        selected = list(corpus)
        if req.split != "all":
            want = Split.TEST if req.split == "test" else Split.TRAIN
            selected = [e for e in selected if split_for_id(e.id) is want]

        may_judge = req.use_judge and cfg.allow_judge
        if req.categories:
            requested = set(req.categories)
            judgment = sorted(c.value for c in requested if c not in DETERMINISTIC_CATEGORIES)
            if judgment and not may_judge:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"categories {judgment} require the LLM judge. Set use_judge=true "
                        "and GUARDRAIL_ALLOW_JUDGE=true, or request only the "
                        "deterministic categories (injection, pii), which grade for free."
                    ),
                )
            selected = [e for e in selected if e.category in requested]
            restricted = False
        elif not may_judge:
            # No judge available and no explicit ask: fall back to the free, offline
            # subset rather than launching a run that would die on the first
            # hallucination row. This is the shape the CI gate uses.
            selected = [e for e in selected if e.category in DETERMINISTIC_CATEGORIES]
            restricted = True
        else:
            restricted = False

        if req.limit:
            selected = selected[: req.limit]
        if not selected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="no prompts match this split/category/limit combination",
            )

        # `out_dir` becomes a path, so it must be a plain name — a caller must not be
        # able to steer writes outside runs/ with '../'.
        name = req.out_dir or cfg.sut
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="out_dir must be a single directory name, not a path",
            )
        out_dir = cfg.runs_dir / name

        try:
            record = app.state.registry.create(
                sut=cfg.sut, split=req.split, requested=len(selected)
            )
        except RunBusyError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        metrics = ensure_metrics() if may_judge else None
        registry = app.state.registry

        def work(progress: Callable[[str], None]) -> dict[str, Any]:
            if restricted:
                progress(
                    "judge unavailable: restricted to deterministic categories "
                    "(injection, pii) — these grade for free and offline"
                )
            summary = run_corpus(
                sut,
                selected,
                out_dir,
                metrics=metrics,
                resume=req.resume,
                progress=progress,
            )
            return {
                "model_id": summary.model_id,
                "n": summary.n,
                "categories": {
                    cat: {"n": c.total, "failures": c.failed}
                    for cat, c in sorted(summary.by_category.items())
                },
                "out_dir": str(out_dir),
            }

        background.add_task(registry.execute, record.id, work)
        return RunRecordBlock(**record.to_dict())

    @app.get("/runs", response_model=RunListResponse, tags=["runs"])
    def list_runs() -> RunListResponse:
        return RunListResponse(
            runs=[RunRecordBlock(**r.to_dict()) for r in app.state.registry.list()]
        )

    @app.get("/runs/{run_id}", response_model=RunRecordBlock, tags=["runs"])
    def get_run(run_id: str) -> RunRecordBlock:
        record = app.state.registry.get(run_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"no run {run_id!r}"
            )
        return RunRecordBlock(**record.to_dict())

    return app


# The ASGI entrypoint uvicorn is pointed at: `uvicorn guardrail.api.app:app`.
app = create_app()


__all__ = ["API_VERSION", "app", "create_app"]
