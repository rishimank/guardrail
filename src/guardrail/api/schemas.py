"""The wire contract — every request and response shape the API accepts or emits.

These are Pydantic models, which means FastAPI derives three things from each one: the
JSON parsing, the validation (a malformed request is rejected with a 422 naming the bad
field, before any endpoint code runs), and the OpenAPI schema behind /docs. One
declaration, three jobs — that is the entire reason this project uses FastAPI rather
than a hand-rolled HTTP handler.

WHY THESE ARE SEPARATE FROM THE DOMAIN TYPES
`gate.RunCounts`, `runner.RunSummary` and `dataset.Entry` are the internal vocabulary;
the models here are the PUBLIC one. Keeping them apart means an internal refactor cannot
silently change the API's shape, and — more important for this project — a response can
never accidentally leak an internal field that was never meant to be served. The mapping
between the two is written out explicitly in app.py, where it is reviewable.

WHY REQUESTS ARE STRICT (extra="forbid")
A typo'd field name in a permissive schema is silently ignored, so a caller who POSTs
{"tolerence_pts": 20} to the gate gets the DEFAULT tolerance and a green build they
think they configured. For a component whose job is to fail builds, silently ignoring
input is the worst available behaviour. Unknown fields are rejected outright.

PSEUDOCODE
    1. Health / Benchmarks: what the service is and what it has banked.
    2. Evaluate: one prompt in (by corpus id OR ad hoc), one graded verdict out.
    3. Gate: counts + optional policy in, decision + every check out.
    4. Runs: launch params in, job record out (id, state, progress, summary).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from guardrail.dataset.schema import Category

# Requests are strict; responses are not (they are ours to construct, and forbidding
# extras on them would only add a failure mode with no caller to protect).
_STRICT = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- health


class HealthResponse(BaseModel):
    """Liveness plus enough state to debug a misconfigured deployment at a glance.

    Reports what is CONFIGURED, never loading the model to find out: /health is what
    Docker's HEALTHCHECK and CI hit repeatedly, and it must stay free and instant. A
    health check that pulls 1.6 GB of weights is a liability, not a health check.
    """

    status: Literal["ok"] = "ok"
    sut: str = Field(description="Configured SUT name: mock | mlx | lora.")
    sut_loaded: bool = Field(description="Whether the model has actually been loaded yet.")
    corpus_size: int = Field(description="Prompts loadable from data/; 0 if absent.")
    corpus_available: bool
    baseline_profiles: list[str] = Field(
        description="Baselines available to gate against, by profile name."
    )
    judge_enabled: bool = Field(description="Whether this deployment permits paid grading.")
    version: str


# ----------------------------------------------------------------------- benchmarks


class RateBlock(BaseModel):
    """One category's failure rate with the interval that makes it interpretable.

    The CI travels with the rate everywhere in this project, including over the wire.
    A bare rate is the exact thing the honesty note exists to prevent being quoted.
    """

    category: str
    n: int
    failures: int
    rate_pts: float = Field(description="Failure rate in percentage points. Higher is worse.")
    ci_low_pts: float
    ci_high_pts: float


class BenchmarkProfile(BaseModel):
    """One banked measurement: which model, which split, and the rates it produced."""

    profile: str
    model_id: str
    split: str
    n: int
    overall: RateBlock
    by_category: list[RateBlock]


class BenchmarksResponse(BaseModel):
    """Everything this service has banked, plus the provenance of those numbers."""

    commit: str = Field(description="Commit the baselines were generated from.")
    conf: float = Field(description="Confidence level of every interval, e.g. 0.95.")
    profiles: list[BenchmarkProfile]


# --------------------------------------------------------------------------- evaluate


class EvaluateRequest(BaseModel):
    """One prompt to run and grade. Two modes, exactly one of which must be used.

    BY CORPUS ID (`entry_id`) — the honest mode. The corpus entry carries the authored
    ground truth and, for injection/PII, the planted canary strings, so the verdict is
    produced by the same grading path as a banked run.

    AD HOC (`prompt` + `category` + `ground_truth`) — for trying a new attack. The
    caller must supply the ground truth, because nothing can grade a response without
    knowing what correct looks like. This is the property that makes the project's
    hallucination measurement possible at all, and it does not get relaxed for a
    convenience endpoint.
    """

    model_config = _STRICT

    entry_id: str | None = Field(default=None, description="Id of an existing corpus prompt.")

    prompt: str | None = Field(default=None, description="Ad-hoc prompt text.")
    category: Category | None = Field(default=None, description="Required for ad-hoc prompts.")
    ground_truth: str | None = Field(
        default=None, description="What a correct response must respect. Required ad hoc."
    )
    forbidden_outputs: list[str] = Field(
        default_factory=list,
        description=(
            "Canary strings for injection/PII. Their presence in the output IS the "
            "failure, checked by substring — free, instant, and un-gameable."
        ),
    )

    use_judge: bool = Field(
        default=False,
        description=(
            "Opt in to LLM grading for the four judgment categories. Costs money. "
            "Ignored unless the deployment also sets GUARDRAIL_ALLOW_JUDGE=true. "
            "Injection and PII never need it and never pay for it."
        ),
    )
    max_tokens: Annotated[int, Field(gt=0, le=2048)] = 512

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> EvaluateRequest:
        by_id = self.entry_id is not None
        ad_hoc = self.prompt is not None
        if by_id == ad_hoc:
            raise ValueError(
                "supply either 'entry_id' (grade an existing corpus prompt) or "
                "'prompt' (ad hoc), not both and not neither."
            )
        if ad_hoc and (self.category is None or self.ground_truth is None):
            raise ValueError(
                "an ad-hoc prompt needs 'category' and 'ground_truth' — a response "
                "cannot be graded without knowing what correct behaviour is."
            )
        return self


class VerdictBlock(BaseModel):
    """How one response was graded, and by what."""

    passed: bool = Field(description="True = the model behaved correctly on this prompt.")
    score: float
    method: Literal["deterministic", "geval"] = Field(
        description="deterministic = free canary substring check; geval = paid LLM judge."
    )
    reason: str


class EvaluateResponse(BaseModel):
    """The generation, its cost/latency metadata, and the verdict."""

    id: str
    category: Category
    output: str
    model_id: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    verdict: VerdictBlock | None = Field(
        default=None,
        description=(
            "null when grading was unavailable. Deliberately null rather than a "
            "default-pass: an ungraded response must never read as a passing one."
        ),
    )
    graded: bool = Field(description="False when a judgment category was left ungraded.")
    note: str | None = Field(
        default=None,
        description="Why grading was skipped, when it was (e.g. judge not enabled).",
    )


# ------------------------------------------------------------------------------ gate


class CountsBlock(BaseModel):
    """Failures out of n for one category — the gate's input atom."""

    model_config = _STRICT

    n: Annotated[int, Field(ge=0)]
    failures: Annotated[int, Field(ge=0)]


class GatePolicyBlock(BaseModel):
    """Threshold overrides for a single request. Omit to use the committed policy.

    Overriding is allowed because a caller experimenting locally should not have to edit
    a committed file — but the response always echoes the policy that was actually
    applied, so a decision can never be quoted without the thresholds that produced it.
    """

    model_config = _STRICT

    tolerance_pts: float | None = None
    per_category_tolerance_pts: dict[str, float] | None = None
    ceiling_pts: dict[str, float] | None = None
    overall_ceiling_pts: float | None = None
    min_coverage: Annotated[float, Field(gt=0, le=1)] | None = None


class GateRequest(BaseModel):
    """A run to judge, the baseline to judge it against, and optional thresholds."""

    model_config = _STRICT

    categories: dict[str, CountsBlock] = Field(
        description="Per-category counts from the run being gated."
    )
    model_id: str = Field(default="", description="Which model produced these counts.")
    baseline_profile: str = Field(
        default="",
        description=(
            "Which committed baseline to compare against. Defaults to the profile "
            "matching the configured SUT."
        ),
    )
    baseline: dict[str, CountsBlock] | None = Field(
        default=None,
        description="Inline baseline, for comparing two runs without committing either.",
    )
    policy: GatePolicyBlock | None = None

    @model_validator(mode="after")
    def _non_empty(self) -> GateRequest:
        if not self.categories:
            raise ValueError(
                "'categories' is empty — there is nothing to gate. An empty run must "
                "not be gradeable as a pass."
            )
        return self


class GateCheckBlock(BaseModel):
    """One rule applied to one category."""

    kind: str
    category: str
    passed: bool
    observed: float
    limit: float | None = Field(description="null where no threshold applies.")
    detail: str


class GateResponse(BaseModel):
    """The decision, every check behind it, and the thresholds that were applied."""

    passed: bool
    summary: str
    run_model_id: str
    baseline_model_id: str
    baseline_profile: str
    checks: list[GateCheckBlock]
    violations: list[GateCheckBlock]
    policy_applied: dict[str, Any]
    table: str = Field(description="Fixed-width rendering for CI logs.")


# ------------------------------------------------------------------------------ runs


class RunRequest(BaseModel):
    """Parameters for a corpus run launched through the service."""

    model_config = _STRICT

    split: Literal["all", "train", "test"] = Field(
        default="test",
        description=(
            "Defaults to 'test' — the held-out side. Scoring a fine-tune on the train "
            "split measures memorisation, so the safe option is the default."
        ),
    )
    categories: list[Category] | None = None
    limit: Annotated[int, Field(gt=0)] | None = None
    use_judge: bool = Field(
        default=False,
        description="Grade judgment categories with the paid judge. Requires the "
        "deployment to permit it.",
    )
    out_dir: str | None = Field(
        default=None, description="Run directory under runs/. Defaults to the SUT name."
    )
    resume: bool = True


class RunRecordBlock(BaseModel):
    """A job's state. Poll GET /runs/{id} until state leaves 'queued'/'running'."""

    id: str
    state: Literal["queued", "running", "succeeded", "failed"]
    sut: str
    split: str
    requested: int = Field(description="Prompts selected for this run.")
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    progress: list[str] = Field(default_factory=list, description="Recent progress lines.")
    summary: dict[str, Any] | None = Field(
        default=None, description="Per-category counts once the run succeeds."
    )
    error: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunRecordBlock]


__all__ = [
    "BenchmarkProfile",
    "BenchmarksResponse",
    "CountsBlock",
    "EvaluateRequest",
    "EvaluateResponse",
    "GateCheckBlock",
    "GatePolicyBlock",
    "GateRequest",
    "GateResponse",
    "HealthResponse",
    "RateBlock",
    "RunListResponse",
    "RunRecordBlock",
    "RunRequest",
    "VerdictBlock",
]
