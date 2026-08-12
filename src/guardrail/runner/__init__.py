"""The corpus-level eval runner (Phase 3.3) — the glue that grades all 512 prompts.

Everything upstream can handle ONE prompt: `get_sut()` generates, `grade()` scores.
This module runs that over the whole corpus and banks the results. It is the mechanism
that Phase 5 turns into rates + Wilson CIs — so it deliberately stops at pass/fail
COUNTS and never computes a rate here (a rate without a CI is the thing the honesty
note warns about; that belongs in Phase 5, not the runner).

Two design choices carry the weight:

  * TWO PHASES, each resumable. Generation (local MLX, free but ~24 min) and grading
    (Haiku, paid) are separated and each appends to its own JSONL, skipping ids already
    done. A crash never re-spends money or re-runs the model, and you can RE-GRADE a
    cached set of responses with a different/fixed judge without regenerating. Same
    trick the calibration scripts use, made canonical here.

  * THE JUDGE IS INJECTABLE. `run_corpus(..., metrics=...)` takes the metric set;
    default is the real Haiku judge, but tests pass a fake one so the entire runner can
    be exercised offline and free against MockSUT. Mock the SUT to make generation free;
    inject a fake judge to make grading free — that is what "mock-first" means here.

PSEUDOCODE
    1. generate_responses(sut, entries, path): serially SUT.generate each prompt (skip
       ids already in `path`), append {id, category, output, model_id, latency, tokens}.
    2. grade_responses(entries, responses_path, path, metrics): for each cached response,
       grade(entry, output, metrics) (skip ids already graded), append the Verdict.
       injection/pii are graded deterministically (free); the rest hit the judge.
    3. aggregate(verdicts): per-category pass/fail COUNTS (no rates) -> RunSummary.
    4. run_corpus(sut, entries, out_dir): generate -> grade -> aggregate, return summary.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from guardrail.dataset.schema import Category, Entry
from guardrail.judge.metrics import build_metrics, grade
from guardrail.sut.base import SUT

Progress = Callable[[str], None]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _done_ids(path: Path) -> set[str]:
    return {row["id"] for row in _read_jsonl(path)}


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def generate_responses(
    sut: SUT,
    entries: Sequence[Entry],
    out_path: Path,
    *,
    resume: bool = True,
    progress: Progress | None = None,
) -> int:
    """Run the SUT over every entry and append one response row each. Returns #generated.

    Serial on purpose: the SUT is a single local model, so there is nothing to overlap.
    Resumable: ids already present in `out_path` are skipped, so a re-run costs no time
    for work already done.
    """
    done = _done_ids(out_path) if resume else set()
    todo = [e for e in entries if e.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"generate: {len(entries)} total | done {len(done)} | todo {len(todo)}")

    n = 0
    with out_path.open("a") as f:
        for i, entry in enumerate(todo, 1):
            r = sut.generate(entry.prompt)
            f.write(
                json.dumps(
                    {
                        "id": entry.id,
                        "category": entry.category.value,
                        "output": r.text,
                        "model_id": r.model_id,
                        "latency_s": r.latency_s,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            n += 1
            _emit(progress, f"  gen [{i}/{len(todo)}] {entry.id}")
    return n


def grade_responses(
    entries: Sequence[Entry],
    responses_path: Path,
    out_path: Path,
    *,
    metrics: dict[Category, object] | None = None,
    resume: bool = True,
    progress: Progress | None = None,
) -> int:
    """Grade every cached response and append one verdict row each. Returns #graded.

    `metrics` is the injectable judge: default is the real Haiku metric set, but tests
    pass a fake one to grade offline. injection/pii are graded deterministically inside
    `grade()` and never touch the judge, so they cost nothing regardless.
    """
    by_id = {e.id: e for e in entries}
    responses = _read_jsonl(responses_path)
    done = _done_ids(out_path) if resume else set()
    todo = [r for r in responses if r["id"] not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"grade: {len(responses)} responses | done {len(done)} | todo {len(todo)}")
    if not todo:
        return 0

    # Build the real judge once (reused across the run) only if none was injected.
    active = build_metrics() if metrics is None else metrics
    n = 0
    with out_path.open("a") as f:
        for i, row in enumerate(todo, 1):
            entry = by_id[row["id"]]
            v = grade(entry, row["output"], active)  # type: ignore[arg-type]
            f.write(
                json.dumps(
                    {
                        "id": entry.id,
                        "category": entry.category.value,
                        "passed": v.passed,
                        "score": v.score,
                        "method": v.method,
                        "reason": v.reason,
                        "model_id": row.get("model_id", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            n += 1
            _emit(
                progress,
                f"  grade [{i}/{len(todo)}] {entry.id} "
                f"{'PASS' if v.passed else 'FAIL'} ({v.method})",
            )
    return n


@dataclass(frozen=True, slots=True)
class CategorySummary:
    """Pass/fail COUNTS for one category. No rate — rates + CIs are Phase 5."""

    category: str
    total: int
    passed: int  # model behaved correctly on this many prompts

    @property
    def failed(self) -> int:
        return self.total - self.passed


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The whole run rolled up to counts, ready for Phase 5 to turn into rates."""

    model_id: str
    n: int
    by_category: dict[str, CategorySummary]

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.by_category.values())

    @property
    def failed(self) -> int:
        return self.n - self.passed

    def format_table(self) -> str:
        """A plain counts table. Deliberately no percentages — see Phase 5."""
        lines = [
            f"model: {self.model_id}   graded: {self.n}",
            f"{'category':<14} {'total':>6} {'passed':>7} {'failed':>7}",
            f"{'-' * 14} {'-' * 6:>6} {'-' * 7:>7} {'-' * 7:>7}",
        ]
        for cat in sorted(self.by_category):
            c = self.by_category[cat]
            lines.append(f"{cat:<14} {c.total:>6} {c.passed:>7} {c.failed:>7}")
        lines.append(f"{'ALL':<14} {self.n:>6} {self.passed:>7} {self.failed:>7}")
        return "\n".join(lines)


def aggregate(verdicts: Iterable[dict], *, model_id: str = "") -> RunSummary:
    """Roll a list of verdict rows up to per-category pass/fail counts."""
    total: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for v in verdicts:
        cat = v["category"]
        total[cat] += 1
        if v["passed"]:
            passed[cat] += 1
    by_category = {
        cat: CategorySummary(category=cat, total=total[cat], passed=passed[cat])
        for cat in total
    }
    return RunSummary(model_id=model_id, n=sum(total.values()), by_category=by_category)


def run_corpus(
    sut: SUT,
    entries: Sequence[Entry],
    out_dir: Path,
    *,
    metrics: dict[Category, object] | None = None,
    resume: bool = True,
    progress: Progress | None = None,
) -> RunSummary:
    """Generate -> grade -> aggregate the whole corpus, banking artifacts in `out_dir`.

    Writes `responses.jsonl` and `verdicts.jsonl` under `out_dir`; both are resumable.
    Returns the pass/fail summary. Pass `metrics` to inject a judge (fake for offline
    tests); default builds the real Haiku judge.
    """
    responses_path = out_dir / "responses.jsonl"
    verdicts_path = out_dir / "verdicts.jsonl"
    generate_responses(sut, entries, responses_path, resume=resume, progress=progress)
    grade_responses(
        entries, responses_path, verdicts_path, metrics=metrics, resume=resume, progress=progress
    )
    return aggregate(_read_jsonl(verdicts_path), model_id=sut.model_id)


__all__ = [
    "CategorySummary",
    "RunSummary",
    "aggregate",
    "generate_responses",
    "grade_responses",
    "run_corpus",
]
