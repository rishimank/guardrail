"""Load and validate the corpus from disk. One JSONL file per category.

Every entry is validated through the Entry schema on the way in, so a corrupt row
fails at load time with a file-and-line pointer — never silently, and never deep
inside a Phase 5 run where it would just skew a rate. This is the read side of
"the dataset is code": you cannot load an invalid corpus.

PSEUDOCODE
    1. DATASET_DIR = src/guardrail/dataset/data/, one <category>.jsonl per category.
    2. load_category(cat):
       - Missing file -> [] (a category may not be authored yet during Phase 2).
       - Read line by line; skip blank lines; json.loads then Entry(**row).
       - On any failure, raise CorpusError naming the file and 1-based line number.
    3. load_corpus(): concatenate all categories; also assert ids are globally unique
       (a duplicate id means one prompt silently shadows another in any id->entry map).
    4. Return plain lists of frozen Entry objects. No caching, no globals — cheap
       enough (hundreds of rows) that a fresh read per call is fine and honest.
"""

from __future__ import annotations

import json
from pathlib import Path

from guardrail.dataset.schema import Category, Entry

DATASET_DIR = Path(__file__).parent / "data"


class CorpusError(ValueError):
    """A corpus file is missing-shaped, malformed, or invalid. Message names where."""


def category_path(category: Category) -> Path:
    return DATASET_DIR / f"{category.value}.jsonl"


def generated_path(category: Category) -> Path:
    # Machine-generated rows (Phase 2.3: templated + synthesized) live in a separate
    # file so the hand-authored seed file stays pristine and reviewable.
    return DATASET_DIR / f"{category.value}.generated.jsonl"


def load_category(category: Category) -> list[Entry]:
    """Load and validate one category: the hand-authored seed file plus, if present,
    the generated file. Missing files -> empty."""
    entries: list[Entry] = []
    for path in (category_path(category), generated_path(category)):
        entries.extend(_load_file(path, category))
    return entries


def _load_file(path: Path, category: Category) -> list[Entry]:
    if not path.exists():
        return []
    entries: list[Entry] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue  # tolerate blank lines between records
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise CorpusError(f"{path.name}:{lineno}: invalid JSON: {e}") from e
        try:
            entry = Entry(**row)
        except Exception as e:
            raise CorpusError(f"{path.name}:{lineno}: {e}") from e
        # The file is named per category; a row claiming a different category is a
        # copy-paste error that would misfile the prompt. Catch it at the door.
        if entry.category != category:
            raise CorpusError(
                f"{path.name}:{lineno}: row category {entry.category.value!r} "
                f"does not match file {path.name!r}"
            )
        entries.append(entry)
    return entries


def load_corpus() -> list[Entry]:
    """Load every category, concatenated. Raises on a globally duplicated id."""
    entries: list[Entry] = []
    for category in Category:
        entries.extend(load_category(category))

    seen: dict[str, str] = {}
    for e in entries:
        if e.id in seen:
            raise CorpusError(
                f"duplicate id {e.id!r} (in {e.category.value} and {seen[e.id]})"
            )
        seen[e.id] = e.category.value
    return entries
