"""LoRASUT — the fine-tuned system under test: base Qwen + a trained LoRA adapter.

This is the second half of the Phase 6 comparison. `MLXSUT` measures the base model,
`LoRASUT` measures the same model with an adapter applied, and the reduction claim is
the delta between two runs of the IDENTICAL harness over the IDENTICAL TEST split.

WHY THIS IS A CLASS AND NOT JUST `MLXSUT(adapter_path=...)`
    Mechanically, applying an adapter is one argument to mlx_lm.load(). If that were the
    whole story this file would be dead weight. Two things make it not the whole story:

    1. CHECKPOINT SELECTION. A training run does not produce "an adapter", it produces a
       SEQUENCE of them — adapters/v2 holds 0000025_, 0000050_, ... plus a final
       adapters.safetensors. Which one you evaluate is a real experimental choice
       (validation loss bottomed mid-run and rose after; see runs/lora_train*.log), and
       mlx_lm cannot express it: `load_adapters` hard-requires a DIRECTORY containing a
       file literally named `adapters.safetensors`, so a mid-training checkpoint is
       unloadable without staging one. This class owns that staging.

    2. PROVENANCE. `MLXSUT.model_id` would report the adapter DIRECTORY, so a run of
       checkpoint 100 and a run of checkpoint 150 would both be recorded as
       "...+adapters/v2" — indistinguishable in verdicts.jsonl. base.py says the whole
       reason model_id exists is that Phase 6 compares two result sets and without it
       they cannot be told apart. A directory name is not enough resolution once the
       directory holds seven candidates. So model_id here names the exact checkpoint:
       `mlx-community/Qwen2.5-3B-Instruct-4bit+v2@100`.

    Generation itself is NOT reimplemented — this composes an MLXSUT and delegates. The
    chat template and sampler must be byte-identical to the baseline run or the
    comparison measures the harness instead of the fine-tune.

THE STAGING TRICK
    To evaluate checkpoint N we build a temporary directory containing a symlink to
    adapter_config.json and a symlink named `adapters.safetensors` pointing at
    `00000N0_adapters.safetensors`, then hand mlx_lm that directory. Symlinks, not
    copies: the checkpoints are 13MB each and we may load several in one session.
    The temp dir is discarded as soon as load() returns, because mlx_lm reads the
    weights eagerly into memory — nothing needs the staged directory afterwards.

PSEUDOCODE
    1. checkpoints(dir) -> {iteration: path} by parsing `<7 digits>_adapters.safetensors`.
       Pure; no mlx; importable and testable on linux CI.
    2. resolve_checkpoint(dir, which) -> (weights_path, label):
       - which is None -> final adapters.safetensors, labelled with config["iters"]
       - which is an int -> that checkpoint, or ValueError listing what IS available
    3. LoRASUT.__init__: validate the dir, resolve the checkpoint, stage it if needed,
       build the inner MLXSUT once, keep the training config for provenance.
    4. model_id -> "<base>+<dirname>@<label>"; generate() delegates unchanged.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from guardrail.sut.base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, Response
from guardrail.sut.mlx_sut import DEFAULT_MODEL_ID, MLXSUT

# mlx_lm's on-disk contract. Both names are fixed by mlx_lm.tuner.utils.load_adapters,
# which opens `<dir>/adapter_config.json` and loads `<dir>/adapters.safetensors`. We do
# not get to rename these; we can only build a directory that satisfies them.
CONFIG_NAME = "adapter_config.json"
FINAL_WEIGHTS_NAME = "adapters.safetensors"

# mlx_lm writes periodic checkpoints as e.g. `0000100_adapters.safetensors`.
_CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


def checkpoints(adapter_dir: Path | str) -> dict[int, Path]:
    """Return {iteration: path} for every periodic checkpoint in `adapter_dir`.

    Excludes the final `adapters.safetensors`, which carries no iteration number in
    its name — its iteration count comes from adapter_config.json instead.
    """
    d = Path(adapter_dir)
    found: dict[int, Path] = {}
    if not d.is_dir():
        return found
    for path in d.iterdir():
        m = _CHECKPOINT_RE.match(path.name)
        if m:
            found[int(m.group(1))] = path
    return found


def read_config(adapter_dir: Path | str) -> dict:
    """Load the training config mlx_lm wrote next to the weights.

    This is the provenance record: rank, num_layers, iters, learning_rate, and which
    data directory produced it. BENCHMARKS.md quotes these, so they are read from the
    adapter itself rather than retyped from a shell history.
    """
    path = Path(adapter_dir) / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — {adapter_dir!r} is not an mlx_lm adapter directory. "
            "Point GUARDRAIL_ADAPTER_PATH at a specific training run (e.g. "
            "'adapters/v2'), not at the parent folder that holds several."
        )
    return json.loads(path.read_text())


def resolve_checkpoint(
    adapter_dir: Path | str, which: int | None = None
) -> tuple[Path, int | None]:
    """Pick which weights file to evaluate. Returns (path, iteration_label).

    `which=None` means the final weights — what a plain `mlx_lm.load(dir)` would use.
    An explicit int selects a periodic checkpoint, and an unavailable one raises with
    the list of what *is* there: getting this wrong silently would mean measuring a
    different model than the one you meant to report, which is unrecoverable after the
    fact because the verdicts would look perfectly normal.
    """
    d = Path(adapter_dir)
    available = checkpoints(d)

    if which is None:
        final = d / FINAL_WEIGHTS_NAME
        if not final.exists():
            raise FileNotFoundError(
                f"{final} not found. Available checkpoints: "
                f"{sorted(available) or 'none'}."
            )
        # The final weights are the last iteration; the config knows which that was.
        label = read_config(d).get("iters")
        return final, label if isinstance(label, int) else None

    if which not in available:
        raise ValueError(
            f"No checkpoint at iteration {which} in {d}. "
            f"Available: {sorted(available) or 'none'} "
            f"(plus the final weights, selected with checkpoint=None)."
        )
    return available[which], which


class LoRASUT:
    """Satisfies the SUT protocol with base Qwen + a specific LoRA checkpoint."""

    def __init__(
        self,
        adapter_path: Path | str,
        model_id: str = DEFAULT_MODEL_ID,
        checkpoint: int | None = None,
    ) -> None:
        self._adapter_dir = Path(adapter_path)
        if not self._adapter_dir.is_dir():
            raise FileNotFoundError(
                f"No adapter directory at {self._adapter_dir!r}. Train one first: "
                "see runs/lora_train_v2.log for the mlx_lm lora invocation."
            )

        self._config = read_config(self._adapter_dir)
        weights, label = resolve_checkpoint(self._adapter_dir, checkpoint)
        self._label = label

        if weights.name == FINAL_WEIGHTS_NAME:
            # Already the shape mlx_lm wants — hand it the directory directly.
            self._inner = MLXSUT(model_id=model_id, adapter_path=str(self._adapter_dir))
        else:
            self._inner = self._load_staged(model_id, weights)

        # Guard against a silently mismatched base model: an adapter trained against
        # one base and applied to another loads without error (load_weights is
        # strict=False) and then produces quiet nonsense.
        trained_on = self._config.get("model")
        if trained_on and trained_on != model_id:
            raise ValueError(
                f"Adapter was trained on {trained_on!r} but is being applied to "
                f"{model_id!r}. LoRA weights are base-model-specific; applying them "
                "across bases loads without error and degrades output invisibly."
            )

    def _load_staged(self, model_id: str, weights: Path) -> MLXSUT:
        """Build a throwaway directory shaped the way mlx_lm insists on, and load it.

        The temp dir only has to survive the load() call — mlx_lm reads the safetensors
        eagerly, so once MLXSUT.__init__ returns, the weights live in memory and the
        staged links are dead. Hence the context manager rather than a long-lived dir.
        """
        with tempfile.TemporaryDirectory(prefix="guardrail-lora-") as tmp:
            staged = Path(tmp)
            (staged / CONFIG_NAME).symlink_to((self._adapter_dir / CONFIG_NAME).resolve())
            (staged / FINAL_WEIGHTS_NAME).symlink_to(weights.resolve())
            return MLXSUT(model_id=model_id, adapter_path=str(staged))

    @property
    def model_id(self) -> str:
        """Base model + which adapter + WHICH CHECKPOINT of it.

        The checkpoint suffix is the part that matters: without it, two runs of two
        different checkpoints of the same training run are indistinguishable in
        verdicts.jsonl, and the reduction number cannot be attributed to any specific
        set of weights.
        """
        stem = f"{self._inner._model_id}+{self._adapter_dir.name}"
        return f"{stem}@{self._label}" if self._label is not None else stem

    @property
    def provenance(self) -> dict:
        """The training settings that produced these weights, for BENCHMARKS.md."""
        keys = (
            "model",
            "data",
            "fine_tune_type",
            "num_layers",
            "iters",
            "batch_size",
            "learning_rate",
            "mask_prompt",
            "lora_parameters",
        )
        out = {k: self._config[k] for k in keys if k in self._config}
        out["checkpoint"] = self._label
        out["adapter_path"] = str(self._adapter_dir)
        return out

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Response:
        """Delegate verbatim, then relabel with this SUT's finer-grained model_id.

        Deliberately NOT a reimplementation: the chat template, the sampler and the
        token accounting must be identical to the baseline run, or the measured delta
        includes harness differences and stops being attributable to the fine-tune.
        """
        r = self._inner.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return Response(
            text=r.text,
            model_id=self.model_id,
            latency_s=r.latency_s,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
        )
