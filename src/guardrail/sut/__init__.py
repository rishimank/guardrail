"""The SUT seam's public surface: the interface, the implementations, the factory.

Everything downstream imports from here (`from guardrail.sut import get_sut`) and
never names a concrete class. That is what makes Phase 6 a config change rather
than a code change: flip GUARDRAIL_SUT=lora and the harness is unchanged.

Importing this module is CHEAP and SAFE on any platform. `mlx_sut` is safe to
import on linux because it only pulls mlx lazily inside MLXSUT.__init__ — so
GitHub Actions and the Docker image can `import guardrail.sut` and run MockSUT
without mlx installed at all.

PSEUDOCODE
    1. Re-export the contract (SUT, Response) and the implementations.
    2. get_sut(name=None):
       a. name defaults to $GUARDRAIL_SUT, which itself defaults to "mock" —
          the free/offline/deterministic option. A wrong default should cost $0
          and download nothing, not silently load 1.6GB.
       b. "mock" -> MockSUT()
          "mlx"  -> MLXSUT()                      (base Qwen)
          "lora" -> LoRASUT($GUARDRAIL_ADAPTER_PATH,
                            checkpoint=$GUARDRAIL_ADAPTER_CHECKPOINT)   (Phase 6)
       c. Anything else -> ValueError listing the valid names.
"""

from __future__ import annotations

import os
from pathlib import Path

from guardrail.sut.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Response,
    SUT,
)
from guardrail.sut.lora_sut import CONFIG_NAME, LoRASUT, checkpoints
from guardrail.sut.mlx_sut import DEFAULT_MODEL_ID, MLXSUT
from guardrail.sut.mock import MockSUT

__all__ = [
    "DEFAULT_ADAPTER_PATH",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TEMPERATURE",
    "LoRASUT",
    "MLXSUT",
    "MockSUT",
    "Response",
    "SUT",
    "VALID_SUTS",
    "checkpoints",
    "get_sut",
]

VALID_SUTS = ("mock", "mlx", "lora")
# A specific training run, not the folder that holds them. `adapters/` is a parent
# directory containing v1, v2, ...; pointing mlx_lm at it fails deep inside the loader
# on a missing adapter_config.json, so the default names a run and get_sut() checks
# for that file rather than for mere directory existence.
DEFAULT_ADAPTER_PATH = "adapters/v2"


def get_sut(name: str | None = None) -> SUT:
    """Build the SUT named by `name`, or by $GUARDRAIL_SUT, defaulting to mock."""
    resolved = (name or os.getenv("GUARDRAIL_SUT") or "mock").strip().lower()

    if resolved == "mock":
        return MockSUT()

    if resolved == "mlx":
        return MLXSUT(model_id=os.getenv("SUT_MODEL") or DEFAULT_MODEL_ID)

    if resolved == "lora":
        adapter = Path(os.getenv("GUARDRAIL_ADAPTER_PATH") or DEFAULT_ADAPTER_PATH)
        # Fail here with a sentence a human can act on, rather than letting mlx
        # raise something opaque several frames down. Checking for the config file
        # (not just the directory) is what catches the easy mistake of pointing at
        # `adapters/` — which exists, but is a parent of several training runs.
        if not (adapter / CONFIG_NAME).exists():
            raise FileNotFoundError(
                f"GUARDRAIL_SUT=lora but {adapter / CONFIG_NAME} is missing, so "
                f"{str(adapter)!r} is not a trained adapter. Set "
                "GUARDRAIL_ADAPTER_PATH to a specific run (e.g. 'adapters/v2'), or "
                "use GUARDRAIL_SUT=mlx to measure the base model."
            )

        # Which checkpoint is an experimental choice, not a detail: validation loss
        # is not monotonic across a run, so the final weights are not automatically
        # the ones worth measuring. Unset = the final weights (mlx_lm's own default).
        raw = os.getenv("GUARDRAIL_ADAPTER_CHECKPOINT")
        try:
            ckpt = int(raw) if raw and raw.strip() else None
        except ValueError:
            raise ValueError(
                f"GUARDRAIL_ADAPTER_CHECKPOINT={raw!r} is not an integer iteration. "
                f"Available in {str(adapter)!r}: {sorted(checkpoints(adapter))}."
            ) from None

        return LoRASUT(
            adapter,
            model_id=os.getenv("SUT_MODEL") or DEFAULT_MODEL_ID,
            checkpoint=ckpt,
        )

    raise ValueError(
        f"Unknown GUARDRAIL_SUT={resolved!r}. Valid options: {', '.join(VALID_SUTS)}."
    )
