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
          "lora" -> MLXSUT(adapter_path=$GUARDRAIL_ADAPTER_PATH)  (Phase 6)
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
from guardrail.sut.mlx_sut import DEFAULT_MODEL_ID, MLXSUT
from guardrail.sut.mock import MockSUT

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TEMPERATURE",
    "MLXSUT",
    "MockSUT",
    "Response",
    "SUT",
    "VALID_SUTS",
    "get_sut",
]

VALID_SUTS = ("mock", "mlx", "lora")
DEFAULT_ADAPTER_PATH = "adapters"


def get_sut(name: str | None = None) -> SUT:
    """Build the SUT named by `name`, or by $GUARDRAIL_SUT, defaulting to mock."""
    resolved = (name or os.getenv("GUARDRAIL_SUT") or "mock").strip().lower()

    if resolved == "mock":
        return MockSUT()

    if resolved == "mlx":
        return MLXSUT(model_id=os.getenv("SUT_MODEL") or DEFAULT_MODEL_ID)

    if resolved == "lora":
        adapter = os.getenv("GUARDRAIL_ADAPTER_PATH") or DEFAULT_ADAPTER_PATH
        # Fail here with a sentence a human can act on, rather than letting mlx
        # raise something opaque several frames down.
        if not Path(adapter).exists():
            raise FileNotFoundError(
                f"GUARDRAIL_SUT=lora but no adapter at {adapter!r}. "
                "Fine-tuning is Phase 6; until then use GUARDRAIL_SUT=mlx for the "
                "base model, or set GUARDRAIL_ADAPTER_PATH."
            )
        return MLXSUT(
            model_id=os.getenv("SUT_MODEL") or DEFAULT_MODEL_ID,
            adapter_path=adapter,
        )

    raise ValueError(
        f"Unknown GUARDRAIL_SUT={resolved!r}. Valid options: {', '.join(VALID_SUTS)}."
    )
