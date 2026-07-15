"""MLXSUT — the real system under test: Qwen2.5-3B-Instruct-4bit, local, via MLX.

This is the model the whole project attacks. It runs on Apple Silicon out of the
HuggingFace cache, costs $0/call, and needs no network after the first download.
That $0 is the budget strategy: the model under test is free, so the entire spend
goes to the judge.

`mlx_lm` is an OPTIONAL dependency (Apple-Silicon-only). It is imported lazily
inside __init__, never at module import time, so that `import guardrail.sut` keeps
working on the linux Docker image and on GitHub Actions runners, which run MockSUT.

Measured 2026-07-15 on an M1 Pro / 16 GB: load ~1.2s, ~71 tok/s steady-state.

PSEUDOCODE
    1. __init__: import mlx_lm lazily, then load(model_id, adapter_path) ONCE and
       keep (model, tokenizer) on the instance. Loading per-prompt would add
       seconds of pure overhead to every one of 500 prompts.
       - adapter_path is None here. Phase 6's LoRASUT passes a path; that is the
         entire difference, and the reason this seam was built first.
    2. generate(prompt):
       a. Wrap the raw user turn in Qwen's chat template. The caller passes plain
          text and must never know Qwen wants ChatML — that is this class's job.
       b. Build a sampler at the requested temperature (0.0 = greedy = the same
          answer every run, which is what makes the eval reproducible).
       c. Time the call, count tokens, return a Response.
    3. The Response is shape-identical to MockSUT's, so nothing downstream can tell
       which one it is talking to — except by reading model_id.
"""

from __future__ import annotations

import time
from typing import Any

from guardrail.sut.base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, Response

DEFAULT_MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"


class MLXSUT:
    """Satisfies the SUT protocol with a real local Qwen via mlx-lm."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_path: str | None = None,
    ) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "mlx-lm is not installed. It is Apple-Silicon-only and optional: "
                'install with `pip install -e ".[mlx]"`, or use MockSUT instead '
                "(GUARDRAIL_SUT=mock), which is what CI and Docker run."
            ) from e

        self._model_id = model_id
        self._adapter_path = adapter_path
        # Loaded once, reused for every prompt. ~1.2s, ~1.6GB resident.
        # load() is typed as returning a 2- OR 3-tuple depending on return_config,
        # so unpacking directly does not typecheck. Index instead of `type: ignore`.
        loaded = load(model_id, adapter_path=adapter_path)
        self._model, self._tokenizer = loaded[0], loaded[1]

    @property
    def model_id(self) -> str:
        # A LoRA run must never be mistaken for a baseline run in a result file.
        if self._adapter_path is not None:
            return f"{self._model_id}+{self._adapter_path}"
        return self._model_id

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Response:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        # Qwen2.5-Instruct is trained on ChatML. Handing it a bare string makes it
        # autocomplete rather than answer, which would silently corrupt every
        # measurement in the project.
        templated: Any = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )

        t0 = time.perf_counter()
        text = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=templated,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )
        latency_s = time.perf_counter() - t0

        prompt_tokens = len(templated) if isinstance(templated, list) else len(
            self._tokenizer.encode(templated)
        )

        return Response(
            text=text,
            model_id=self.model_id,
            latency_s=latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(self._tokenizer.encode(text)),
        )
