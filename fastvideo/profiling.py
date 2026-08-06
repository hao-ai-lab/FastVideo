# SPDX-License-Identifier: Apache-2.0
"""Env-gated, bounded torch.profiler capture for denoising loops.

Enable with ``FASTVIDEO_PROFILE_STEPS=N`` (capture N steps, skipping step 0 as
warmup) and optionally ``FASTVIDEO_PROFILE_DIR`` (default ``/tmp``). Writes one
JSON + one table dump per rank, then generation continues unprofiled.

Usage in a denoising stage:

    profiler = StepProfiler.from_env(num_steps=len(timesteps), tag="h3")
    try:
        for index, ... in enumerate(...):
            profiler.on_step(index)
            ...
    finally:
        profiler.close()

Profiling must never break generation: malformed env disables it, dump
failures log and continue, and ``close()`` stops a still-running profiler on
exception paths. ``record_shapes`` is enabled so gemms can be attributed to
attention vs MLP by their dimensions post-hoc — deliberately no
``record_function`` scopes in model code, which would graph-break under
torch.compile.
"""

from __future__ import annotations

import json
import os

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class StepProfiler:
    """Bounded per-step profiler; a no-op when disabled."""

    def __init__(self, profile_steps: int, out_dir: str, tag: str) -> None:
        self._steps = profile_steps
        self._out_dir = out_dir
        self._tag = tag
        self._profiler: torch.profiler.profile | None = None
        if profile_steps > 0:
            self._profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            )

    @classmethod
    def from_env(cls, num_steps: int, tag: str = "denoise") -> "StepProfiler":
        try:
            requested = int(os.environ.get("FASTVIDEO_PROFILE_STEPS", "0"))
        except ValueError:
            requested = 0
        # clamp so the dump step is always reached (step 0 is skipped as warmup)
        profile_steps = min(max(requested, 0), max(num_steps - 2, 0))
        return cls(profile_steps, os.environ.get("FASTVIDEO_PROFILE_DIR", "/tmp"), tag)

    def on_step(self, index: int) -> None:
        """Call at the top of each denoising iteration."""
        if self._profiler is None:
            return
        if index == 1:
            self._profiler.start()
        elif index == 1 + self._steps:
            self._profiler.stop()
            self._dump()
            self._profiler = None

    def _dump(self) -> None:
        try:
            assert self._profiler is not None
            rank = os.environ.get("RANK", "0")
            stem = f"{self._out_dir}/{self._tag}_profile_rank{rank}"
            averages = self._profiler.key_averages(group_by_input_shape=True)
            rows = [{
                "name": e.key,
                "shapes": str(e.input_shapes),
                "cuda_us": e.device_time_total,
                "count": e.count,
            } for e in averages]
            with open(f"{stem}.json", "w") as fh:
                json.dump({"profiled_steps": self._steps, "rows": rows}, fh)
            with open(f"{stem}.txt", "w") as fh:
                fh.write(averages.table(sort_by="cuda_time_total", row_limit=60))
        except Exception:  # noqa: BLE001 -- profiling must never kill generation
            logger.exception("profile dump failed; continuing generation")

    def close(self) -> None:
        """Stop a still-running profiler (exception paths); no dump of partial data."""
        if self._profiler is not None:
            try:
                self._profiler.stop()
            except Exception:  # noqa: BLE001
                pass
            self._profiler = None
