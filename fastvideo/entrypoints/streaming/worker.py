# SPDX-License-Identifier: Apache-2.0
"""Per-GPU worker subprocess entry for :class:`SubprocessGpuPool`.

The pool manages binding, lifecycle, and message dispatch in the parent
process. The worker constructs its :class:`VideoGenerator` from a typed
:class:`GeneratorConfig`, runs the two-segment warmup so both
initial-segment and continuation-branch compile graphs are hot, and
then loops on the job queue.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any

from fastvideo.api.schema import (
    GeneratorConfig,
    GenerationRequest,
    InputConfig,
    OutputConfig,
    SamplingConfig,
    WarmupConfig,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Synthetic warmup dimensions: small enough to keep boot fast, big enough
# to exercise the real shape-dependent compile paths. Keep in sync with
# WarmupConfig if these become user-tunable.
_WARMUP_NUM_FRAMES = 8
_WARMUP_HEIGHT = 256
_WARMUP_WIDTH = 256
_WARMUP_NUM_INFERENCE_STEPS = 1


def worker_main(
    *,
    gpu_id: int,
    worker_id: str,
    generator_config: GeneratorConfig,
    warmup_config: WarmupConfig,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: Any,
) -> None:  # pragma: no cover - exercised via integration only
    """Per-worker subprocess entry.

    Runs inside the child spawned by ``SubprocessGpuPool``. Blocking
    ``VideoGenerator`` construction + generation happens here, not in
    the parent's event loop.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(config=generator_config)
        if warmup_config.enabled:
            _warmup_worker(generator, warmup_config)
        result_queue.put({"kind": "ready", "worker_id": worker_id})
    except Exception as exc:
        result_queue.put({"kind": "error", "error": repr(exc)})
        return

    while not shutdown_event.is_set():
        try:
            item = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        job_id = item["job_id"]
        request = item["request"]
        try:
            result = generator.generate(request)
            result_queue.put({
                "kind": "result",
                "job_id": job_id,
                "result": result,
            })
        except Exception as exc:
            result_queue.put({
                "kind": "error",
                "job_id": job_id,
                "error": repr(exc),
            })


def _warmup_worker(
    generator: Any,
    warmup_config: WarmupConfig,
) -> None:
    """Run two synthetic generations so both compile branches are primed.

    Segment 1 is a fresh start (no continuation state) and exercises
    the initial-segment graph. Segment 2 feeds segment 1's continuation
    state back in so the conditioning branch is also compiled before
    the first user request lands.
    """
    sampling = SamplingConfig(
        num_frames=_WARMUP_NUM_FRAMES,
        height=_WARMUP_HEIGHT,
        width=_WARMUP_WIDTH,
        num_inference_steps=_WARMUP_NUM_INFERENCE_STEPS,
    )
    seg1 = GenerationRequest(
        prompt=warmup_config.prompt,
        sampling=sampling,
        inputs=InputConfig(),
        output=OutputConfig(save_video=False, return_frames=False, return_state=True),
    )
    seg1_result = generator.generate(seg1)

    seg2 = GenerationRequest(
        prompt=warmup_config.prompt,
        sampling=sampling,
        inputs=InputConfig(),
        output=OutputConfig(save_video=False, return_frames=False),
        state=_extract_continuation_state(seg1_result),
    )
    generator.generate(seg2)


def _extract_continuation_state(result: Any) -> Any:
    state = getattr(result, "state", None)
    if state is None and isinstance(result, dict):
        state = result.get("state")
    return state


__all__ = ["worker_main"]
