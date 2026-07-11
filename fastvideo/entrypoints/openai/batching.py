# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.batching.signature import can_dynamic_batch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _VideoBatchJob:
    request_id: str
    kwargs: dict[str, Any]
    future: asyncio.Future
    enqueue_time: float


class VideoBatchScheduler:
    """Async FIFO scheduler for OpenAI-compatible video generation."""

    def __init__(self, generator: Any, fastvideo_args: FastVideoArgs) -> None:
        self._generator = generator
        self._fastvideo_args = fastvideo_args
        self._queue: asyncio.Queue[_VideoBatchJob | None] = asyncio.Queue()
        self._pending: deque[_VideoBatchJob] = deque()
        self._task: asyncio.Task | None = None
        self._stopped = False

    @property
    def enabled(self) -> bool:
        return self._fastvideo_args.batching_mode == "dynamic" and self._fastvideo_args.batching_max_size > 1

    async def start(self) -> None:
        if self._task is not None:
            return
        pipeline_config = self._fastvideo_args.pipeline_config
        if self.enabled and not pipeline_config.dynamic_batching_supported():
            raise RuntimeError(
                f"Pipeline config {type(pipeline_config).__name__} does not support dynamic request batching; "
                "run with --batching-mode disabled, or use a pipeline that opts in via "
                "supports_dynamic_batching=True (DMD/causal denoising variants are always excluded)")
        self._task = asyncio.create_task(self._run(), name="fastvideo-video-batch-scheduler")
        self._task.add_done_callback(self._on_run_done)

    def _on_run_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            exc: BaseException | None = asyncio.CancelledError()
        else:
            exc = task.exception()
        if exc is None:
            return
        logger.error("Video batch scheduler task died; failing all pending requests", exc_info=exc)
        self._stopped = True
        self._drain_and_fail_waiting(RuntimeError(f"Video batch scheduler crashed and cannot dispatch: {exc!r}"))

    def _drain_and_fail_waiting(self, error: BaseException) -> None:
        """Fail every job still waiting in ``_pending`` or ``_queue``."""
        while self._pending:
            job = self._pending.popleft()
            if not job.future.done():
                job.future.set_exception(error)
        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if queued is not None and not queued.future.done():
                queued.future.set_exception(error)

    async def stop(self) -> None:
        self._stopped = True
        await self._queue.put(None)
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, request_id: str, kwargs: dict[str, Any]) -> Any:
        if self._stopped:
            raise RuntimeError("Video batch scheduler is stopped; cannot submit new requests")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(
            _VideoBatchJob(
                request_id=request_id,
                kwargs=dict(kwargs),
                future=future,
                enqueue_time=time.perf_counter(),
            ))
        return await future

    async def _run(self) -> None:
        while not self._stopped:
            job = await self._get_next_job()
            if job is None:
                break
            batch = await self._collect_batch(job)
            await self._dispatch(batch)

        # Fail jobs stranded in _pending AND still sitting in _queue (e.g.
        # submitted while the final dispatch was in flight when stop() landed).
        self._drain_and_fail_waiting(RuntimeError("Video batch scheduler stopped before dispatch"))

    async def _get_next_job(self) -> _VideoBatchJob | None:
        if self._pending:
            return self._pending.popleft()
        return await self._queue.get()

    async def _collect_batch(self, first: _VideoBatchJob) -> list[_VideoBatchJob]:
        batch = [first]
        max_size = self._fastvideo_args.batching_max_size
        delay_s = max(0.0, self._fastvideo_args.batching_delay_ms / 1000.0)
        deadline = first.enqueue_time + delay_s

        while len(batch) < max_size:
            timeout = deadline - time.perf_counter()
            if timeout > 0:
                try:
                    candidate = await asyncio.wait_for(self._get_next_job(), timeout=timeout)
                except (TimeoutError, asyncio.TimeoutError):
                    # asyncio.TimeoutError is only aliased to the builtin on
                    # Python 3.11+; catch both so 3.10 does not kill _run.
                    break
            else:
                # delay=0 means "don't wait", and an already-expired deadline
                # means the first job waited behind an in-flight generation:
                # in both cases greedily drain whatever is already queued so
                # max_size still coalesces instead of dispatching batches of 1.
                if self._pending:
                    candidate = self._pending.popleft()
                else:
                    try:
                        candidate = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            if candidate is None:
                await self._queue.put(None)
                break
            if self._jobs_are_compatible(batch[0], candidate):
                batch.append(candidate)
                continue
            self._pending.appendleft(candidate)
            break
        return batch

    async def _dispatch(self, batch: list[_VideoBatchJob]) -> None:
        loop = asyncio.get_running_loop()
        request_ids = [job.request_id for job in batch]
        queue_wait_ms = (time.perf_counter() - min(job.enqueue_time for job in batch)) * 1000.0
        if self._fastvideo_args.enable_batching_metrics:
            logger.info(
                "Dispatching video batch: request_ids=%s size=%d queue_wait_ms=%.2f",
                request_ids,
                len(batch),
                queue_wait_ms,
            )

        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._generator.generate_video_batch([job.kwargs for job in batch]),
            )
        except Exception as exc:
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
            return

        if len(results) != len(batch):
            error = RuntimeError(f"Video batch returned {len(results)} results for {len(batch)} requests")
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(error)
            return

        for job, result in zip(batch, results, strict=True):
            if not job.future.done():
                job.future.set_result(result)

    def _jobs_are_compatible(self, base: _VideoBatchJob, candidate: _VideoBatchJob) -> bool:
        try:
            base_sampling, base_extra = self._sampling_param_from_kwargs(base.kwargs)
            candidate_sampling, candidate_extra = self._sampling_param_from_kwargs(candidate.kwargs)
        except Exception as exc:
            # Deduplicated per message so a systematically broken probe (which
            # silently disables batching) is visible without log spam.
            logger.warning_once("Dynamic batch compatibility probe failed; treating requests as "
                                f"incompatible (batching disabled for these requests): {exc!r}")
            return False
        return can_dynamic_batch(
            base_sampling,
            candidate_sampling,
            base_extra=base_extra,
            candidate_extra=candidate_extra,
        ).can_batch

    def _sampling_param_from_kwargs(self, kwargs: dict[str, Any]) -> tuple[SamplingParam, dict[str, Any]]:
        sampling_param = SamplingParam.from_pretrained(self._fastvideo_args.model_path)
        updates = dict(kwargs)
        prompt = updates.pop("prompt", None)
        extra: dict[str, Any] = {}
        for key in (
                "ltx2_audio_latents",
                "ltx2_audio_clean_latent",
                "ltx2_audio_denoise_mask",
                "audio_num_frames",
                "video_position_offset_sec",
        ):
            if key in updates:
                extra[key] = updates.pop(key)
        sampling_param.update(updates)
        sampling_param.prompt = prompt
        return sampling_param, extra
