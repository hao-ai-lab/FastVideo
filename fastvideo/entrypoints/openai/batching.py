# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any

from fastvideo.api.compat import (
    legacy_generate_call_to_request,
    request_to_pipeline_overrides,
    request_to_sampling_param,
)
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
            try:
                await self._task
            except (Exception, asyncio.CancelledError):
                # _on_run_done already logged the crash and failed queued jobs.
                pass
            finally:
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
            batch = [job]
            try:
                batch = await self._collect_batch(job)
                await self._dispatch(batch)
            except BaseException:
                # Every collected job is absent from both waiting containers.
                # Keep the whole batch reachable so the crash drain in
                # _on_run_done settles every unfinished future.
                self._pending.extend(batch)
                raise

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
                partial(self._generator.generate_video_batch, [job.kwargs for job in batch],
                        _capture_postprocess_errors=True),
            )
        except Exception as exc:
            await self._handle_batch_failure(batch, exc)
            return

        if len(results) != len(batch):
            error = RuntimeError(f"Video batch returned {len(results)} results for {len(batch)} requests")
            await self._handle_batch_failure(batch, error)
            return

        failed_postprocessing: list[tuple[_VideoBatchJob, Exception]] = []
        for job, result in zip(batch, results, strict=True):
            postprocess_error = result.get("_postprocess_error") if isinstance(result, dict) else None
            if isinstance(postprocess_error, Exception):
                failed_postprocessing.append((job, postprocess_error))
                continue
            if not job.future.done():
                job.future.set_result(result)
        if failed_postprocessing:
            jobs = [job for job, _error in failed_postprocessing]
            errors = "; ".join(str(error) for _job, error in failed_postprocessing)
            await self._retry_jobs_individually(
                jobs,
                RuntimeError(f"Per-request postprocessing failed: {errors}"),
            )

    async def _handle_batch_failure(self, batch: list[_VideoBatchJob], error: Exception) -> None:
        if len(batch) == 1:
            if not batch[0].future.done():
                batch[0].future.set_exception(error)
            return
        await self._retry_jobs_individually(batch, error)

    async def _retry_jobs_individually(
        self,
        batch: list[_VideoBatchJob],
        error: Exception,
    ) -> None:
        logger.warning(
            "Video batch failed; retrying %d requests individually: %s",
            len(batch),
            error,
        )
        loop = asyncio.get_running_loop()
        for job in batch:
            try:
                results = await loop.run_in_executor(
                    None,
                    partial(self._generator.generate_video_batch, [job.kwargs]),
                )
                if len(results) != 1:
                    raise RuntimeError(f"Single video request returned {len(results)} results")
            except Exception as item_error:
                if not job.future.done():
                    job.future.set_exception(item_error)
            else:
                if not job.future.done():
                    job.future.set_result(results[0])

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
        compatibility = can_dynamic_batch(
            base_sampling,
            candidate_sampling,
            base_extra=base_extra,
            candidate_extra=candidate_extra,
        )
        if self._fastvideo_args.enable_batching_metrics and not compatibility.can_batch:
            logger.info(
                "Dynamic batch candidate rejected: request_id=%s reason=%s",
                candidate.request_id,
                compatibility.reason or "incompatible",
            )
        return compatibility.can_batch

    def _sampling_param_from_kwargs(self, kwargs: dict[str, Any]) -> tuple[SamplingParam, dict[str, Any]]:
        base_sampling_param = SamplingParam.from_pretrained(self._fastvideo_args.model_path)
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

        request = legacy_generate_call_to_request(
            prompt,
            base_sampling_param,
            legacy_kwargs=updates,
        )
        sampling_param = request_to_sampling_param(
            request,
            model_path=self._fastvideo_args.model_path,
        )
        sampling_param.prompt = request.prompt
        pipeline_overrides = request_to_pipeline_overrides(request)
        if pipeline_overrides:
            extra["pipeline_overrides"] = pipeline_overrides
        return sampling_param, extra
