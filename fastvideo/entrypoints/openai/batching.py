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
        self._task = asyncio.create_task(self._run(), name="fastvideo-video-batch-scheduler")

    async def stop(self) -> None:
        self._stopped = True
        await self._queue.put(None)
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, request_id: str, kwargs: dict[str, Any]) -> Any:
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

        while self._pending:
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("Video batch scheduler stopped before dispatch"))

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
            if timeout <= 0 and delay_s > 0:
                break
            try:
                candidate = await asyncio.wait_for(self._get_next_job(), timeout=max(0.0, timeout))
            except TimeoutError:
                break
            if candidate is None:
                await self._queue.put(None)
                break
            if self._jobs_are_compatible(batch[0], candidate):
                batch.append(candidate)
                continue
            self._pending.append(candidate)
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
        except Exception:
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
