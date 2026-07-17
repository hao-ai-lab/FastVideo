# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the dynamic VideoBatchScheduler (no GPU needed)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config,
    SelfForcingWanT2V480PConfig,
    WanT2V480PConfig,
)
from fastvideo.entrypoints.openai.batching import VideoBatchScheduler


class _FakeBatchGenerator:

    def __init__(self):
        self.calls: list[list[dict]] = []

    def generate_video_batch(self, request_kwargs, *, _capture_postprocess_errors=False):
        self.calls.append([dict(item) for item in request_kwargs])
        return [{"prompts": item["prompt"], "video_path": item["output_path"]} for item in request_kwargs]


class _BlockingFirstCallGenerator(_FakeBatchGenerator):
    """Blocks the first generate_video_batch call until released.

    Lets a test deterministically hold one generation "in flight" while more
    requests queue up behind it.
    """

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self._first_call = True

    def generate_video_batch(self, request_kwargs, *, _capture_postprocess_errors=False):
        if self._first_call:
            self._first_call = False
            self.started.set()
            assert self.release.wait(timeout=10)
        return super().generate_video_batch(request_kwargs, _capture_postprocess_errors=_capture_postprocess_errors)


def _batch_scheduler_args(**overrides):
    defaults = dict(
        model_path="test-model",
        batching_mode="dynamic",
        batching_max_size=2,
        batching_delay_ms=25.0,
        enable_batching_metrics=False,
        pipeline_config=PipelineConfig(supports_dynamic_batching=True),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _job_kwargs(tmp_path, index: int) -> dict:
    return {
        "prompt": f"p{index}",
        "height": 256,
        "width": 256,
        "num_frames": 1,
        "num_inference_steps": 2,
        "seed": index,
        "output_path": str(tmp_path / f"video_{index}.mp4"),
        "save_video": False,
    }


def test_expired_deadline_coalesces_jobs_queued_behind_inflight_generation(tmp_path):
    """Jobs whose delay deadline expired while a generation was in flight must
    still coalesce (greedy drain), not dispatch as batches of 1."""

    async def run():
        generator = _BlockingFirstCallGenerator()
        scheduler = VideoBatchScheduler(generator, _batch_scheduler_args(batching_max_size=4, batching_delay_ms=25.0))
        await scheduler.start()
        try:
            loop = asyncio.get_running_loop()
            first = asyncio.ensure_future(scheduler.submit("req-1", _job_kwargs(tmp_path, 1)))
            await asyncio.wait_for(loop.run_in_executor(None, generator.started.wait, 5), timeout=10)
            rest = [asyncio.ensure_future(scheduler.submit(f"req-{i}", _job_kwargs(tmp_path, i))) for i in (2, 3, 4)]
            # Let the in-queue jobs outlive the 25 ms batching window while the
            # first generation is still blocking the scheduler.
            await asyncio.sleep(0.1)
            generator.release.set()
            await asyncio.wait_for(asyncio.gather(first, *rest), timeout=10)
        finally:
            await scheduler.stop()
        return [[item["prompt"] for item in call] for call in generator.calls]

    assert asyncio.run(run()) == [["p1"], ["p2", "p3", "p4"]]


def test_submit_after_stop_raises(tmp_path):

    async def run():
        scheduler = VideoBatchScheduler(_FakeBatchGenerator(), _batch_scheduler_args())
        await scheduler.start()
        await scheduler.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            await scheduler.submit("req-1", _job_kwargs(tmp_path, 1))

    asyncio.run(run())


def test_stop_fails_jobs_still_queued_without_hanging(tmp_path):
    """stop() during an in-flight generation must fail the still-queued jobs
    instead of stranding their futures forever."""

    async def run():
        generator = _BlockingFirstCallGenerator()
        scheduler = VideoBatchScheduler(generator, _batch_scheduler_args())
        await scheduler.start()
        loop = asyncio.get_running_loop()
        first = asyncio.ensure_future(scheduler.submit("req-1", _job_kwargs(tmp_path, 1)))
        await asyncio.wait_for(loop.run_in_executor(None, generator.started.wait, 5), timeout=10)
        queued = [asyncio.ensure_future(scheduler.submit(f"req-{i}", _job_kwargs(tmp_path, i))) for i in (2, 3)]
        await asyncio.sleep(0)  # let the queued submits enqueue
        stop_task = asyncio.ensure_future(scheduler.stop())
        await asyncio.sleep(0)  # let stop() mark the scheduler stopped
        generator.release.set()

        result = await asyncio.wait_for(first, timeout=10)
        assert result["prompts"] == "p1"
        for task in queued:
            with pytest.raises(RuntimeError, match="stopped before dispatch"):
                await asyncio.wait_for(task, timeout=10)
        await asyncio.wait_for(stop_task, timeout=10)

    asyncio.run(run())


def test_dispatch_exception_fails_all_group_futures(tmp_path):

    class _ExplodingGenerator:

        def __init__(self):
            self.calls = []

        def generate_video_batch(self, request_kwargs, *, _capture_postprocess_errors=False):
            self.calls.append([item["prompt"] for item in request_kwargs])
            raise ValueError("generation exploded")

    async def run():
        generator = _ExplodingGenerator()
        scheduler = VideoBatchScheduler(generator, _batch_scheduler_args())
        await scheduler.start()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("req-1", _job_kwargs(tmp_path, 1)),
                    scheduler.submit("req-2", _job_kwargs(tmp_path, 2)),
                    return_exceptions=True,
                ),
                timeout=10,
            )
        finally:
            await scheduler.stop()
        return results, generator.calls

    results, calls = asyncio.run(run())

    assert calls == [["p1", "p2"], ["p1"], ["p2"]]
    assert len(results) == 2
    for result in results:
        assert isinstance(result, ValueError)
        assert "generation exploded" in str(result)


def test_dispatch_exception_retries_group_and_preserves_successful_requests(tmp_path):

    class _PartiallyExplodingGenerator(_FakeBatchGenerator):

        def generate_video_batch(self, request_kwargs, *, _capture_postprocess_errors=False):
            self.calls.append([dict(item) for item in request_kwargs])
            if len(request_kwargs) > 1:
                raise RuntimeError("merged generation failed")
            if request_kwargs[0]["prompt"] == "p2":
                raise ValueError("bad second request")
            return [{"prompts": request_kwargs[0]["prompt"]}]

    async def run():
        generator = _PartiallyExplodingGenerator()
        scheduler = VideoBatchScheduler(generator, _batch_scheduler_args())
        await scheduler.start()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("req-1", _job_kwargs(tmp_path, 1)),
                    scheduler.submit("req-2", _job_kwargs(tmp_path, 2)),
                    return_exceptions=True,
                ),
                timeout=10,
            )
        finally:
            await scheduler.stop()
        return results, [[item["prompt"] for item in call] for call in generator.calls]

    results, calls = asyncio.run(run())

    assert calls == [["p1", "p2"], ["p1"], ["p2"]]
    assert results[0] == {"prompts": "p1"}
    assert isinstance(results[1], ValueError)
    assert "bad second request" in str(results[1])


def test_run_crash_fails_inflight_and_queued_futures(tmp_path):
    """If the _run task ever dies, every waiting future must fail instead of
    hanging, and later submits must be rejected."""

    async def run():
        scheduler = VideoBatchScheduler(_FakeBatchGenerator(), _batch_scheduler_args())

        async def boom(first):
            raise RuntimeError("kaboom")

        scheduler._collect_batch = boom  # type: ignore[method-assign]
        await scheduler.start()
        results = await asyncio.wait_for(
            asyncio.gather(
                scheduler.submit("req-1", _job_kwargs(tmp_path, 1)),
                scheduler.submit("req-2", _job_kwargs(tmp_path, 2)),
                return_exceptions=True,
            ),
            timeout=10,
        )
        with pytest.raises(RuntimeError, match="stopped"):
            await scheduler.submit("req-3", _job_kwargs(tmp_path, 3))
        await scheduler.stop()
        assert scheduler._task is None
        return results

    results = asyncio.run(run())

    assert len(results) == 2
    for result in results:
        assert isinstance(result, RuntimeError)
        assert "crashed" in str(result)


def test_dispatch_cancellation_fails_every_collected_future(tmp_path):

    async def run():
        scheduler = VideoBatchScheduler(_FakeBatchGenerator(), _batch_scheduler_args())
        dispatch_started = asyncio.Event()
        dispatched_prompts = []

        async def block_dispatch(batch):
            dispatched_prompts.extend(job.kwargs["prompt"] for job in batch)
            dispatch_started.set()
            await asyncio.Event().wait()

        scheduler._dispatch = block_dispatch  # type: ignore[method-assign]
        await scheduler.start()
        submitted = [
            asyncio.ensure_future(scheduler.submit(f"req-{index}", _job_kwargs(tmp_path, index)))
            for index in (1, 2)
        ]
        await asyncio.wait_for(dispatch_started.wait(), timeout=10)
        assert scheduler._task is not None
        scheduler._task.cancel()

        results = await asyncio.wait_for(
            asyncio.gather(*submitted, return_exceptions=True),
            timeout=10,
        )
        await scheduler.stop()
        return dispatched_prompts, results

    dispatched_prompts, results = asyncio.run(run())

    assert dispatched_prompts == ["p1", "p2"]
    assert len(results) == 2
    for result in results:
        assert isinstance(result, RuntimeError)
        assert "crashed" in str(result)


def test_scheduler_start_rejects_pipeline_without_capability(tmp_path):

    async def run():
        scheduler = VideoBatchScheduler(
            _FakeBatchGenerator(),
            _batch_scheduler_args(pipeline_config=PipelineConfig()),
        )
        with pytest.raises(RuntimeError, match="does not support dynamic request batching"):
            await scheduler.start()
        assert scheduler._task is None

    asyncio.run(run())


def test_dynamic_batching_capability_gate_defaults():
    # Default: every pipeline is excluded until its family opts in.
    assert PipelineConfig().dynamic_batching_supported() is False
    # The verified generic Wan denoising path opts in.
    assert WanT2V480PConfig().dynamic_batching_supported() is True
    # DMD and causal denoising variants are always vetoed (shared-RNG risk),
    # even though they inherit the Wan opt-in flag.
    assert FastWan2_1_T2V_480P_Config().dynamic_batching_supported() is False
    assert SelfForcingWanT2V480PConfig().dynamic_batching_supported() is False


def test_scheduler_metrics_report_compatibility_rejection_reason(monkeypatch, tmp_path):
    scheduler = VideoBatchScheduler(
        _FakeBatchGenerator(),
        _batch_scheduler_args(enable_batching_metrics=True),
    )
    info_logs = []

    def fake_sampling_param(kwargs):
        return SamplingParam(
            prompt=kwargs["prompt"],
            guidance_scale=kwargs["guidance_scale"],
        ), {}

    monkeypatch.setattr(scheduler, "_sampling_param_from_kwargs", fake_sampling_param)
    monkeypatch.setattr(
        "fastvideo.entrypoints.openai.batching.logger.info",
        lambda message, *args: info_logs.append((message, args)),
    )
    base_kwargs = _job_kwargs(tmp_path, 1)
    base_kwargs["guidance_scale"] = 1.0
    candidate_kwargs = _job_kwargs(tmp_path, 2)
    candidate_kwargs["guidance_scale"] = 3.0
    base = SimpleNamespace(request_id="req-1", kwargs=base_kwargs)
    candidate = SimpleNamespace(request_id="req-2", kwargs=candidate_kwargs)

    assert scheduler._jobs_are_compatible(base, candidate) is False
    assert info_logs == [(
        "Dynamic batch candidate rejected: request_id=%s reason=%s",
        ("req-2", "sampling_params.guidance_scale"),
    )]


def test_scheduler_batches_matching_pipeline_overrides_and_separates_mismatches(tmp_path):
    scheduler = VideoBatchScheduler(_FakeBatchGenerator(), _batch_scheduler_args())
    first_kwargs = _job_kwargs(tmp_path, 1)
    first_kwargs["embedded_cfg_scale"] = 7.5
    matching_kwargs = _job_kwargs(tmp_path, 2)
    matching_kwargs["embedded_cfg_scale"] = 7.5
    mismatched_kwargs = _job_kwargs(tmp_path, 3)
    mismatched_kwargs["embedded_cfg_scale"] = 6.0

    first = SimpleNamespace(request_id="req-1", kwargs=first_kwargs)
    matching = SimpleNamespace(request_id="req-2", kwargs=matching_kwargs)
    mismatched = SimpleNamespace(request_id="req-3", kwargs=mismatched_kwargs)

    assert scheduler._jobs_are_compatible(first, matching) is True
    assert scheduler._jobs_are_compatible(first, mismatched) is False


def test_scheduler_retries_only_failed_postprocess_item(tmp_path):

    class _PartialPostprocessGenerator:

        def __init__(self):
            self.calls = []

        def generate_video_batch(self, request_kwargs, *, _capture_postprocess_errors=False):
            prompts = [item["prompt"] for item in request_kwargs]
            self.calls.append(prompts)
            if len(request_kwargs) > 1:
                assert _capture_postprocess_errors is True
                return [
                    {
                        "prompts": prompts[0],
                        "video_path": "first-batched.mp4",
                    },
                    {"_postprocess_error": OSError("second save failed")},
                ]
            return [{"prompts": prompts[0], "video_path": "second-retry.mp4"}]

    async def run():
        generator = _PartialPostprocessGenerator()
        scheduler = VideoBatchScheduler(generator, _batch_scheduler_args())
        await scheduler.start()
        try:
            results = await asyncio.gather(
                scheduler.submit("req-1", _job_kwargs(tmp_path, 1)),
                scheduler.submit("req-2", _job_kwargs(tmp_path, 2)),
            )
        finally:
            await scheduler.stop()
        return generator.calls, results

    calls, results = asyncio.run(run())

    assert calls == [["p1", "p2"], ["p2"]]
    assert [result["video_path"] for result in results] == ["first-batched.mp4", "second-retry.mp4"]
