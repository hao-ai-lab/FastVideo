import os
import threading
import warnings
from types import SimpleNamespace

import pytest
import torch

import fastvideo.entrypoints.video_generator as video_generator_module
from fastvideo.api import (
    ConfigValidationError,
    GenerationRequest,
    GenerationResult,
    GeneratorConfig,
    InputConfig,
    OutputConfig,
    SamplingConfig,
    load_run_config,
)
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.entrypoints.video_generator import VideoGenerator, _resolve_output_size
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines import ForwardBatch
from fastvideo.worker.gpu_worker import Worker
from fastvideo.worker.ray_distributed_executor import RayDistributedExecutor


def _new_video_generator() -> VideoGenerator:
    # Bypass __init__ since we only test a pure helper method.
    return VideoGenerator.__new__(VideoGenerator)


def _new_runtime_video_generator() -> VideoGenerator:
    generator = _new_video_generator()
    generator.fastvideo_args = SimpleNamespace(
        model_path="test-model",
        prompt_txt=None,
        workload_type=SimpleNamespace(value="t2v"),
    )
    generator.executor = SimpleNamespace(
        set_log_queue=lambda queue: None,
        clear_log_queue=lambda: None,
    )
    generator.config = None
    return generator


def _batching_fastvideo_args(**overrides):
    defaults = dict(
        model_path="test-model",
        prompt_txt=None,
        workload_type=SimpleNamespace(value="t2v"),
        batching_mode="dynamic",
        batching_max_size=4,
        batching_config=None,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        output_type="latent",
        pin_cpu_memory=False,
        VSA_sparsity=0.0,
        pipeline_config=PipelineConfig(supports_dynamic_batching=True),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_from_fastvideo_args(monkeypatch):
    captured = {}

    def fake_from_fastvideo_args(cls, fastvideo_args, *, log_queue=None):
        generator = cls.__new__(cls)
        generator.fastvideo_args = fastvideo_args
        generator.executor = None
        generator.config = None
        captured["fastvideo_args"] = fastvideo_args
        captured["log_queue"] = log_queue
        return generator

    monkeypatch.setattr(
        VideoGenerator,
        "from_fastvideo_args",
        classmethod(fake_from_fastvideo_args),
    )
    return captured


def _patch_fastvideo_args_from_kwargs(monkeypatch):
    captured = {}

    def fake_from_kwargs(cls, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            model_path=kwargs["model_path"],
            num_gpus=kwargs["num_gpus"],
            workload_type=WorkloadType.from_string(kwargs.get("workload_type", "t2v")),
        )

    monkeypatch.setattr(
        "fastvideo.api.compat.FastVideoArgs.from_kwargs",
        classmethod(fake_from_kwargs),
    )
    return captured


def _patch_sampling_param_from_pretrained(monkeypatch):

    def fake_from_pretrained(cls, model_path):
        return cls()

    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(fake_from_pretrained))


class _NoCpuMaterializationOutput:

    @property
    def shape(self):
        raise AssertionError("metadata-only output path should not inspect output shape")

    def cpu(self):
        raise AssertionError("metadata-only output path should not move output to CPU")

    def __mul__(self, other):
        raise AssertionError("metadata-only output path should not build frames")


def _single_video_args(output_type="pil"):
    return SimpleNamespace(
        output_type=output_type,
        pin_cpu_memory=False,
        VSA_sparsity=0.0,
        pipeline_config=SimpleNamespace(flow_shift=1.0, embedded_cfg_scale=1.0),
        workload_type=SimpleNamespace(value="t2v"),
    )


def _small_sampling_param(*, save_video=False, return_frames=False):
    return SamplingParam(
        num_frames=2,
        height=16,
        width=16,
        fps=8,
        num_inference_steps=1,
        save_video=save_video,
        return_frames=return_frames,
    )


def _single_video_output_batch(output, *, extra=None):
    return SimpleNamespace(
        output=output,
        extra=extra or {},
        logging_info=None,
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_decoded=None,
    )


def _single_video_generator(output_batch, fastvideo_args):
    generator = _new_video_generator()
    generator.fastvideo_args = fastvideo_args
    generator.executor = SimpleNamespace(execute_forward=lambda batch, args: output_batch)
    generator.config = None
    return generator


def test_resolve_output_size_uses_produced_pixel_geometry() -> None:
    """Report refined dimensions instead of the base request geometry."""
    samples = torch.empty(1, 3, 5, 64, 96)
    assert _resolve_output_size(samples, (32, 48, 1), pixel_output=True) == (64, 96, 5)
    assert _resolve_output_size(samples, (32, 48, 1), pixel_output=False) == (32, 48, 1)


def test_prepare_output_path_file_sanitization(tmp_path):
    vg = _new_video_generator()
    target_dir = tmp_path / "dir"
    raw_path = target_dir / "inv:al*id?.mp4"

    result = vg._prepare_output_path(str(raw_path), prompt="ignored")

    assert os.path.dirname(result) == str(target_dir)
    assert os.path.basename(result) == "invalid.mp4"
    assert os.path.isdir(target_dir)


def test_prepare_output_path_directory_prompt_derived(tmp_path):
    vg = _new_video_generator()
    out_dir = tmp_path / "outputs"
    prompt = "Hello:/\\*?<>| world"

    result = vg._prepare_output_path(str(out_dir), prompt=prompt)

    assert os.path.dirname(result) == str(out_dir)
    # spaces are preserved (collapsed) by sanitizer; here it becomes "Hello world.mp4"
    assert os.path.basename(result) == "Hello world.mp4"
    assert os.path.isdir(out_dir)


def test_prepare_output_path_non_mp4_treated_as_dir(tmp_path):
    vg = _new_video_generator()
    weird_dir = tmp_path / "foo.gif"
    prompt = "My Video"

    result = vg._prepare_output_path(str(weird_dir), prompt=prompt)

    assert os.path.dirname(result) == str(weird_dir)
    assert os.path.basename(result) == "My Video.mp4"
    assert os.path.isdir(weird_dir)


def test_prepare_output_path_uniqueness_suffix(tmp_path):
    vg = _new_video_generator()
    out_dir = tmp_path / "outputs"
    prompt = "Sample Name"

    first = vg._prepare_output_path(str(out_dir), prompt=prompt)
    # simulate existing file
    os.makedirs(os.path.dirname(first), exist_ok=True)
    with open(first, "wb") as f:
        f.write(b"")

    second = vg._prepare_output_path(str(out_dir), prompt=prompt)
    assert os.path.basename(second) == "Sample Name_1.mp4"

    # simulate second existing file as well
    with open(second, "wb") as f:
        f.write(b"")
    third = vg._prepare_output_path(str(out_dir), prompt=prompt)
    assert os.path.basename(third) == "Sample Name_2.mp4"


def test_prepare_output_path_empty_prompt_fallback(tmp_path):
    vg = _new_video_generator()
    out_dir = tmp_path / "outputs"
    bad_prompt = ":/\\*?<>|   .."  # sanitizes to empty, should fallback to "video"

    result = vg._prepare_output_path(str(out_dir), prompt=bad_prompt)

    assert os.path.dirname(result) == str(out_dir)
    assert os.path.basename(result) == "output.mp4"


def test_generate_single_video_metadata_only_skips_output_materialization(monkeypatch, tmp_path):
    output_batch = _single_video_output_batch(
        _NoCpuMaterializationOutput(),
        extra={"peak_memory_mb": 42.0},
    )
    fastvideo_args = _single_video_args()
    generator = _single_video_generator(output_batch, fastvideo_args)
    empty_calls = []
    real_empty = video_generator_module.torch.empty

    def record_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return real_empty(*args, **kwargs)

    def fail_make_grid(*args, **kwargs):
        raise AssertionError("metadata-only output path should not build frame grids")

    monkeypatch.setattr(video_generator_module.torch, "empty", record_empty)
    monkeypatch.setattr(video_generator_module.torchvision.utils, "make_grid", fail_make_grid)

    result = generator._generate_single_video(
        prompt="metadata only",
        sampling_param=_small_sampling_param(save_video=False, return_frames=False),
        fastvideo_args=fastvideo_args,
        output_path=str(tmp_path / "unused.mp4"),
    )

    assert result["samples"] is None
    assert result["frames"] is None
    assert result["video_path"] is None
    assert result["peak_memory_mb"] == 42.0
    assert empty_calls == [((0,), {"device": "cpu"})]


def test_run_forward_batch_overlaps_pixel_allocation_with_forward(monkeypatch):
    generator = _new_video_generator()
    forward_started = threading.Event()
    allocation_started = threading.Event()
    expected_shape = (1, 3, 2, 16, 16)

    def execute_forward(batch, fastvideo_args):
        forward_started.set()
        assert allocation_started.wait(timeout=5)
        return ForwardBatch(data_type=batch.data_type, output=torch.ones(expected_shape))

    generator.executor = SimpleNamespace(execute_forward=execute_forward)
    real_empty = video_generator_module.torch.empty

    def track_empty(shape, *args, **kwargs):
        if shape == expected_shape:
            assert forward_started.wait(timeout=5)
            allocation_started.set()
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(video_generator_module.torch, "empty", track_empty)
    batch = ForwardBatch(
        data_type="video",
        prompt="overlap",
        height=16,
        width=16,
        num_frames=2,
        save_video=False,
        return_frames=True,
    )

    output_batch, _, _ = generator._run_forward_batch(
        batch,
        _single_video_args(),
    )

    assert allocation_started.is_set()
    assert output_batch.output is not None
    assert output_batch.output.shape == expected_shape
    assert output_batch.output.device.type == "cpu"


def test_generate_single_video_return_frames_still_materializes_output(tmp_path):
    output = torch.ones((1, 3, 2, 16, 16), dtype=torch.float32) * 0.5
    output_batch = _single_video_output_batch(output)
    fastvideo_args = _single_video_args()
    generator = _single_video_generator(output_batch, fastvideo_args)

    result = generator._generate_single_video(
        prompt="return frames",
        sampling_param=_small_sampling_param(save_video=False, return_frames=True),
        fastvideo_args=fastvideo_args,
        output_path=str(tmp_path / "unused.mp4"),
    )

    torch.testing.assert_close(result["samples"], output)
    assert len(result["frames"]) == 2
    assert result["video_path"] is None


def test_generate_single_video_save_video_still_builds_frames(monkeypatch, tmp_path):
    output = torch.ones((1, 3, 2, 16, 16), dtype=torch.float32) * 0.5
    output_batch = _single_video_output_batch(output)
    fastvideo_args = _single_video_args()
    generator = _single_video_generator(output_batch, fastvideo_args)
    saved = {}

    def fake_mimsave(path, frames, *, fps, format):
        saved["path"] = path
        saved["frame_count"] = len(frames)
        saved["fps"] = fps
        saved["format"] = format

    monkeypatch.setattr(video_generator_module.imageio, "mimsave", fake_mimsave)

    output_path = str(tmp_path / "saved.mp4")
    result = generator._generate_single_video(
        prompt="save only",
        sampling_param=_small_sampling_param(save_video=True, return_frames=False),
        fastvideo_args=fastvideo_args,
        output_path=output_path,
    )

    assert result["samples"] is None
    assert result["frames"] is None
    assert result["video_path"] == output_path
    assert saved == {
        "path": output_path,
        "frame_count": 2,
        "fps": 8,
        "format": "mp4",
    }


def test_generate_single_video_audio_only_metadata_returns_audio_without_frames(tmp_path):
    audio = torch.zeros((16,), dtype=torch.float32)
    output_batch = _single_video_output_batch(
        _NoCpuMaterializationOutput(),
        extra={
            "audio_only": True,
            "audio": audio,
            "audio_sample_rate": 44100,
        },
    )
    fastvideo_args = _single_video_args()
    generator = _single_video_generator(output_batch, fastvideo_args)

    result = generator._generate_single_video(
        prompt="audio only",
        sampling_param=_small_sampling_param(save_video=False, return_frames=False),
        fastvideo_args=fastvideo_args,
        output_path=str(tmp_path / "unused.mp4"),
    )

    assert result["samples"] is None
    assert result["frames"] is None
    assert result["audio"] is audio
    assert result["audio_sample_rate"] == 44100


def test_generate_single_video_audio_only_save_skips_placeholder_materialization(tmp_path):
    audio = torch.zeros((16,), dtype=torch.float32)
    output_batch = _single_video_output_batch(
        _NoCpuMaterializationOutput(),
        extra={
            "audio_only": True,
            "audio": audio,
            "audio_sample_rate": 44100,
        },
    )
    fastvideo_args = _single_video_args()
    generator = _single_video_generator(output_batch, fastvideo_args)
    output_path = str(tmp_path / "audio.mp4")

    result = generator._generate_single_video(
        prompt="audio only",
        sampling_param=_small_sampling_param(save_video=True, return_frames=False),
        fastvideo_args=fastvideo_args,
        output_path=output_path,
    )

    assert result["samples"] is None
    assert result["frames"] is None
    assert result["audio"] is audio
    assert result["audio_sample_rate"] == 44100
    assert result["video_path"] == str(tmp_path / "audio.wav")


def test_generate_single_video_ray_audio_only_save_preserves_worker_metadata(monkeypatch, tmp_path):
    audio = torch.zeros((16,), dtype=torch.float32)
    worker_output = ForwardBatch(
        data_type="audio",
        output=torch.ones((1, 3, 1, 8, 8)),
        extra={
            "audio_only": True,
            "audio": audio,
            "audio_sample_rate": 44100,
        },
    )
    worker = Worker.__new__(Worker)
    worker.fastvideo_args = SimpleNamespace()
    worker.pipeline = SimpleNamespace(forward=lambda batch, args: worker_output)

    monkeypatch.setattr(RayDistributedExecutor, "__abstractmethods__", frozenset())
    executor = RayDistributedExecutor.__new__(RayDistributedExecutor)
    executor.shutdown = lambda: None

    def collective_rpc(method, *, kwargs, **_):
        assert method == "execute_forward"
        return [worker.execute_forward(**kwargs)]

    executor.collective_rpc = collective_rpc
    fastvideo_args = _single_video_args()
    generator = _new_video_generator()
    generator.fastvideo_args = fastvideo_args
    generator.executor = executor
    generator.config = None
    written = {}

    def fake_write_pcm_wav(path, samples, sample_rate):
        written.update(path=path, samples=samples, sample_rate=sample_rate)

    monkeypatch.setattr(generator, "_write_pcm_wav", fake_write_pcm_wav)
    sampling_param = _small_sampling_param(save_video=True, return_frames=False)
    sampling_param.data_type = "audio"

    result = generator._generate_single_video(
        prompt="audio only",
        sampling_param=sampling_param,
        fastvideo_args=fastvideo_args,
        output_path=str(tmp_path / "audio.mp4"),
    )

    assert worker_output.output is not None
    assert worker_output.output.numel() == 0
    assert result["samples"] is None
    assert result["frames"] is None
    assert result["audio"] is audio
    assert result["audio_sample_rate"] == 44100
    assert result["video_path"] == str(tmp_path / "audio.wav")
    assert written["path"] == str(tmp_path / "audio.wav")
    assert written["samples"] is audio
    assert written["sample_rate"] == 44100


def test_generate_single_video_latent_metadata_skips_cpu_materialization(tmp_path):
    output_batch = _single_video_output_batch(_NoCpuMaterializationOutput())
    fastvideo_args = _single_video_args(output_type="latent")
    generator = _single_video_generator(output_batch, fastvideo_args)

    result = generator._generate_single_video(
        prompt="latent metadata",
        sampling_param=_small_sampling_param(save_video=False, return_frames=False),
        fastvideo_args=fastvideo_args,
        output_path=str(tmp_path / "unused.mp4"),
    )

    assert result["samples"] is None
    assert result["frames"] is None
    assert result["video_path"] is None

def test_generate_prepared_work_items_merges_compatible_latent_requests(monkeypatch, tmp_path):
    vg = _new_video_generator()
    vg.fastvideo_args = _batching_fastvideo_args()
    calls = []

    def fake_device_memory(gpu_id):
        return 48.0

    def fake_run_forward(batch, fastvideo_args):
        calls.append(batch)
        batch_size = len(batch.prompt) if isinstance(batch.prompt, list) else 1
        output = torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4, 1, 1, 1)
        return ForwardBatch(data_type=batch.data_type, output=output, extra={"peak_memory_mb": 1.0}), 0.5, 10.0

    monkeypatch.setattr(
        "fastvideo.batching.admission.BatchAdmissionController._get_device_memory_gb",
        staticmethod(fake_device_memory),
    )
    monkeypatch.setattr(vg, "_run_forward_batch", fake_run_forward)

    first = SamplingParam(prompt="one", height=8, width=8, num_frames=1, seed=11, return_frames=True, save_video=False)
    second = SamplingParam(prompt="two", height=8, width=8, num_frames=1, seed=22, return_frames=True, save_video=False)
    work_items = [
        vg._prepare_generation_work_item("one", first, vg.fastvideo_args, output_path=str(tmp_path / "one.mp4")),
        vg._prepare_generation_work_item("two", second, vg.fastvideo_args, output_path=str(tmp_path / "two.mp4")),
    ]

    results = vg._generate_prepared_work_items(work_items)

    assert len(calls) == 1
    assert calls[0].prompt == ["one", "two"]
    assert calls[0].seeds == [11, 22]
    assert [result["prompts"] for result in results] == ["one", "two"]
    assert [result["samples"].shape for result in results] == [(1, 4, 1, 1, 1), (1, 4, 1, 1, 1)]
    assert results[0]["samples"].flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert results[1]["samples"].flatten().tolist() == [4.0, 5.0, 6.0, 7.0]


@pytest.mark.parametrize(
    ("return_frame_values", "save_video_values"),
    [
        ((False, True), (False, False)),
        ((True, False), (False, False)),
        ((False, False), (False, True)),
        ((False, False), (True, False)),
    ],
)
def test_merge_work_items_aggregates_output_requirements_in_both_orders(
    tmp_path,
    return_frame_values,
    save_video_values,
):
    vg = _new_video_generator()
    vg.fastvideo_args = _batching_fastvideo_args()
    prompts = ("one", "two")
    params = [
        _small_sampling_param(save_video=save_video, return_frames=return_frames)
        for return_frames, save_video in zip(return_frame_values, save_video_values, strict=True)
    ]
    work_items = [
        vg._prepare_generation_work_item(
            prompt,
            param,
            vg.fastvideo_args,
            output_path=str(tmp_path / f"{prompt}.mp4"),
        ) for prompt, param in zip(prompts, params, strict=True)
    ]

    merged = vg._merge_work_items(work_items)

    assert merged.batch.return_frames is any(return_frame_values)
    assert merged.batch.save_video is any(save_video_values)


def test_generate_prepared_work_items_falls_back_for_incompatible_requests(monkeypatch, tmp_path):
    vg = _new_video_generator()
    vg.fastvideo_args = _batching_fastvideo_args()
    calls = []

    def fake_run_forward(batch, fastvideo_args):
        calls.append(batch)
        output = torch.zeros((1, 4, 1, 1, 1), dtype=torch.float32)
        return ForwardBatch(data_type=batch.data_type, output=output), 0.5, 10.0

    monkeypatch.setattr(vg, "_run_forward_batch", fake_run_forward)

    first = SamplingParam(prompt="one", height=8, width=8, num_frames=1, guidance_scale=1.0, save_video=False)
    second = SamplingParam(prompt="two", height=8, width=8, num_frames=1, guidance_scale=3.0, save_video=False)
    work_items = [
        vg._prepare_generation_work_item("one", first, vg.fastvideo_args, output_path=str(tmp_path / "one.mp4")),
        vg._prepare_generation_work_item("two", second, vg.fastvideo_args, output_path=str(tmp_path / "two.mp4")),
    ]

    results = vg._generate_prepared_work_items(work_items)

    assert len(calls) == 2
    assert all(isinstance(call.prompt, str) for call in calls)
    assert [result["prompts"] for result in results] == ["one", "two"]


def test_generate_prepared_work_items_retries_failed_group_per_prompt(monkeypatch, tmp_path):
    vg = _new_video_generator()
    vg.fastvideo_args = _batching_fastvideo_args()
    calls = []

    def fake_run_forward(batch, fastvideo_args):
        calls.append(batch.prompt)
        if isinstance(batch.prompt, list):
            raise RuntimeError("merged generation failed")
        if batch.prompt == "two":
            raise ValueError("bad second prompt")
        output = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
        return ForwardBatch(data_type=batch.data_type, output=output), 0.5, 10.0

    monkeypatch.setattr(vg, "_run_forward_batch", fake_run_forward)
    params = [
        SamplingParam(prompt="one", height=8, width=8, num_frames=1, return_frames=True, save_video=False),
        SamplingParam(prompt="two", height=8, width=8, num_frames=1, return_frames=True, save_video=False),
    ]
    work_items = [
        vg._prepare_generation_work_item(
            param.prompt,
            param,
            vg.fastvideo_args,
            output_path=str(tmp_path / f"{param.prompt}.mp4"),
        ) for param in params
    ]

    results = vg._generate_prepared_work_items(work_items, tolerate_failures=True)

    assert calls == [["one", "two"], "one", "two"]
    assert results[0]["prompts"] == "one"
    assert results[0]["samples"].flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert results[1] == {"error": "bad second prompt", "prompt": "two"}


def test_video_generator_validates_batching_config_before_executor_start(tmp_path):
    config_path = tmp_path / "batching.json"
    config_path.write_text('{"rules": {"model": "test-model"}}', encoding="utf-8")
    executor_started = False

    class _Executor:

        def __init__(self, *args, **kwargs):
            nonlocal executor_started
            executor_started = True

    with pytest.raises(ValueError, match="rules must be a list"):
        VideoGenerator(_batching_fastvideo_args(batching_config=str(config_path)), _Executor, log_stats=False)
    assert executor_started is False


def test_generate_prepared_work_items_uses_startup_cached_admission_config(monkeypatch, tmp_path):
    config_path = tmp_path / "batching.json"
    config_path.write_text(
        '{"rules": [{"model": "test-model", "max_batch_size": 1}]}',
        encoding="utf-8",
    )

    class _Executor:

        def __init__(self, *args, **kwargs):
            pass

    args = _batching_fastvideo_args(batching_config=str(config_path))
    generator = VideoGenerator(args, _Executor, log_stats=False)
    config_path.unlink()
    request_args = _batching_fastvideo_args(batching_config=str(config_path))
    calls = []

    def fake_run_forward(batch, fastvideo_args):
        calls.append(batch.prompt)
        output = torch.zeros((1, 4, 1, 1, 1), dtype=torch.float32)
        return ForwardBatch(data_type=batch.data_type, output=output), 0.5, 10.0

    monkeypatch.setattr(generator, "_run_forward_batch", fake_run_forward)
    params = [
        SamplingParam(prompt="one", height=8, width=8, num_frames=1, return_frames=True, save_video=False),
        SamplingParam(prompt="two", height=8, width=8, num_frames=1, return_frames=True, save_video=False),
    ]
    work_items = [
        generator._prepare_generation_work_item(
            param.prompt,
            param,
            request_args,
            output_path=str(tmp_path / f"{param.prompt}.mp4"),
        ) for param in params
    ]

    results = generator._generate_prepared_work_items(work_items)

    assert calls == ["one", "two"]
    assert [result["prompts"] for result in results] == ["one", "two"]


def test_generate_video_batch_routes_compat_kwargs(monkeypatch, tmp_path):
    vg = _new_video_generator()
    vg.fastvideo_args = _batching_fastvideo_args()
    calls = []

    def fake_device_memory(gpu_id):
        return 48.0

    def fake_run_forward(batch, fastvideo_args):
        calls.append((batch, fastvideo_args))
        output = torch.zeros((len(batch.prompt), 4, 1, 1, 1), dtype=torch.float32)
        return ForwardBatch(data_type=batch.data_type, output=output), 0.5, 10.0

    monkeypatch.setattr(
        "fastvideo.batching.admission.BatchAdmissionController._get_device_memory_gb",
        staticmethod(fake_device_memory),
    )
    monkeypatch.setattr(vg, "_run_forward_batch", fake_run_forward)

    results = vg.generate_video_batch([
        {
            "prompt": "one",
            "height": 8,
            "width": 8,
            "num_frames": 1,
            "embedded_cfg_scale": 7.5,
            "save_video": False,
            "return_frames": True,
            "output_path": str(tmp_path / "one.mp4"),
        },
        {
            "prompt": "two",
            "height": 8,
            "width": 8,
            "num_frames": 1,
            "embedded_cfg_scale": 7.5,
            "save_video": False,
            "return_frames": True,
            "output_path": str(tmp_path / "two.mp4"),
        },
    ])

    assert len(calls) == 1
    batch, fastvideo_args = calls[0]
    assert batch.prompt == ["one", "two"]
    assert fastvideo_args.pipeline_config.embedded_cfg_scale == 7.5
    assert [result["prompts"] for result in results] == ["one", "two"]


def test_from_config_normalizes_and_translates(monkeypatch):
    captured = _patch_from_fastvideo_args(monkeypatch)
    _patch_fastvideo_args_from_kwargs(monkeypatch)
    config = GeneratorConfig(model_path="test-model")
    config.engine.num_gpus = 2
    config.pipeline.workload_type = "t2v"

    generator = VideoGenerator.from_config(config)

    assert captured["fastvideo_args"].model_path == "test-model"
    assert captured["fastvideo_args"].num_gpus == 2
    assert captured["fastvideo_args"].workload_type.value == "t2v"
    assert generator.config == config


def test_from_file_loads_generator_from_run_config(tmp_path, monkeypatch):
    captured = _patch_from_fastvideo_args(monkeypatch)
    _patch_fastvideo_args_from_kwargs(monkeypatch)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "  engine:\n"
        "    num_gpus: 3\n"
        "request:\n"
        "  prompt: hello\n",
        encoding="utf-8",
    )

    VideoGenerator.from_file(str(config_path))

    assert captured["fastvideo_args"].model_path == "test-model"
    assert captured["fastvideo_args"].num_gpus == 3


def test_from_pretrained_convenience_kwargs_do_not_warn(monkeypatch):
    captured = _patch_from_fastvideo_args(monkeypatch)
    fastvideo_args_capture = _patch_fastvideo_args_from_kwargs(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generator = VideoGenerator.from_pretrained(
            "test-model",
            num_gpus=4,
            use_fsdp_inference=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
        )

    assert not caught
    assert captured["fastvideo_args"].model_path == "test-model"
    assert captured["fastvideo_args"].num_gpus == 4
    assert fastvideo_args_capture["kwargs"]["use_fsdp_inference"] is False
    assert fastvideo_args_capture["kwargs"]["text_encoder_cpu_offload"] is True
    assert fastvideo_args_capture["kwargs"]["pin_cpu_memory"] is True
    assert fastvideo_args_capture["kwargs"]["dit_cpu_offload"] is False
    assert fastvideo_args_capture["kwargs"]["vae_cpu_offload"] is False
    assert generator.config is not None
    assert generator.config.model_path == "test-model"
    assert generator.config.engine.num_gpus == 4


def test_from_pretrained_legacy_only_kwargs_warn(monkeypatch):
    captured = _patch_from_fastvideo_args(monkeypatch)
    _patch_fastvideo_args_from_kwargs(monkeypatch)

    with pytest.warns(DeprecationWarning, match="legacy-only kwargs"):
        generator = VideoGenerator.from_pretrained(
            "test-model",
            num_gpus=4,
            workload_type="t2v",
        )

    assert captured["fastvideo_args"].model_path == "test-model"
    assert captured["fastvideo_args"].num_gpus == 4
    assert captured["fastvideo_args"].workload_type.value == "t2v"
    assert generator.config is not None
    assert generator.config.pipeline.workload_type == "t2v"


def test_generate_uses_typed_request_path(monkeypatch):
    generator = _new_runtime_video_generator()
    _patch_sampling_param_from_pretrained(monkeypatch)
    captured = {}

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured["prompt"] = prompt
        captured["sampling_param"] = sampling_param
        captured["kwargs"] = kwargs
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    result = generator.generate(
        GenerationRequest(
            prompt="hello world",
            sampling=SamplingConfig(num_frames=81, height=480, width=832),
        ))

    assert isinstance(result, GenerationResult)
    assert captured["prompt"] == "hello world"
    assert captured["sampling_param"].num_frames == 81
    assert captured["sampling_param"].height == 480
    assert captured["sampling_param"].width == 832
    assert result.video_path == "outputs/test.mp4"


def test_generate_rejects_stage_override_outside_registered_stage(monkeypatch) -> None:
    """Validate typed stage overrides at the public generation entrypoint."""
    generator = _new_runtime_video_generator()
    monkeypatch.setattr(
        "fastvideo.registry.get_preset_selection",
        lambda _model_path: ("lingbot_video_moe_refiner_t2v", "lingbot_video"),
    )
    with pytest.raises(ConfigValidationError, match="stage_overrides.denoise.t_thresh"):
        generator.generate({
            "prompt": "hello world",
            "stage_overrides": {
                "denoise": {
                    "t_thresh": 0.7
                }
            },
        })


def test_generate_preserves_schema_defaults_for_dataclass_request(monkeypatch):
    generator = _new_runtime_video_generator()
    captured = {}

    def fake_from_pretrained(cls, model_path):
        return cls(
            negative_prompt="model default",
            num_frames=61,
            height=448,
            width=832,
        )

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured["sampling_param"] = sampling_param
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(fake_from_pretrained))
    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    generator.generate(
        GenerationRequest(
            prompt="hello world",
            negative_prompt=None,
            sampling=SamplingConfig(num_frames=125, height=720, width=1280),
        ))

    assert captured["sampling_param"].negative_prompt is None
    assert captured["sampling_param"].num_frames == 125
    assert captured["sampling_param"].height == 720
    assert captured["sampling_param"].width == 1280


def test_generate_mapping_request_preserves_model_defaults_for_omitted_fields(monkeypatch, ):
    generator = _new_runtime_video_generator()
    captured = {}

    def fake_from_pretrained(cls, model_path):
        return cls(
            negative_prompt="model default",
            num_frames=61,
            height=448,
            width=832,
            fps=16,
            guidance_scale=3.0,
        )

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured["sampling_param"] = sampling_param
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(fake_from_pretrained))
    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    generator.generate({
        "prompt": "hello world",
    })

    assert captured["sampling_param"].negative_prompt == "model default"
    assert captured["sampling_param"].num_frames == 61
    assert captured["sampling_param"].height == 448
    assert captured["sampling_param"].width == 832
    assert captured["sampling_param"].fps == 16
    assert captured["sampling_param"].guidance_scale == 3.0


def test_generate_honors_post_load_request_mutations(monkeypatch, tmp_path):
    generator = _new_runtime_video_generator()
    captured = {}
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "request:\n"
        "  prompt: hello world\n",
        encoding="utf-8",
    )

    def fake_from_pretrained(cls, model_path):
        return cls(seed=1024, num_frames=61, height=448, width=832)

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured["sampling_param"] = sampling_param
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(fake_from_pretrained))
    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    config = load_run_config(config_path)
    config.request.sampling.seed = 7

    generator.generate(config.request)

    assert captured["sampling_param"].seed == 7


def test_generate_honors_post_load_mutations_matching_schema_defaults(
    monkeypatch,
    tmp_path,
):
    generator = _new_runtime_video_generator()
    captured = {}
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "request:\n"
        "  prompt: hello world\n",
        encoding="utf-8",
    )

    def fake_from_pretrained(cls, model_path):
        return cls(guidance_scale=3.0)

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured["sampling_param"] = sampling_param
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(fake_from_pretrained))
    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    config = load_run_config(config_path)
    config.request.sampling.guidance_scale = 1.0

    generator.generate(config.request)

    assert captured["sampling_param"].guidance_scale == 1.0


def test_generate_removes_deleted_loaded_stage_overrides(monkeypatch, tmp_path):
    generator = _new_runtime_video_generator()
    captured = {}
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "generator:\n"
        "  model_path: test-model\n"
        "request:\n"
        "  prompt: hello world\n"
        "  stage_overrides:\n"
        "    refine:\n"
        "      t_thresh: 0.8\n",
        encoding="utf-8",
    )

    def fake_from_pretrained(cls, model_path):
        return cls(t_thresh=0.5)

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured["sampling_param"] = sampling_param
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(fake_from_pretrained))
    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    config = load_run_config(config_path)
    del config.request.stage_overrides["refine"]

    generator.generate(config.request)

    assert captured["sampling_param"].t_thresh == 0.5


def test_generate_video_legacy_call_uses_legacy_impl(monkeypatch):
    generator = _new_runtime_video_generator()
    captured = {}

    def fake_generate_video_impl(
        prompt=None,
        sampling_param=None,
        mouse_cond=None,
        keyboard_cond=None,
        grid_sizes=None,
        **kwargs,
    ):
        captured["prompt"] = prompt
        captured["sampling_param"] = sampling_param
        captured["mouse_cond"] = mouse_cond
        captured["keyboard_cond"] = keyboard_cond
        captured["grid_sizes"] = grid_sizes
        captured["kwargs"] = kwargs
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    with pytest.warns(DeprecationWarning):
        result = generator.generate_video(
            prompt="legacy prompt",
            num_frames=49,
            output_path="outputs/legacy",
            save_video=False,
            log_queue="queue-token",
        )

    assert captured["prompt"] == "legacy prompt"
    assert captured["sampling_param"].num_frames == 49
    assert captured["sampling_param"].output_path == "outputs/legacy"
    assert captured["sampling_param"].save_video is False
    assert result["video_path"] == "outputs/test.mp4"


def test_generate_video_legacy_call_routes_compat_kwargs(monkeypatch):
    generator = _new_runtime_video_generator()
    generator.fastvideo_args.pipeline_config = SimpleNamespace(embedded_cfg_scale=1.0)
    captured = {}

    def fake_generate_video_impl(
        prompt=None,
        sampling_param=None,
        mouse_cond=None,
        keyboard_cond=None,
        grid_sizes=None,
        **kwargs,
    ):
        captured["prompt"] = prompt
        captured["sampling_param"] = sampling_param
        captured["kwargs"] = kwargs
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    with pytest.warns(DeprecationWarning):
        result = generator.generate_video(
            prompt="legacy prompt",
            neg_prompt="custom negative",
            embedded_cfg_scale=7.5,
        )

    assert captured["prompt"] == "legacy prompt"
    assert captured["sampling_param"].negative_prompt == "custom negative"
    assert captured["kwargs"]["fastvideo_args"].pipeline_config.embedded_cfg_scale == 7.5
    assert not hasattr(captured["sampling_param"], "embedded_cfg_scale")
    assert result["video_path"] == "outputs/test.mp4"


def test_generate_batch_prompt_file_returns_typed_results(tmp_path, monkeypatch):
    generator = _new_runtime_video_generator()
    _patch_sampling_param_from_pretrained(monkeypatch)
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    captured_prompts = []

    def fake_generate_single_video(prompt, sampling_param=None, **kwargs):
        captured_prompts.append(prompt)
        return {"prompts": prompt, "video_path": kwargs["output_path"]}

    monkeypatch.setattr(generator, "_generate_single_video", fake_generate_single_video)

    results = generator.generate({
        "inputs": {
            "prompt_path": str(prompt_file)
        },
        "output": {
            "output_path": str(output_dir),
            "save_video": False,
            "return_frames": False,
        },
    })

    assert isinstance(results, list)
    assert [result.prompt for result in results] == ["first prompt", "second prompt"]
    assert [result.prompt_index for result in results] == [0, 1]
    assert captured_prompts == ["first prompt", "second prompt"]


def test_generate_prompt_file_clears_source_path_and_batches_prompts(monkeypatch, tmp_path):
    generator = _new_video_generator()
    generator.fastvideo_args = _batching_fastvideo_args()
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    calls = []

    def fake_run_forward(batch, fastvideo_args):
        calls.append(batch)
        output = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1, 1)
        return ForwardBatch(data_type=batch.data_type, output=output), 0.5, 10.0

    monkeypatch.setattr(generator, "_run_forward_batch", fake_run_forward)
    sampling_param = SamplingParam(
        prompt_path=str(prompt_file),
        output_path=str(tmp_path / "outputs"),
        height=8,
        width=8,
        num_frames=1,
        return_frames=True,
        save_video=False,
    )

    results = generator._generate_video_impl(
        sampling_param=sampling_param,
        fastvideo_args=generator.fastvideo_args,
    )

    assert len(calls) == 1
    assert calls[0].prompt == ["first prompt", "second prompt"]
    assert [result["prompt"] for result in results] == ["first prompt", "second prompt"]
    assert sampling_param.prompt_path == str(prompt_file)


def test_generate_batched_request_fans_out_media_inputs(monkeypatch):
    generator = _new_runtime_video_generator()
    _patch_sampling_param_from_pretrained(monkeypatch)
    captured: list[tuple[str | None, str | None, str | None]] = []

    def fake_generate_video_impl(prompt=None, sampling_param=None, **kwargs):
        captured.append((prompt, sampling_param.image_path, sampling_param.video_path))
        return {"prompts": prompt, "video_path": "outputs/test.mp4"}

    monkeypatch.setattr(generator, "_generate_video_impl", fake_generate_video_impl)

    results = generator.generate(
        GenerationRequest(
            prompt=["first prompt", "second prompt"],
            inputs=InputConfig(
                image_path=["first.png", "second.png"],
                video_path=["first.mp4", "second.mp4"],
            ),
        ))

    assert [result.prompt for result in results] == ["first prompt", "second prompt"]
    assert captured == [
        ("first prompt", "first.png", "first.mp4"),
        ("second prompt", "second.png", "second.mp4"),
    ]


def test_generate_typed_prompt_list_routes_through_dynamic_batching(monkeypatch, tmp_path):
    generator = _new_video_generator()
    generator.fastvideo_args = _batching_fastvideo_args()
    _patch_sampling_param_from_pretrained(monkeypatch)
    calls = []

    def fake_device_memory(gpu_id):
        return 48.0

    def fake_run_forward(batch, fastvideo_args):
        calls.append(batch)
        batch_size = len(batch.prompt) if isinstance(batch.prompt, list) else 1
        output = torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4, 1, 1, 1)
        return ForwardBatch(data_type=batch.data_type, output=output), 0.5, 10.0

    monkeypatch.setattr(
        "fastvideo.batching.admission.BatchAdmissionController._get_device_memory_gb",
        staticmethod(fake_device_memory),
    )
    monkeypatch.setattr(generator, "_run_forward_batch", fake_run_forward)

    results = generator.generate(
        GenerationRequest(
            prompt=["first prompt", "second prompt"],
            sampling=SamplingConfig(num_frames=1, height=8, width=8),
            output=OutputConfig(
                output_path=str(tmp_path),
                save_video=False,
                return_frames=True,
            ),
        )
    )

    assert len(calls) == 1
    assert calls[0].prompt == ["first prompt", "second prompt"]
    assert [result.prompt for result in results] == ["first prompt", "second prompt"]
    assert [result.prompt_index for result in results] == [0, 1]
    assert all(isinstance(result, GenerationResult) for result in results)


def test_generate_batched_request_rejects_mismatched_media_inputs(monkeypatch):
    generator = _new_runtime_video_generator()
    _patch_sampling_param_from_pretrained(monkeypatch)

    with pytest.raises(ValueError, match="image_path"):
        generator.generate(
            GenerationRequest(
                prompt=["first prompt", "second prompt"],
                inputs=InputConfig(image_path=["first.png"]),
            ))
