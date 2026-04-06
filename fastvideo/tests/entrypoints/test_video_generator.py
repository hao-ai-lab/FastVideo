import os
from types import SimpleNamespace
import warnings

import pytest

from fastvideo.api import (
    GenerationRequest,
    GenerationResult,
    GeneratorConfig,
    InputConfig,
    SamplingConfig,
    load_run_config,
)
from fastvideo.configs.sample import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import WorkloadType


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
        )
    )

    assert isinstance(result, GenerationResult)
    assert captured["prompt"] == "hello world"
    assert captured["sampling_param"].num_frames == 81
    assert captured["sampling_param"].height == 480
    assert captured["sampling_param"].width == 832
    assert result.video_path == "outputs/test.mp4"


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
        )
    )

    assert captured["sampling_param"].negative_prompt is None
    assert captured["sampling_param"].num_frames == 125
    assert captured["sampling_param"].height == 720
    assert captured["sampling_param"].width == 1280


def test_generate_mapping_request_preserves_model_defaults_for_omitted_fields(
    monkeypatch,
):
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

    generator.generate(
        {
            "prompt": "hello world",
        }
    )

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
    assert captured["kwargs"]["num_frames"] == 49
    assert captured["kwargs"]["output_path"] == "outputs/legacy"
    assert captured["kwargs"]["save_video"] is False
    assert result["video_path"] == "outputs/test.mp4"


def test_generate_video_legacy_call_preserves_unknown_kwargs(monkeypatch):
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
    assert captured["kwargs"]["neg_prompt"] == "custom negative"
    assert captured["kwargs"]["embedded_cfg_scale"] == 7.5
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

    results = generator.generate(
        {
            "inputs": {"prompt_path": str(prompt_file)},
            "output": {
                "output_path": str(output_dir),
                "save_video": False,
                "return_frames": False,
            },
        }
    )

    assert isinstance(results, list)
    assert [result.prompt for result in results] == ["first prompt", "second prompt"]
    assert [result.prompt_index for result in results] == [0, 1]
    assert captured_prompts == ["first prompt", "second prompt"]


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
        )
    )

    assert [result.prompt for result in results] == ["first prompt", "second prompt"]
    assert captured == [
        ("first prompt", "first.png", "first.mp4"),
        ("second prompt", "second.png", "second.mp4"),
    ]


def test_generate_batched_request_rejects_mismatched_media_inputs(monkeypatch):
    generator = _new_runtime_video_generator()
    _patch_sampling_param_from_pretrained(monkeypatch)

    with pytest.raises(ValueError, match="image_path"):
        generator.generate(
            GenerationRequest(
                prompt=["first prompt", "second prompt"],
                inputs=InputConfig(image_path=["first.png"]),
            )
        )
