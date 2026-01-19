from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.configs.sample import SamplingParam
from fastvideo.entrypoints.generation_types import GenerationResult, OutputOptions
from fastvideo.entrypoints.video_generator import VideoGenerator


class _DummyExecutor:

    def __init__(self, samples: torch.Tensor):
        self._samples = samples
        self.calls = 0

    def execute_forward(self, batch, fastvideo_args):
        self.calls += 1
        return SimpleNamespace(
            output=self._samples,
            logging_info=SimpleNamespace(),
            trajectory_latents="trajectory_latents",
            trajectory_timesteps="trajectory_timesteps",
            trajectory_decoded="trajectory_decoded",
        )


def _new_stub_generator(samples: torch.Tensor) -> VideoGenerator:
    generator = VideoGenerator.__new__(VideoGenerator)
    generator.fastvideo_args = SimpleNamespace(
        model_path="dummy",
        num_gpus=1,
        prompt_txt=None,
        VSA_sparsity=0.0,
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(temporal_compression_ratio=4),
                use_temporal_scaling_frames=True,
            ),
            flow_shift=0.0,
            embedded_cfg_scale=0.0,
        ),
    )
    generator.executor = _DummyExecutor(samples)
    return generator


def _base_params(tmp_path) -> SamplingParam:
    return SamplingParam(
        num_frames=2,
        height=64,
        width=64,
        fps=8,
        num_inference_steps=1,
        seed=0,
        save_video=False,
        output_path=str(tmp_path / "outputs"),
    )


def test_generate_single_minimal_dict_does_not_build_frames(tmp_path,
                                                           monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    def _fail_make_grid(*args, **kwargs):
        raise AssertionError("make_grid should not be called")

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fail_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
    )

    assert isinstance(result, dict)
    assert "output_path" in result
    assert "frames" not in result
    assert "samples" not in result


def test_generate_single_dataclass_minimal_does_not_build_frames(
        tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    def _fail_make_grid(*args, **kwargs):
        raise AssertionError("make_grid should not be called")

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fail_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
        return_format="dataclass",
    )

    assert isinstance(result, GenerationResult)
    assert result.frames is None
    assert result.samples is None


def test_generate_single_return_frames_in_dict(tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    called = {"count": 0}

    def _fake_make_grid(x, nrow=6):
        called["count"] += 1
        return x[0]

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fake_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
        return_frames_in_dict=True,
    )

    assert called["count"] > 0
    assert isinstance(result, dict)
    assert isinstance(result["frames"], list)
    assert isinstance(result["frames"][0], np.ndarray)
    assert "samples" not in result


def test_generate_single_dataclass_with_output_options(tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    called = {"count": 0}

    def _fake_make_grid(x, nrow=6):
        called["count"] += 1
        return x[0]

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fake_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
        output_options=OutputOptions(
            include_frames=True,
            include_samples=True,
        ),
        return_format="dataclass",
    )

    assert called["count"] > 0
    assert isinstance(result, GenerationResult)
    assert isinstance(result.frames, list)
    assert isinstance(result.frames[0], np.ndarray)
    assert result.samples is samples


def test_generate_single_return_samples(tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    def _fail_make_grid(*args, **kwargs):
        raise AssertionError("make_grid should not be called")

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fail_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
        return_samples=True,
    )

    assert isinstance(result, dict)
    assert result["samples"] is samples
    assert "frames" not in result


def test_generate_single_return_frames_list(tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    def _fake_make_grid(x, nrow=6):
        return x[0]

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fake_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
        return_frames=True,
    )

    assert isinstance(result, list)
    assert isinstance(result[0], np.ndarray)


def test_generate_single_trajectory_flags(tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    def _fail_make_grid(*args, **kwargs):
        raise AssertionError("make_grid should not be called")

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fail_make_grid,
    )

    result = generator.generate_video(
        prompt="hello",
        sampling_param=_base_params(tmp_path),
        return_trajectory_latents=True,
        return_trajectory_decoded=True,
    )

    assert result["trajectory"] == "trajectory_latents"
    assert result["trajectory_timesteps"] == "trajectory_timesteps"
    assert result["trajectory_decoded"] == "trajectory_decoded"


def test_generate_batch_prompt_path_minimal_outputs(tmp_path, monkeypatch):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    def _fail_make_grid(*args, **kwargs):
        raise AssertionError("make_grid should not be called")

    monkeypatch.setattr(
        "fastvideo.entrypoints.video_generator.torchvision.utils.make_grid",
        _fail_make_grid,
    )

    prompt_path = tmp_path / "prompts.txt"
    prompt_path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")

    results = generator.generate_video(
        sampling_param=_base_params(tmp_path),
        prompt_path=str(prompt_path),
        monitor=object(),
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for idx, result in enumerate(results):
        assert isinstance(result, dict)
        assert result["prompt_index"] == idx
        assert result["prompt"] in {"first prompt", "second prompt"}
        assert "frames" not in result
        assert "samples" not in result


def test_generate_prompt_path_requires_txt_extension(tmp_path):
    samples = torch.rand(1, 3, 2, 4, 4)
    generator = _new_stub_generator(samples)

    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text("hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prompt_path must be a txt file"):
        generator.generate_video(
            sampling_param=_base_params(tmp_path),
            prompt_path=str(prompt_path),
        )
