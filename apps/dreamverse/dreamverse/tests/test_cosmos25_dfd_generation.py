from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreamverse.cosmos25_dfd_generation import Cosmos25DFDGenerationBackend

COSMOS_CONFIG = {
    "name": "Cosmos Predict2.5 DFD",
    "generation_backend": "cosmos25_dfd",
    "default_sp_size": 1,
    "model_path": "/models/cosmos25-t2w",
    "continuation_model_path": "/models/cosmos25-dfd",
    "attention_backend": "TORCH_SDPA",
    "height": 704,
    "width": 1280,
    "bootstrap_num_frames": 77,
    "continuation_num_frames": 81,
    "fps": 24,
    "num_inference_steps": 4,
    "seed": 42,
}


class _RecordingGenerator:
    def __init__(self, pixel_value: int = 20) -> None:
        self.pixel_value = pixel_value
        self.calls: list[dict] = []
        self.shutdown_calls = 0

    def generate_video(self, prompt, sampling_param):
        condition = sampling_param.pil_image
        self.calls.append({
            "prompt": prompt,
            "sampling": sampling_param,
            "conditioning_pixels": None if condition is None else np.asarray(condition).copy(),
        })
        frames = [
            np.full((2, 3, 3), self.pixel_value, dtype=np.uint8)
            for _ in range(sampling_param.num_frames)
        ]
        frames[-1] = np.full((2, 3, 3), self.pixel_value + 1, dtype=np.uint8)
        return {
            "frames": frames,
            "generation_time": 0.25,
        }

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def backend(monkeypatch) -> Cosmos25DFDGenerationBackend:
    instance = Cosmos25DFDGenerationBackend(gpu_id=0)
    instance.model_config = dict(COSMOS_CONFIG)
    instance.bootstrap_generator = _RecordingGenerator(pixel_value=20)
    instance.continuation_generator = _RecordingGenerator(pixel_value=40)
    monkeypatch.setattr("dreamverse.cosmos25_dfd_generation.torch.cuda.synchronize", lambda: None)

    def fake_sampling_param(*, conditioned):
        return SimpleNamespace(
            negative_prompt="",
            save_video=False,
            return_frames=True,
            height=704,
            width=1280,
            num_frames=81 if conditioned else 77,
            fps=24,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
            num_cond_frames=1 if conditioned else 0,
            pil_image=None,
        )

    monkeypatch.setattr(instance, "_sampling_param", fake_sampling_param)
    return instance


def test_initialize_loads_both_package_roles(monkeypatch):
    loaded_paths = []
    generators = [_RecordingGenerator(), _RecordingGenerator()]
    backend = Cosmos25DFDGenerationBackend(gpu_id=0)

    def fake_load(model_path):
        loaded_paths.append(model_path)
        return generators[len(loaded_paths) - 1]

    monkeypatch.setattr(backend, "_load_generator", fake_load)
    monkeypatch.setattr(backend, "_gpu_mem", lambda: "alloc=0.00GiB, reserved=0.00GiB")
    monkeypatch.setattr("dreamverse.cosmos25_dfd_generation.gc.collect", lambda: 0)
    monkeypatch.setattr("dreamverse.cosmos25_dfd_generation.torch.cuda.is_available", lambda: False)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "test-attention")
    monkeypatch.setenv("FASTVIDEO_INFERENCE_TORCH_COMPILE", "1")

    backend.initialize(COSMOS_CONFIG)

    assert loaded_paths == [
        "/models/cosmos25-t2w",
        "/models/cosmos25-dfd",
    ]
    assert backend.bootstrap_generator is generators[0]
    assert backend.continuation_generator is generators[1]
    assert backend.model_config == COSMOS_CONFIG
    assert os.environ["FASTVIDEO_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert "FASTVIDEO_INFERENCE_TORCH_COMPILE" not in os.environ


def test_unconditioned_start_uses_t2w_and_retains_terminal_frame(backend):
    result = backend.generate_step("first prompt", 1, None, True)

    assert len(backend.bootstrap_generator.calls) == 1
    assert backend.continuation_generator.calls == []
    sampling = backend.bootstrap_generator.calls[0]["sampling"]
    assert sampling.height == 704
    assert sampling.width == 1280
    assert sampling.num_frames == 77
    assert sampling.fps == 24
    assert sampling.num_inference_steps == 4
    assert sampling.guidance_scale == 1.0
    assert sampling.seed == 42
    assert sampling.num_cond_frames == 0
    assert sampling.pil_image is None
    assert result.head_trim_frames == 0
    assert result.head_trim_audio_frames == 0
    assert result.audio_sample_rate == 24_000
    assert result.audio.shape == (77_000, )
    assert result.audio.count_nonzero() == 0
    assert np.asarray(backend.continuation_image).tolist() == np.full((2, 3, 3), 21).tolist()


def test_retained_frame_uses_dfd_and_trims_repeated_boundary(backend):
    backend.generate_step("first prompt", 1, None, True)
    result = backend.generate_step("pivot right", 2, None, False)

    assert len(backend.continuation_generator.calls) == 1
    call = backend.continuation_generator.calls[0]
    sampling = call["sampling"]
    assert sampling.num_frames == 81
    assert sampling.num_cond_frames == 1
    assert call["conditioning_pixels"].tolist() == np.full((2, 3, 3), 21).tolist()
    assert result.head_trim_frames == 1
    assert result.head_trim_audio_frames == 1
    assert result.audio.shape == (81_000, )
    assert np.asarray(backend.continuation_image).tolist() == np.full((2, 3, 3), 41).tolist()


def test_initial_image_uses_dfd_without_stream_trim(backend, tmp_path: Path):
    from PIL import Image

    image_path = tmp_path / "initial.png"
    Image.fromarray(np.full((2, 3, 3), 7, dtype=np.uint8)).save(image_path)

    result = backend.generate_step("animate", 1, str(image_path), True)

    assert backend.bootstrap_generator.calls == []
    call = backend.continuation_generator.calls[0]
    assert call["conditioning_pixels"].tolist() == np.full((2, 3, 3), 7).tolist()
    assert result.head_trim_frames == 0
    assert result.head_trim_audio_frames == 0


def test_missing_later_continuation_fails_before_generation(backend):
    with pytest.raises(RuntimeError, match="requires a retained continuation frame"):
        backend.generate_step("later prompt", 2, None, False)

    assert backend.bootstrap_generator.calls == []
    assert backend.continuation_generator.calls == []


def test_reset_later_segment_uses_fresh_t2w_bootstrap(backend):
    backend.generate_step("first prompt", 1, None, True)

    result = backend.generate_step("new scene", 2, None, True)

    assert len(backend.bootstrap_generator.calls) == 2
    assert backend.continuation_generator.calls == []
    assert result.head_trim_frames == 0


def test_warmup_exercises_bootstrap_and_dfd_paths(backend):
    timings = backend.warmup("warmup prompt")

    assert len(backend.bootstrap_generator.calls) == 1
    assert len(backend.continuation_generator.calls) == 1
    assert backend.continuation_image is None
    assert "warmup_bootstrap_ms" in timings
    assert "warmup_continuation_ms" in timings
    assert "warmup_total_ms" in timings


def test_shutdown_releases_both_generators_and_conditioning(backend):
    bootstrap = backend.bootstrap_generator
    continuation = backend.continuation_generator
    backend.generate_step("first prompt", 1, None, True)

    backend.shutdown()

    assert bootstrap.shutdown_calls == 1
    assert continuation.shutdown_calls == 1
    assert backend.bootstrap_generator is None
    assert backend.continuation_generator is None
    assert backend.continuation_image is None
