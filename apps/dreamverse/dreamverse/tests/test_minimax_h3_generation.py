from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import dreamverse.generation_worker as generation_worker
from dreamverse.minimax_h3_generation import MiniMaxH3GenerationBackend


FASTH3_MODEL_CONFIG = {
    "name": "FastH3",
    "generation_backend": "minimax_h3",
    "default_sp_size": 4,
    "model_path": "MiniMaxAI/MiniMax-H3",
    "adapter_repo": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
    "adapter_filename": "vsa-datafree/adapter_model.safetensors",
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "height": 768,
    "width": 1344,
    "num_frames": 124,
    "num_inference_steps": 5,
    "seed": 1000,
}


class _RecordingGenerator:
    """Record typed requests and return small synchronized media fixtures."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.conditioning_pixels: list[np.ndarray | None] = []

    def generate(self, request):
        """Capture the request and return two tiny video frames with audio."""
        self.requests.append(request)
        conditioning_image = request.inputs.pil_image
        self.conditioning_pixels.append(
            None if conditioning_image is None else np.asarray(conditioning_image).copy())
        frames = [
            np.full((2, 3, 3), 10, dtype=np.uint8),
            np.full((2, 3, 3), 20, dtype=np.uint8),
        ]
        return SimpleNamespace(
            frames=frames,
            audio=np.zeros((2, 16), dtype=np.float32),
            audio_sample_rate=44100,
            generation_time=0.25,
        )


def test_initialize_builds_vsa_datafree_fasth3_generator(monkeypatch):
    """Initialization translates the DreamVerse profile into typed FastVideo config."""
    from fastvideo import VideoGenerator

    captured = {}
    fake_generator = SimpleNamespace(shutdown=lambda: None)

    def fake_from_config(config):
        captured["config"] = config
        return fake_generator

    def fake_download(**kwargs):
        captured["download"] = kwargs
        return f"/models/{kwargs['filename']}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(VideoGenerator, "from_config", fake_from_config)
    monkeypatch.setattr("dreamverse.minimax_h3_generation.DREAMVERSE_SP_SIZE", 4)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "test-attention")
    monkeypatch.setenv("FASTVIDEO_FA4", "0")
    monkeypatch.setenv("FASTVIDEO_MINIMAX_H3_FUSIONS", "0")
    monkeypatch.setenv("FASTVIDEO_VSA_SM100A", "1")
    monkeypatch.setenv("FASTVIDEO_INFERENCE_TORCH_COMPILE", "1")

    backend = MiniMaxH3GenerationBackend(gpu_id=0)
    monkeypatch.setattr(backend, "_gpu_mem", lambda: "alloc=0.00GiB, reserved=0.00GiB")
    backend.initialize(FASTH3_MODEL_CONFIG)

    config = captured["config"]
    assert captured["download"] == {
        "repo_id": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
        "filename": "vsa-datafree/adapter_model.safetensors",
    }
    assert config.model_path == "MiniMaxAI/MiniMax-H3"
    assert config.pipeline.components.lora_path.endswith("vsa-datafree/adapter_model.safetensors")
    assert config.pipeline.components.lora_strength == 1.0
    assert config.pipeline.experimental == {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "inference_torch_compile": False,
        "vae_parallel_decode": True,
        "vae_parallel_decode_strategy": "gather",
        "VSA_sparsity": 0.9,
        "VSA_tile_size": 64,
    }
    assert config.engine.num_gpus == 4
    assert config.engine.parallelism.tp_size == 1
    assert config.engine.parallelism.sp_size == 4
    assert config.engine.offload.dit is False
    assert config.engine.offload.dit_layerwise is False
    assert config.engine.offload.text_encoder is True
    assert config.engine.offload.vae is True
    assert config.engine.compile.vae_enabled is True
    assert config.engine.use_fsdp_inference is False
    assert os.environ["FASTVIDEO_ATTENTION_BACKEND"] == "VIDEO_SPARSE_ATTN_H3"
    assert os.environ["FASTVIDEO_FA4"] == "1"
    assert os.environ["FASTVIDEO_MINIMAX_H3_FUSIONS"] == "all"
    assert os.environ["FASTVIDEO_VSA_SM100A"] == "0"
    assert "FASTVIDEO_INFERENCE_TORCH_COMPILE" not in os.environ


def test_initialize_selects_declared_generation_backend(monkeypatch):
    """The GPU worker constructs the backend that the active model profile declares."""
    from unittest.mock import Mock

    selected_backend = Mock()
    monkeypatch.setattr(
        generation_worker,
        "_create_generation_backend",
        lambda backend_name, gpu_id: selected_backend,
    )
    worker = generation_worker.VideoGenerationWorker(gpu_id=3)

    worker.initialize(FASTH3_MODEL_CONFIG)

    assert worker.backend_name == "minimax_h3"
    assert worker.backend is selected_backend
    selected_backend.initialize.assert_called_once_with(FASTH3_MODEL_CONFIG)


def test_initialize_failure_clears_backend_ownership(monkeypatch):
    """A failed family change leaves the GPU worker explicitly uninitialized."""
    ltx_backend = SimpleNamespace(initialize=lambda config: None, shutdown=lambda: None)

    def fail_initialize(config):
        del config
        raise RuntimeError("load failed")

    fasth3_backend = SimpleNamespace(
        initialize=fail_initialize,
        shutdown=lambda: None,
    )
    backends = {
        "ltx2": ltx_backend,
        "minimax_h3": fasth3_backend,
    }
    monkeypatch.setattr(
        generation_worker,
        "_create_generation_backend",
        lambda backend_name, gpu_id: backends[backend_name],
    )
    worker = generation_worker.VideoGenerationWorker(gpu_id=3)
    worker.initialize({"generation_backend": "ltx2"})

    with pytest.raises(RuntimeError, match="load failed"):
        worker.initialize(FASTH3_MODEL_CONFIG)

    assert worker.backend is None
    assert worker.backend_name is None
    assert worker.model_config == {"generation_backend": "ltx2"}


def test_generate_step_uses_last_frame_for_continuation(monkeypatch):
    """A later segment receives the prior segment's last decoded frame."""
    backend = MiniMaxH3GenerationBackend(gpu_id=0)
    backend.model_config = dict(FASTH3_MODEL_CONFIG)
    backend.generator = _RecordingGenerator()
    monkeypatch.setattr("dreamverse.minimax_h3_generation.torch.cuda.synchronize", lambda: None)

    first_result = backend.generate_step(
        "first prompt",
        segment_idx=1,
        image_path=None,
        reset_conditioning=True,
    )
    second_result = backend.generate_step(
        "second prompt",
        segment_idx=2,
        image_path=None,
        reset_conditioning=False,
    )

    first_request = backend.generator.requests[0]
    assert first_request.inputs.pil_image is None
    assert first_request.negative_prompt == ""
    assert first_request.sampling.height == 768
    assert first_request.sampling.width == 1344
    assert first_request.sampling.num_frames == 124
    assert first_request.sampling.num_inference_steps == 5
    assert first_request.sampling.fps == 24
    assert first_request.sampling.guidance_scale == 1.0
    assert first_request.sampling.batch_cfg is False
    assert first_request.sampling.seed == 1000
    assert first_request.output.save_video is False
    assert first_request.output.return_frames is True
    assert backend.generator.conditioning_pixels[1].tolist() == np.full((2, 3, 3), 20).tolist()
    assert first_result.head_trim_frames == 0
    assert first_result.head_trim_audio_frames == 0
    assert second_result.head_trim_frames == 1
    assert second_result.head_trim_audio_frames == 1
    assert second_result.audio_sample_rate == 44100


def test_generate_step_reset_uses_text_to_video_path(monkeypatch):
    """Resetting continuation produces an unconditioned text-to-video request."""
    backend = MiniMaxH3GenerationBackend(gpu_id=0)
    backend.model_config = dict(FASTH3_MODEL_CONFIG)
    backend.generator = _RecordingGenerator()
    monkeypatch.setattr("dreamverse.minimax_h3_generation.torch.cuda.synchronize", lambda: None)

    backend.generate_step("first prompt", 1, None, True)
    reset_result = backend.generate_step("reset prompt", 2, None, True)

    assert backend.generator.requests[-1].inputs.pil_image is None
    assert reset_result.head_trim_frames == 0
    assert reset_result.head_trim_audio_frames == 0


def test_generate_step_missing_continuation_frame(monkeypatch):
    """A later segment fails when no reset or retained frame defines its input."""
    backend = MiniMaxH3GenerationBackend(gpu_id=0)
    backend.model_config = dict(FASTH3_MODEL_CONFIG)
    backend.generator = _RecordingGenerator()

    with pytest.raises(RuntimeError, match="requires a retained continuation frame"):
        backend.generate_step("later prompt", 2, None, False)

    assert backend.generator.requests == []


def test_warmup_exercises_text_and_first_frame_paths(monkeypatch):
    """Warmup covers both request shapes used by a DreamVerse session."""
    backend = MiniMaxH3GenerationBackend(gpu_id=0)
    backend.model_config = dict(FASTH3_MODEL_CONFIG)
    backend.generator = _RecordingGenerator()
    monkeypatch.setattr("dreamverse.minimax_h3_generation.torch.cuda.synchronize", lambda: None)

    timings = backend.warmup("warmup prompt")

    assert backend.generator.conditioning_pixels[0] is None
    assert backend.generator.conditioning_pixels[1] is not None
    assert backend.continuation_image is None
    assert "warmup_text_to_video_ms" in timings
    assert "warmup_first_frame_to_video_ms" in timings
