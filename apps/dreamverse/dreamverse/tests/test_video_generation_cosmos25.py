from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamverse.video_generation import VideoGenerationWorker


def _cosmos_config() -> dict:
    return {
        "name": "Cosmos Predict2.5 Distilled",
        "family": "cosmos25_distilled",
        "model_path": "/models/cosmos25-distilled",
        "supports_audio": False,
        "supports_continuation": False,
        "supports_lora": False,
        "height": 704,
        "width": 1280,
        "num_frames": 77,
        "fps": 16,
        "num_inference_steps": 4,
        "seed": 42,
    }


class _RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_video(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "frames": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(9)],
            "generation_time": 0.25,
        }


def _worker(monkeypatch: pytest.MonkeyPatch) -> tuple[VideoGenerationWorker, _RecordingGenerator]:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    worker = VideoGenerationWorker(gpu_id=0)
    generator = _RecordingGenerator()
    worker.generator = generator
    worker.current_model_config = _cosmos_config()
    return worker, generator


def test_cosmos25_step_uses_distilled_profile_and_synthesizes_silence(monkeypatch):
    worker, generator = _worker(monkeypatch)

    result = worker.generate_step(
        "A robot crossing a desert",
        segment_idx=2,
        image_path=None,
        reset_conditioning=False,
    )

    assert len(generator.calls) == 1
    request = generator.calls[0]
    assert request["height"] == 704
    assert request["width"] == 1280
    assert request["num_frames"] == 77
    assert request["fps"] == 16
    assert request["num_inference_steps"] == 4
    assert request["seed"] == 42
    assert request["return_frames"] is True
    assert request["save_video"] is False
    assert "ltx2_image_crf" not in request
    assert "conditioning_images" not in request
    assert "audio_latents" not in request

    assert result.fps == 16
    assert result.audio_sample_rate == 24000
    assert isinstance(result.audio, torch.Tensor)
    assert result.audio.dtype is torch.float32
    assert result.audio.shape == (13500,)
    assert torch.count_nonzero(result.audio) == 0
    assert result.head_trim_frames == 0
    assert result.head_trim_audio_frames == 0
    assert worker.continuation.video_images is None
    assert worker.continuation.audio_latents is None


def test_cosmos25_step_rejects_initial_image(monkeypatch):
    worker, generator = _worker(monkeypatch)

    with pytest.raises(RuntimeError, match="text-to-world only"):
        worker.generate_step(
            "Animate this image",
            segment_idx=1,
            image_path="input.png",
            reset_conditioning=True,
        )

    assert generator.calls == []


def test_cosmos25_warmup_generates_one_independent_segment(monkeypatch):
    worker, _generator = _worker(monkeypatch)
    calls: list[tuple[int, bool]] = []

    def fake_generate_step(prompt, segment_idx, image_path, reset_conditioning):
        del prompt, image_path
        calls.append((segment_idx, reset_conditioning))
        return SimpleNamespace(timings={"e2e_latency_ms": 123.0})

    monkeypatch.setattr(worker, "generate_step", fake_generate_step)

    timings = worker.warmup("A warmup prompt")

    assert calls == [(1, True)]
    assert timings["warmup_segment1_ms"] == 123.0
    assert timings["warmup_total_ms"] >= 0.0


def test_ltx_profile_keeps_existing_generation_defaults(monkeypatch):
    worker, generator = _worker(monkeypatch)
    worker.current_model_config = {
        "name": "FastLTX2",
        "family": "ltx2",
        "model_path": "/models/ltx2",
        "supports_audio": True,
        "supports_continuation": True,
        "supports_lora": True,
    }

    result = worker.generate_step(
        "A reference image comes alive",
        segment_idx=1,
        image_path="input.png",
        reset_conditioning=True,
    )

    request = generator.calls[0]
    assert request["height"] == 1088
    assert request["width"] == 1920
    assert request["num_frames"] == 121
    assert request["fps"] == 24
    assert request["num_inference_steps"] == 5
    assert request["seed"] == 10
    assert request["ltx2_image_crf"] == 0.0
    assert request["image_path"] == "input.png"
    assert result.audio is None
    assert result.audio_sample_rate is None
    assert result.fps == 24
