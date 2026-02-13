# SPDX-License-Identifier: Apache-2.0
"""
Smoke test for Stable Audio pipeline.

Loads the pipeline, runs one generate_audio call, and asserts:
- Output audio shape, dtype, sample_rate match config
- No exceptions
"""
import pytest
import torch

from fastvideo import VideoGenerator


def _model_path() -> str:
    return "stabilityai/stable-audio-open-1.0"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stable Audio smoke test requires CUDA.",
)
def test_stable_audio_pipeline_smoke() -> None:
    model_path = _model_path()
    sample_rate = 44100
    duration_seconds = 5.0
    expected_sample_size = int(duration_seconds * sample_rate)
    expected_channels = 2
    downsampling_ratio = 2048

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
    )

    result = generator.generate_audio(
        prompt="A short piano note.",
        duration_seconds=duration_seconds,
        num_inference_steps=10,
        seed=42,
    )

    generator.shutdown()

    audio = result["audio"]
    assert isinstance(audio, torch.Tensor), "Expected audio to be a tensor"
    assert audio.ndim == 3, f"Expected (B, C, T), got {audio.shape}"
    b, c, t = audio.shape
    assert b >= 1, "Expected at least one batch"
    assert c == expected_channels, f"Expected stereo ({expected_channels}), got {c}"
    assert abs(t - expected_sample_size) <= downsampling_ratio, (
        f"Expected ~{expected_sample_size} samples ({duration_seconds}s @ {sample_rate}Hz), got {t}"
    )

    assert result["sample_rate"] == sample_rate, (
        f"Expected sample_rate {sample_rate}, got {result['sample_rate']}"
    )

    assert audio.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"Expected float dtype, got {audio.dtype}"
    )

    assert "generation_time" in result
    assert result["prompt"] == "A short piano note."
