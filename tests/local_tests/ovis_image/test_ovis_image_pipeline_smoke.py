# SPDX-License-Identifier: Apache-2.0
"""Smoke test: a full ``generate_video()`` through VideoGenerator yields a
correctly shaped, finite output and writes a file. Numerical parity lives in the
component and pipeline parity tests."""

import os
import tempfile
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29521")

OVIS_WEIGHTS = os.getenv("OVIS_WEIGHTS", "AIDC-AI/Ovis-Image-7B")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Ovis-Image pipeline smoke test requires CUDA.")
def test_ovis_image_pipeline_smoke():
    """Smoke test: load Ovis-Image-7B and run a single forward pass."""
    from fastvideo import VideoGenerator

    # Use a small resolution and few steps so the test runs quickly.
    prompt = (
        'A vibrant poster with the text "FAST VIDEO" written in bold red '
        "letters on a clean white background. High contrast, 4k quality.")
    height = 128
    width = 128
    num_frames = 1
    num_inference_steps = 4
    seed = 42

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = VideoGenerator.from_pretrained(
            OVIS_WEIGHTS,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            pin_cpu_memory=False,
        )
        try:
            result = generator.generate_video(
                prompt,
                output_path=tmpdir,
                save_video=True,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=5.0,
                seed=seed,
                fps=1,
            )
        finally:
            generator.shutdown()

        # --- shape check ---
        samples = result["samples"]
        # Expected: (B, C, T, H, W) = (1, 3, 1, 128, 128)
        assert samples.ndim == 5, f"Expected 5-D tensor, got shape {samples.shape}"
        assert samples.shape[0] == 1
        assert samples.shape[2] == num_frames
        assert samples.shape[3] == height
        assert samples.shape[4] == width

        # --- finite check ---
        assert torch.isfinite(samples).all(), "Output contains NaN or Inf values"

        # --- output file check (inside the tmpdir context, before cleanup) ---
        output_files = list(Path(tmpdir).glob("*.mp4")) + list(
            Path(tmpdir).glob("*.png"))
        assert len(output_files) > 0, (
            f"No output file was saved under {tmpdir}. Files: {os.listdir(tmpdir)}")
