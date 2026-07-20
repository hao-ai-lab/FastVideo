# SPDX-License-Identifier: Apache-2.0
"""Real-weight production-loader smoke gate for Cosmos3 T2V."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from PIL import Image

MODEL_DIR = Path(os.getenv("COSMOS3_WEIGHTS_DIR", ""))


def _require_t2v_gpu_assets() -> Path:
    if os.getenv("COSMOS3_RUN_GPU_TESTS") != "1":
        pytest.skip("set COSMOS3_RUN_GPU_TESTS=1 on an allocated GPU")
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 production-loader smoke requires CUDA")
    if not MODEL_DIR.is_dir():
        pytest.skip("set COSMOS3_WEIGHTS_DIR to the pinned Cosmos3-Nano snapshot")
    return MODEL_DIR


@pytest.mark.parametrize("workload", ["t2v", "i2v"])
def test_cosmos3_production_loader_video_latent(tmp_path: Path, workload: str) -> None:
    """Materialize the real components and run one reduced video denoising step."""
    model_dir = _require_t2v_gpu_assets()
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        workload_type=workload,
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
        output_type="latent",
    )
    try:
        height, width, num_frames = 256, 448, 25
        initial_noise = torch.zeros(1, 48, 7, height // 16, width // 16)
        conditioning = Image.new("RGB", (width, height), color=(64, 128, 192)) if workload == "i2v" else None
        result = generator.generate_video(
            prompt="A red panda walks through a bamboo forest.",
            pil_image=conditioning,
            output_path=str(tmp_path),
            save_video=False,
            return_frames=True,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=1,
            guidance_scale=6.0,
            fps=24,
            seed=0,
            latents=initial_noise,
        )
        samples = cast(dict[str, Any], result)["samples"]
        actual = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
        assert actual.shape == initial_noise.shape
        assert torch.isfinite(actual).all()
    finally:
        generator.shutdown()
