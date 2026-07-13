# SPDX-License-Identifier: Apache-2.0
"""Production-loader smoke coverage for the base-only LingBot-Video MoE pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tests.local_tests.lingbot_video.hf_assets import FASTVIDEO_MOE, materialize_component_view


def test_lingbot_video_moe_base_pipeline_smoke(tmp_path: Path) -> None:
    """Load one 30B MoE DiT and run one sequential-CFG latent step on an H200."""
    if os.environ.get("LINGBOT_VIDEO_RUN_MOE_PIPELINE_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_MOE_PIPELINE_TESTS=1 on a scheduled H200.")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video MoE pipeline smoke requires CUDA.")
    from fastvideo import VideoGenerator

    model_dir = materialize_component_view(
        FASTVIDEO_MOE,
        tmp_path / "base_model",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    )
    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=1,
        sp_size=1,
        use_fsdp_inference=False,
        refine_enabled=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
    )
    try:
        result = generator.generate_video(
            prompt="A red fox runs through fresh snow at sunrise.",
            output_path=str(tmp_path),
            save_video=False,
            return_frames=True,
            height=32,
            width=32,
            num_frames=1,
            num_inference_steps=1,
            guidance_scale=3.0,
            batch_cfg=False,
            seed=42,
        )
    finally:
        generator.shutdown()
    samples = cast(dict[str, Any], result)["samples"]
    assert torch.is_tensor(samples)
    assert tuple(samples.shape) == (1, 16, 1, 4, 4)
    assert torch.isfinite(samples).all()
