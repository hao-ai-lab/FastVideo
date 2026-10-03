# SPDX-License-Identifier: Apache-2.0
"""Production-loader smoke coverage for the base-only LingBot-Video MoE pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
import torch

from tests.local_tests.lingbot_video.hf_assets import FASTVIDEO_MOE, materialize_component_view


def test_lingbot_video_moe_base_pipeline_smoke(tmp_path: Path) -> None:
    """Run two batched-CFG steps over two temporal latent frames with the 30B MoE."""
    if os.environ.get("LINGBOT_VIDEO_RUN_MOE_PIPELINE_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_MOE_PIPELINE_TESTS=1 on a scheduled H200.")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video MoE pipeline smoke requires CUDA.")
    from fastvideo import VideoGenerator
    from fastvideo.api import GenerationResult

    model_dir = materialize_component_view(
        FASTVIDEO_MOE,
        tmp_path / "base_model",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    )
    generator = VideoGenerator.from_config({
        "model_path": str(model_dir),
        "engine": {
            "num_gpus": 1,
            "parallelism": {"sp_size": 1},
            "use_fsdp_inference": False,
            "offload": {
                "dit": False,
                "dit_layerwise": False,
                "vae": True,
                "text_encoder": True,
                "pin_cpu_memory": False,
            },
        },
        "pipeline": {"experimental": {"refine_enabled": False, "output_type": "latent"}},
    })
    try:
        result = generator.generate({
            "prompt": "A red fox runs through fresh snow at sunrise.",
            "sampling": {
                "height": 32,
                "width": 32,
                "num_frames": 5,
                "num_inference_steps": 2,
                "guidance_scale": 3.0,
                "batch_cfg": True,
                "seed": 42,
            },
            "output": {"output_path": str(tmp_path), "save_video": False, "return_frames": True},
        })
    finally:
        generator.shutdown()
    samples = cast(GenerationResult, result).samples
    assert torch.is_tensor(samples)
    assert tuple(samples.shape) == (1, 16, 2, 4, 4)
    assert torch.isfinite(samples).all()
