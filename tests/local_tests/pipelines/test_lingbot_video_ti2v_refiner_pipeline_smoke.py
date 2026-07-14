# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU production smoke for LingBot-Video TI2V plus refinement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from tests.local_tests.lingbot_video.hf_assets import FASTVIDEO_MOE, download_components


def test_lingbot_video_moe_ti2v_refiner_pipeline_smoke(tmp_path: Path) -> None:
    """Run one conditioned frame through image prompting, both VAEs, and both MoE DiTs."""
    if os.environ.get("LINGBOT_VIDEO_RUN_REFINER_PIPELINE_TESTS") != "1":
        pytest.skip("Set LINGBOT_VIDEO_RUN_REFINER_PIPELINE_TESTS=1 on a scheduled H200 node.")
    required_gpus = int(os.environ.get("LINGBOT_VIDEO_REFINER_NUM_GPUS", "8"))
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
        pytest.skip(f"LingBot-Video TI2V refiner smoke requires {required_gpus} CUDA devices.")
    local_checkpoint = os.environ.get("LINGBOT_VIDEO_TI2V_CHECKPOINT")
    model_dir = Path(local_checkpoint) if local_checkpoint else download_components(
        FASTVIDEO_MOE,
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "transformer_2",
        "vae",
    )
    image_path = tmp_path / "condition.png"
    Image.fromarray(np.full((64, 96, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        workload_type="i2v",
        num_gpus=required_gpus,
        sp_size=required_gpus,
        use_fsdp_inference=True,
        refine_enabled=True,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    try:
        result = generator.generate_video(
            prompt="A red fox runs through fresh snow at sunrise.",
            image_path=str(image_path),
            output_path=str(tmp_path),
            save_video=False,
            return_frames=True,
            height=32,
            width=32,
            height_sr=64,
            width_sr=64,
            num_frames=1,
            num_inference_steps=1,
            num_inference_steps_sr=1,
            guidance_scale=3.0,
            guidance_scale_2=3.0,
            t_thresh=0.85,
            batch_cfg=True,
            seed=42,
        )
    finally:
        generator.shutdown()
    samples = cast(dict[str, Any], result)["samples"]
    assert torch.is_tensor(samples)
    assert tuple(samples.shape) == (1, 3, 1, 64, 64)
    assert torch.isfinite(samples).all()
