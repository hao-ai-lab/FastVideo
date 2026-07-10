# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the DreamX-World-5B-Cam pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from PIL import Image, ImageDraw


os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

MODEL_DIR = Path(os.getenv("DREAMX_WORLD_MODEL_DIR", "converted_weights/dreamx_world"))




def _write_smoke_image(path: Path) -> None:
    image = Image.new("RGB", (96, 96), color=(42, 76, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 56, 96, 96), fill=(36, 44, 52))
    draw.polygon([(0, 56), (48, 30), (96, 56)], fill=(120, 142, 158))
    draw.rectangle((34, 42, 62, 72), fill=(178, 198, 212))
    image.save(path)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DreamX-World pipeline smoke requires CUDA",
)


def test_dreamx_world_typed_surface_preflight() -> None:
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.pipelines.dreamx_world import DreamXWorld5BCamPipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.dreamx_world.dreamx_world_pipeline import (
        DreamXWorldPipeline,
        EntryClass,
    )

    assert DreamXWorldPipeline.__name__ == "DreamXWorldPipeline"
    assert EntryClass is DreamXWorldPipeline
    assert DreamXWorldPipeline._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    default_preset, model_family = registry.get_preset_selection(
        "GD-ML/DreamX-World-5B-Cam"
    )
    assert model_family == "dreamx_world"
    assert default_preset == "dreamx_world_5b_cam"

    info = registry.get_model_info(
        "GD-ML/DreamX-World-5B-Cam",
        workload_type=WorkloadType.I2V,
        override_pipeline_cls_name="DreamXWorldPipeline",
    )
    assert info.pipeline_cls is DreamXWorldPipeline
    assert info.pipeline_config_cls is DreamXWorld5BCamPipelineConfig

    names = {p.name for p in get_presets_for_family("dreamx_world")}
    assert "dreamx_world_5b_cam" in names
    preset = get_preset("dreamx_world_5b_cam", "dreamx_world")
    assert preset.defaults["num_inference_steps"] == 30
    assert preset.defaults["height"] == 480
    assert preset.defaults["width"] == 832
    assert preset.defaults["num_frames"] == 161
    assert preset.defaults["guidance_scale"] == 5.0

    cfg = DreamXWorld5BCamPipelineConfig()
    assert cfg.flow_shift == 3.0
    assert cfg.ti2v_task is True
    assert cfg.expand_timesteps is True
    assert cfg.dit_config.arch_config.add_control_adapter is True
    assert cfg.dit_config.arch_config.cam_method == "prope"


def test_dreamx_world_camera_stage_writes_y_camera() -> None:
    from types import SimpleNamespace

    from fastvideo.pipelines.basic.dreamx_world.stages import (
        DREAMX_Y_CAMERA_KEY,
        DreamXWorldCameraConditioningStage,
    )
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    batch = ForwardBatch(
        data_type="video",
        prompt="camera smoke",
        latents=torch.zeros(1, 48, 3, 8, 8, dtype=torch.bfloat16, device="cuda"),
        num_frames=9,
        height=64,
        width=64,
        action_list=["w", "d"],
        action_speed_list=[2.0, 1.0],
    )
    out = DreamXWorldCameraConditioningStage().forward(batch, cast(Any, SimpleNamespace()))

    y_camera = out.extra[DREAMX_Y_CAMERA_KEY]
    assert set(y_camera) == {"viewmats", "K"}
    assert y_camera["viewmats"].shape == (1, 3, 4, 4)
    assert y_camera["K"].shape == (1, 3, 3, 3)
    assert y_camera["viewmats"].device.type == "cuda"
    assert y_camera["viewmats"].dtype == torch.bfloat16


def test_dreamx_world_pipeline_load_generate_latent_smoke(tmp_path: Path) -> None:
    if not MODEL_DIR.exists():
        pytest.fail(f"DreamX-World converted model directory is missing: {MODEL_DIR}")

    from fastvideo import VideoGenerator

    image_path = tmp_path / "dreamx_world_smoke_input.png"
    _write_smoke_image(image_path)

    generator = VideoGenerator.from_pretrained(
        str(MODEL_DIR),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="DreamXWorldPipeline",
    )
    try:
        result = generator.generate_video(
            prompt="a quiet road through a futuristic city at sunrise",
            output_path="outputs_video/dreamx_world_smoke",
            save_video=False,
            return_frames=True,
            height=64,
            width=64,
            num_frames=9,
            num_inference_steps=1,
            guidance_scale=1.0,
            image_path=str(image_path),
            action_list=["w"],
            action_speed_list=[2.0],
            seed=0,
        )
    finally:
        generator.shutdown()

    assert isinstance(result, dict)
    samples = cast(dict[str, Any], result)["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim == 5
    assert samples.shape[1] == 48
    assert torch.isfinite(samples).all()
