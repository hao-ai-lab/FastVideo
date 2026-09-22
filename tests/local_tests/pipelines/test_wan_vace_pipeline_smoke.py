# SPDX-License-Identifier: Apache-2.0
"""Import/registry preflight and optional real Wan-VACE smoke."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest
import torch

MODEL_DIR_ENV = os.getenv("WAN_VACE_MODEL_DIR")


def test_wan_vace_typed_surface_preflight() -> None:
    from fastvideo.api import GeneratorConfig, PipelineSelection
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.pipelines.wan import WanVACE1_3B_Config, WanVACE14B_Config
    from fastvideo.pipelines.basic.wan.wan_vace_pipeline import EntryClass, WanVACEPipeline

    assert WanVACEPipeline.__name__ == "WanVACEPipeline"
    assert EntryClass is WanVACEPipeline
    assert WanVACEPipeline._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    for model_path, preset_name, config_cls in (
        ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "wan_vace_1_3b", WanVACE1_3B_Config),
        ("Wan-AI/Wan2.1-VACE-14B-diffusers", "wan_vace_14b", WanVACE14B_Config),
    ):
        default_preset, model_family = registry.get_preset_selection(model_path)
        assert (default_preset, model_family) == (preset_name, "wan")
        info = registry.get_model_info(
            model_path,
            override_pipeline_cls_name="WanVACEPipeline",
        )
        assert info.pipeline_cls is WanVACEPipeline
        assert info.pipeline_config_cls is config_cls

    vace_presets = {preset.name for preset in get_presets_for_family("wan") if "vace" in preset.name}
    assert vace_presets == {"wan_vace_1_3b", "wan_vace_14b"}

    preset_1_3b = get_preset("wan_vace_1_3b", "wan")
    assert preset_1_3b.defaults["height"] == 480
    assert preset_1_3b.defaults["width"] == 832
    assert preset_1_3b.defaults["num_inference_steps"] == 50
    assert preset_1_3b.defaults["guidance_scale"] == 5.0

    config = WanVACE1_3B_Config()
    assert config.flow_shift == 16.0
    assert config.vae_config.load_encoder is True
    assert config.vae_config.load_decoder is True

    args = generator_config_to_fastvideo_args(
        GeneratorConfig(
            model_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            pipeline=PipelineSelection(),
        ))
    assert isinstance(args.pipeline_config, WanVACE1_3B_Config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan-VACE load/generate smoke requires CUDA")
def test_wan_vace_pipeline_load_generate_smoke() -> None:
    if MODEL_DIR_ENV is None:
        pytest.skip("Set WAN_VACE_MODEL_DIR to activate the real load/generate smoke")
    model_dir = Path(MODEL_DIR_ENV)
    if not (model_dir / "model_index.json").is_file():
        pytest.skip(f"Wan-VACE model_index.json not found under {model_dir}")

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
        output_type="latent",
    )
    try:
        result = generator.generate_video(
            prompt="a red panda reading a book",
            negative_prompt="",
            output_path="outputs/wan_vace/smoke",
            save_video=False,
            return_frames=True,
            height=64,
            width=64,
            num_frames=5,
            fps=16,
            num_inference_steps=1,
            guidance_scale=5.0,
            seed=42,
        )
    finally:
        generator.shutdown()

    result_dict = cast(dict[str, Any], result)
    samples = result_dict["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim in (4, 5)
    assert torch.isfinite(samples).all()
