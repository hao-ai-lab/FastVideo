# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the LingBot World 2 causal-fast pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch


MODEL_DIR = Path("/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast-fastvideo")
FASTVIDEO_ROOT = Path(__file__).resolve().parents[3]
ACTION_PATH = FASTVIDEO_ROOT / "examples" / "dataset" / "lingbotworld2"


def test_lingbotworld2_checkpoint_selects_expected_pipeline_and_defaults() -> None:
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.pipelines.lingbotworld2 import LingBotWorld2CausalFastI2V480PConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.lingbotworld2.causal_fast_pipeline import (
        EntryClass,
        LingBotWorld2CausalFastPipeline,
    )

    assert EntryClass is LingBotWorld2CausalFastPipeline
    assert LingBotWorld2CausalFastPipeline._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    default_preset, model_family = registry.get_preset_selection(str(MODEL_DIR))
    assert model_family == "lingbotworld2"
    assert default_preset == "lingbotworld2_causal_fast_i2v"

    info = registry.get_model_info(
        str(MODEL_DIR),
        workload_type=WorkloadType.I2V,
        override_pipeline_cls_name="LingBotWorld2CausalFastPipeline",
    )
    assert info.pipeline_cls is LingBotWorld2CausalFastPipeline
    assert info.pipeline_config_cls is LingBotWorld2CausalFastI2V480PConfig

    names = {preset.name for preset in get_presets_for_family("lingbotworld2")}
    assert "lingbotworld2_causal_fast_i2v" in names
    preset = get_preset("lingbotworld2_causal_fast_i2v", "lingbotworld2")
    assert preset.defaults["num_inference_steps"] == 4
    assert preset.defaults["height"] == 480
    assert preset.defaults["width"] == 832
    assert preset.defaults["num_frames"] == 65
    assert preset.defaults["guidance_scale"] == 1.0

    cfg = LingBotWorld2CausalFastI2V480PConfig()
    arch = cfg.dit_config.arch_config
    assert arch.model_type == "i2v"
    assert arch.in_dim == 36
    assert arch.local_attn_size == 18
    assert arch.sink_size == 6
    assert arch.chunk_size == 4
    assert tuple(arch.timesteps_index) == (0, 250, 500, 750)


@pytest.mark.skipif(
    os.getenv("LINGBOTWORLD2_RUN_HEAVY_SMOKE") != "1",
    reason="Set LINGBOTWORLD2_RUN_HEAVY_SMOKE=1 to load the 14B causal-fast model.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="LingBot World 2 heavy smoke requires CUDA")
def test_lingbotworld2_14b_generates_finite_latents_on_8_gpus() -> None:
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(MODEL_DIR),
        num_gpus=8,
        sp_size=8,
        hsdp_shard_dim=8,
        use_fsdp_inference=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
        output_type="latent",
        override_pipeline_cls_name="LingBotWorld2CausalFastPipeline",
    )
    try:
        result = generator.generate_video(
            "A serene lakeside scene with a lone tree standing in calm water.",
            image_path=str(ACTION_PATH / "image.jpg"),
            action_path=str(ACTION_PATH),
            output_path="/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/outputs/fastvideo/heavy_smoke",
            save_video=False,
            return_frames=True,
            height=480,
            width=832,
            num_frames=17,
            num_inference_steps=4,
            guidance_scale=1.0,
            negative_prompt="",
            fps=16,
            seed=42,
        )
    finally:
        generator.shutdown()

    samples = cast(dict[str, Any], result)["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim == 5
    assert samples.shape[1] == 16
    assert torch.isfinite(samples).all()
