# SPDX-License-Identifier: Apache-2.0
"""Smoke / preflight tests for the Flux2 Klein T2I pipeline.

The default test is CPU-safe and validates import, registry, preset, and config
wiring. The optional load/generate smoke is activated with CUDA plus
``FLUX2_MODEL_DIR=/path/to/black-forest-labs__FLUX.2-klein-4B``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


MODEL_DIR = Path(os.getenv("FLUX2_MODEL_DIR", ""))


def test_flux2_klein_typed_surface_preflight() -> None:
    """No-GPU preflight: imports + registry + preset wiring are intact."""
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.pipelines.flux_2 import Flux2KleinPipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.flux_2.flux_2_klein_pipeline import (
        EntryClass,
        Flux2KleinPipeline,
    )

    assert Flux2KleinPipeline.__name__ == "Flux2KleinPipeline"
    assert EntryClass is Flux2KleinPipeline

    default_preset, model_family = registry.get_preset_selection(
        "black-forest-labs/FLUX.2-klein-4B"
    )
    assert model_family == "flux2"
    assert default_preset == "flux2_klein_4b"

    info = registry.get_model_info(
        "black-forest-labs/FLUX.2-klein-4B",
        workload_type=WorkloadType.T2I,
        override_pipeline_cls_name="Flux2KleinPipeline",
    )
    assert info.pipeline_cls is Flux2KleinPipeline
    assert info.pipeline_config_cls is Flux2KleinPipelineConfig

    names = {p.name for p in get_presets_for_family("flux2")}
    assert "flux2_klein_4b" in names
    preset = get_preset("flux2_klein_4b", "flux2")
    assert preset.defaults["num_inference_steps"] == 4
    assert preset.defaults["height"] == 1024
    assert preset.defaults["width"] == 1024
    assert preset.defaults["guidance_scale"] == 1.0
    assert preset.defaults["num_frames"] == 1

    assert Flux2KleinPipeline._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 Klein pipeline load/generate smoke requires CUDA",
)
def test_flux2_klein_pipeline_load_generate_smoke() -> None:
    """Optional real load + four-step latent generate smoke for local weights."""
    if not MODEL_DIR.exists():
        pytest.skip("Set FLUX2_MODEL_DIR to activate Flux2 Klein load/generate smoke")

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(MODEL_DIR),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="Flux2KleinPipeline",
    )
    try:
        result = generator.generate_video(
            prompt="a photo of a banana on a wooden table, studio lighting",
            output_path="outputs_video/flux2_klein_smoke",
            save_video=False,
            return_frames=True,
            height=1024,
            width=1024,
            num_frames=1,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=0,
        )
    finally:
        generator.shutdown()

    samples = result["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim in (3, 5)
    assert torch.isfinite(samples).all()
