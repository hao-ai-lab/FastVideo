# SPDX-License-Identifier: Apache-2.0
"""Smoke / preflight tests for the daVinci-MagiHuman base text-to-AV pipeline.

Two tests:

  * `test_magi_human_typed_surface_preflight` — pure-Python, no GPU, no
    weights. Verifies that the scaffold is importable, that the preset
    registers cleanly, and that the DiT module tree matches the upstream
    HuggingFace checkpoint shape-for-shape on `meta` device. This is what
    CI should run on every PR.

  * `test_magi_human_pipeline_smoke` — end-to-end pipeline construction +
    a tiny generate_video call, gated on local converted-weights paths.
    Skips cleanly when weights are missing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch


def test_magi_human_typed_surface_preflight() -> None:
    """Import + registry + module-tree surface check.

    Covers regressions that would otherwise only surface on a GPU host:
    preset drop from ALL_PRESETS, renamed modules, registry mis-wiring,
    or DiT module-tree drift from the upstream checkpoint.
    """
    import fastvideo.registry  # noqa: F401 — triggers preset registration
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.models.dits.magi_human import (
        MagiHumanArchConfig,
        MagiHumanVideoConfig,
    )
    from fastvideo.configs.models.encoders.t5gemma import (
        T5GemmaEncoderArchConfig,
        T5GemmaEncoderConfig,
    )
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from fastvideo.pipelines.basic.magi_human.magi_human_pipeline import (  # noqa: F401
        MagiHumanI2VPipeline,
        MagiHumanPipeline,
        MagiHumanSRI2VPipeline,
        MagiHumanSRPipeline,
    )
    from fastvideo.pipelines.basic.magi_human.pipeline_configs import (
        MagiHumanBaseConfig,
        MagiHumanBaseI2VConfig,
        MagiHumanDistillI2VConfig,
        MagiHumanSR540pConfig,
        MagiHumanSR540pI2VConfig,
    )
    from fastvideo.pipelines.basic.magi_human.stages import (  # noqa: F401
        MagiHumanDenoisingStage,
        MagiHumanLatentPreparationStage,
        MagiHumanReferenceImageStage,
        MagiHumanSRDenoisingStage,
        MagiHumanSRLatentPreparationStage,
    )

    # Presets are registered under the expected family.
    names = {p.name for p in get_presets_for_family("magi_human")}
    assert names == {
        "magi_human_base",
        "magi_human_distill",
        "magi_human_base_ti2v",
        "magi_human_distill_ti2v",
        "magi_human_sr_540p",
        "magi_human_sr_540p_ti2v",
    }

    base_preset = get_preset("magi_human_base", "magi_human")
    assert base_preset.workload_type == "t2v"
    assert base_preset.defaults["num_inference_steps"] == 32
    assert base_preset.defaults["fps"] == 25

    distill_preset = get_preset("magi_human_distill", "magi_human")
    assert distill_preset.workload_type == "t2v"
    assert distill_preset.defaults["num_inference_steps"] == 8
    assert distill_preset.defaults["guidance_scale"] == 1.0

    base_ti2v_preset = get_preset("magi_human_base_ti2v", "magi_human")
    assert base_ti2v_preset.workload_type == "i2v"
    assert base_ti2v_preset.defaults["num_inference_steps"] == 32

    distill_ti2v_preset = get_preset("magi_human_distill_ti2v", "magi_human")
    assert distill_ti2v_preset.workload_type == "i2v"
    assert distill_ti2v_preset.defaults["num_inference_steps"] == 8

    sr_preset = get_preset("magi_human_sr_540p", "magi_human")
    assert sr_preset.workload_type == "t2v"
    assert sr_preset.defaults["num_inference_steps"] == 32

    sr_ti2v_preset = get_preset("magi_human_sr_540p_ti2v", "magi_human")
    assert sr_ti2v_preset.workload_type == "i2v"
    assert sr_ti2v_preset.defaults["num_inference_steps"] == 32

    # Distill pipeline config: same arch as base, CFG=1, 8 steps.
    from fastvideo.pipelines.basic.magi_human.pipeline_configs import MagiHumanDistillConfig
    distill_pc = MagiHumanDistillConfig()
    assert distill_pc.num_inference_steps == 8
    assert distill_pc.cfg_number == 1
    assert distill_pc.dit_config.arch_config.num_layers == 40

    base_i2v_pc = MagiHumanBaseI2VConfig()
    assert base_i2v_pc.image_conditioning is True
    assert base_i2v_pc.vae_config.load_encoder is True
    assert base_i2v_pc.vae_config.load_decoder is True

    distill_i2v_pc = MagiHumanDistillI2VConfig()
    assert distill_i2v_pc.num_inference_steps == 8
    assert distill_i2v_pc.cfg_number == 1
    assert distill_i2v_pc.image_conditioning is True
    assert distill_i2v_pc.vae_config.load_encoder is True

    sr_pc = MagiHumanSR540pConfig()
    assert sr_pc.num_inference_steps == 32
    assert sr_pc.sr_num_inference_steps == 5
    assert sr_pc.noise_value == 220
    assert sr_pc.sr_audio_noise_scale == 0.7
    assert sr_pc.sr_video_txt_guidance_scale == 3.5
    assert sr_pc.sr_height == 512
    assert sr_pc.sr_width == 896

    sr_i2v_pc = MagiHumanSR540pI2VConfig()
    assert sr_i2v_pc.image_conditioning is True
    assert sr_i2v_pc.vae_config.load_encoder is True

    # Config constructs with the documented defaults.
    pc = MagiHumanBaseConfig()
    assert pc.flow_shift == 5.0
    assert pc.cfg_number == 2
    assert pc.num_inference_steps == 32
    assert pc.dit_config.arch_config.num_layers == 40
    assert pc.dit_config.arch_config.hidden_size == 5120
    assert pc.dit_config.arch_config.num_attention_heads == 40
    assert pc.dit_config.arch_config.num_heads_kv == 8
    assert pc.dit_config.arch_config.mm_layers == (0, 1, 2, 3, 36, 37, 38, 39)
    assert pc.text_encoder_configs[0].arch_config.hidden_size == 3584

    # The DiT module tree matches the upstream HF base/ checkpoint
    # shape-for-shape. This is checkpoint-loading parity, not numerical
    # parity — but a regression here means loaded weights won't align.
    dit_cfg = MagiHumanVideoConfig()
    with torch.device("meta"):
        dit = MagiHumanDiT(dit_cfg)
    fv_shapes = {n: tuple(p.shape) for n, p in dit.state_dict().items()}

    index_path = _hf_index_path_or_none()
    if index_path is None:
        pytest.skip("HF repo unavailable (no network / no token) — "
                    "skipping cross-check against GAIR/daVinci-MagiHuman.")
    with open(index_path) as f:
        wmap = json.load(f)["weight_map"]
    hf_keys = set(wmap.keys())
    fv_keys = set(fv_shapes.keys())

    missing_in_fv = sorted(hf_keys - fv_keys)
    extra_in_fv = sorted(fv_keys - hf_keys)
    assert not missing_in_fv, f"fastvideo missing keys: {missing_in_fv[:5]}"
    assert not extra_in_fv, f"fastvideo extra keys: {extra_in_fv[:5]}"
    assert len(fv_keys) == 331


def _hf_index_path_or_none() -> str | None:
    """Return a local path to the base/ index.json, or None if unavailable."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return hf_hub_download(
            repo_id="GAIR/daVinci-MagiHuman",
            filename="base/model.safetensors.index.json",
        )
    except Exception:
        return None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman pipeline smoke test requires CUDA.",
)
def test_magi_human_pipeline_smoke() -> None:
    """End-to-end smoke: build the pipeline and run a tiny denoise.

    Skips cleanly when the converted-weights directory is not present.
    """
    diffusers_path = os.getenv(
        "MAGI_HUMAN_DIFFUSERS_PATH",
        "converted_weights/magi_human_base",
    )
    if not os.path.isdir(diffusers_path):
        pytest.skip(
            f"Missing converted MagiHuman repo at {diffusers_path}. "
            f"Run scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py "
            f"first."
        )
    if not os.path.isfile(os.path.join(diffusers_path, "model_index.json")):
        pytest.skip(f"Missing model_index.json in {diffusers_path}")

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

    from fastvideo import VideoGenerator

    # Small shapes to keep the smoke test cheap.
    prompt = "A cheerful person waving at the camera in a well-lit room."
    seed = 42
    height = 256
    width = 448
    num_frames = 13   # seconds=1, 12fps for smoke; the pipeline derives it
    fps = 12.0
    steps = 2

    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    try:
        result = generator.generate_video(
            prompt=prompt,
            output_path="outputs_video/magi_human_smoke",
            save_video=False,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=steps,
            seed=seed,
        )
    finally:
        generator.shutdown()

    samples = result["samples"]
    assert samples.ndim == 5, f"expected [B,C,T,H,W], got {samples.shape}"
    assert samples.shape[0] == 1
