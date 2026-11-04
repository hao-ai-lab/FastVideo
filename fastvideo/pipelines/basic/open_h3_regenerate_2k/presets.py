# SPDX-License-Identifier: Apache-2.0
"""Inference preset for Open-H3-Regenerate-2K."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_REFINE_STAGE = PresetStageSpec(
    name="refine",
    kind="denoising",
    description="Latent upscale, then three fixed-sigma Euler refine steps",
    allowed_overrides=frozenset({"num_inference_steps"}),
)

OPEN_H3_REGENERATE_2K = InferencePreset(
    name="open_h3_regenerate_2k",
    version=1,
    model_family="open_h3_regenerate_2k",
    description="Open-H3-Regenerate-2K: regenerate a 768P MiniMax-H3 draft video at 2560x1440 (silent output)",
    workload_type=None,
    stage_schemas=(_REFINE_STAGE, ),
    defaults={
        "height": 1440,
        "width": 2560,
        "num_frames": 124,
        "fps": 24,
        "guidance_scale": 1.0,
        "batch_cfg": False,
        "negative_prompt": "",
        "num_inference_steps": 3,
        "seed": 1766580932,
    },
)

ALL_PRESETS = (OPEN_H3_REGENERATE_2K, )

__all__ = ["ALL_PRESETS", "OPEN_H3_REGENERATE_2K"]
