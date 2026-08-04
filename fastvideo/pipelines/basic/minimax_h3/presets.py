# SPDX-License-Identifier: Apache-2.0
"""Inference presets for MiniMax H3 joint video/audio generation."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Joint video/audio flow-matching denoising",
    allowed_overrides=frozenset({"num_inference_steps"}),
)

_SHARED_DEFAULTS = {
    "fps": 24,
    "guidance_scale": 1.0,
    "batch_cfg": False,
    "negative_prompt": "",
    "num_inference_steps": 50,
    "seed": 0,
}

MINIMAX_H3_T2VA = InferencePreset(
    name="minimax_h3_t2va",
    version=1,
    model_family="minimax_h3",
    description="MiniMax H3 text-to-video with synchronized stereo audio at 768p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        **_SHARED_DEFAULTS,
        "height": 768,
        "width": 1344,
        "num_frames": 124,
    },
)

MINIMAX_H3_FL2VA = InferencePreset(
    name="minimax_h3_fl2va",
    version=1,
    model_family="minimax_h3",
    description="MiniMax H3 first/last-frame-to-video with synchronized stereo audio",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        **_SHARED_DEFAULTS,
        "height": 768,
        "width": 1344,
        "num_frames": 192,
    },
)

MINIMAX_H3_REF2VA = InferencePreset(
    name="minimax_h3_ref2va",
    version=1,
    model_family="minimax_h3",
    description="MiniMax H3 ordered image/video/audio references to joint video and stereo audio",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        **_SHARED_DEFAULTS,
        "height": 768,
        "width": 1344,
        "num_frames": 124,
    },
)

ALL_PRESETS = (MINIMAX_H3_T2VA, MINIMAX_H3_FL2VA, MINIMAX_H3_REF2VA)

__all__ = [
    "ALL_PRESETS",
    "MINIMAX_H3_FL2VA",
    "MINIMAX_H3_REF2VA",
    "MINIMAX_H3_T2VA",
]
