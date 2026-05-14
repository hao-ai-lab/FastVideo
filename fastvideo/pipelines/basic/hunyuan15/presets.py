# SPDX-License-Identifier: Apache-2.0
"""Hunyuan 1.5 model family pipeline presets."""
import numpy as np

from fastvideo.api.presets import InferencePreset, PresetStageSpec


def _sigmas(n: int) -> list[float]:
    """Precompute sigmas schedule for *n* inference steps."""
    return np.linspace(1.0, 0.0, n + 1).tolist()[:-1]


_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

_SR_STAGE = PresetStageSpec(
    name="sr",
    kind="super_resolution",
    description="Super-resolution upscaling pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
    }),
)

# -------------------------------------------------------------------
# Hunyuan 1.5 T2V presets
# -------------------------------------------------------------------

HUNYUAN15_T2V_480P = InferencePreset(
    name="hunyuan15_t2v_480p",
    version=1,
    model_family="hunyuan15",
    description="HunyuanVideo 1.5 T2V at 480p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 848,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
        "sigmas": _sigmas(50),
    },
)

HUNYUAN15_T2V_720P = InferencePreset(
    name="hunyuan15_t2v_720p",
    version=1,
    model_family="hunyuan15",
    description="HunyuanVideo 1.5 T2V at 720p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
        "sigmas": _sigmas(50),
    },
)

# -------------------------------------------------------------------
# Hunyuan 1.5 I2V presets
# -------------------------------------------------------------------

HUNYUAN15_I2V_480P_DISTILLED = InferencePreset(
    name="hunyuan15_i2v_480p_distilled",
    version=1,
    model_family="hunyuan15",
    description="HunyuanVideo 1.5 I2V 480p step-distilled",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 12,
        "negative_prompt": "",
        "sigmas": _sigmas(12),
    },
)

HUNYUAN15_I2V_720P_DISTILLED = InferencePreset(
    name="hunyuan15_i2v_720p_distilled",
    version=1,
    model_family="hunyuan15",
    description="HunyuanVideo 1.5 I2V 720p distilled",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
        "sigmas": _sigmas(50),
    },
)

# -------------------------------------------------------------------
# Hunyuan 1.5 SR preset (two-stage)
# -------------------------------------------------------------------

HUNYUAN15_SR_1080P = InferencePreset(
    name="hunyuan15_sr_1080p",
    version=1,
    model_family="hunyuan15",
    description="HunyuanVideo 1.5 SR to 1080p (two-stage)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, _SR_STAGE),
    defaults={
        "height": 480,
        "width": 848,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 12,
        "negative_prompt": "",
        "sigmas": _sigmas(12),
    },
    stage_defaults={
        "sr": {
            "height_sr": 1072,
            "width_sr": 1920,
            "num_inference_steps": 8,
        },
    },
)

ALL_PRESETS = (
    HUNYUAN15_T2V_480P,
    HUNYUAN15_T2V_720P,
    HUNYUAN15_I2V_480P_DISTILLED,
    HUNYUAN15_I2V_720P_DISTILLED,
    HUNYUAN15_SR_1080P,
)
