# SPDX-License-Identifier: Apache-2.0
"""TurboDiffusion model family pipeline profiles."""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Fast few-step denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

TURBO_T2V_1_3B = PipelineProfile(
    name="turbo_t2v_1_3b",
    version="1",
    model_family="turbodiffusion",
    description="TurboWan 2.1 T2V 1.3B (4-step)",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
    },
)

TURBO_T2V_14B = PipelineProfile(
    name="turbo_t2v_14b",
    version="1",
    model_family="turbodiffusion",
    description="TurboWan 2.1 T2V 14B (4-step)",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
    },
)

TURBO_I2V_A14B = PipelineProfile(
    name="turbo_i2v_a14b",
    version="1",
    model_family="turbodiffusion",
    description="TurboWan 2.2 I2V A14B (4-step)",
    workload_type="i2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
    },
)

ALL_PROFILES = (
    TURBO_T2V_1_3B,
    TURBO_T2V_14B,
    TURBO_I2V_A14B,
)
