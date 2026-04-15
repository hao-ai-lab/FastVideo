# SPDX-License-Identifier: Apache-2.0
"""LongCat model family pipeline profiles."""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

LONGCAT_T2V = PipelineProfile(
    name="longcat_t2v",
    version="1",
    model_family="longcat",
    description="LongCat-Video T2V at 480p",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 848,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
    },
)

LONGCAT_I2V = PipelineProfile(
    name="longcat_i2v",
    version="1",
    model_family="longcat",
    description="LongCat-Video I2V at 480p",
    workload_type="i2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 848,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
    },
)

LONGCAT_VC = PipelineProfile(
    name="longcat_vc",
    version="1",
    model_family="longcat",
    description="LongCat-Video continuation at 480p",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 848,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
    },
)

ALL_PROFILES = (LONGCAT_T2V, LONGCAT_I2V, LONGCAT_VC)
