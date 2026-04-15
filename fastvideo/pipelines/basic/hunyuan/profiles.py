# SPDX-License-Identifier: Apache-2.0
"""Hunyuan model family pipeline profiles."""
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

HUNYUAN_T2V = PipelineProfile(
    name="hunyuan_t2v",
    version="1",
    model_family="hunyuan",
    description="HunyuanVideo T2V at 720p",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 125,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
    },
)

FAST_HUNYUAN_T2V = PipelineProfile(
    name="fast_hunyuan_t2v",
    version="1",
    model_family="hunyuan",
    description="FastHunyuan T2V at 720p",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 125,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 6,
    },
)

ALL_PROFILES = (HUNYUAN_T2V, FAST_HUNYUAN_T2V)
