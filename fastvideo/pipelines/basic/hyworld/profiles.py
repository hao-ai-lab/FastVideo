# SPDX-License-Identifier: Apache-2.0
"""HYWorld model family pipeline profiles."""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Camera-controlled denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

HYWORLD_T2V = PipelineProfile(
    name="hyworld_t2v",
    version="1",
    model_family="hyworld",
    description="HY-WorldPlay bidirectional at 480p",
    workload_type="t2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 125,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
    },
)

ALL_PROFILES = (HYWORLD_T2V, )
