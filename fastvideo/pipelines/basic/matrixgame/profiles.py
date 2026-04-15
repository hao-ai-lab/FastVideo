# SPDX-License-Identifier: Apache-2.0
"""MatrixGame model family pipeline profiles."""
from fastvideo.api.profiles import PipelineProfile, ProfileStageSpec

_DENOISE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    description="Causal denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

MATRIXGAME_I2V = PipelineProfile(
    name="matrixgame_i2v",
    version="1",
    model_family="matrixgame",
    description="Matrix-Game 2.0 I2V",
    workload_type="i2v",
    stages=(_DENOISE_STAGE, ),
    defaults={
        "height": 352,
        "width": 640,
        "num_frames": 57,
        "fps": 25,
        "guidance_scale": 1.0,
        "num_inference_steps": 3,
        "negative_prompt": None,
    },
)

ALL_PROFILES = (MATRIXGAME_I2V, )
