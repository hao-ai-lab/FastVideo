# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.0 pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Iterative denoising pass with action and camera conditioning",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

MATRIXGAME3_I2V = InferencePreset(
    name="matrixgame3_i2v",
    version=1,
    model_family="matrixgame",
    description="Matrix-Game 3.0 I2V (5B, 720p)",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 57,
        "fps": 25,
        "guidance_scale": 1.0,
        "num_inference_steps": 3,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (MATRIXGAME3_I2V, )
