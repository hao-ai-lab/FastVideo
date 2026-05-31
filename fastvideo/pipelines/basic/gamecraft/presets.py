# SPDX-License-Identifier: Apache-2.0
"""HunyuanGameCraft model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Action-controlled denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

GAMECRAFT_I2V = InferencePreset(
    name="gamecraft_i2v",
    version=1,
    model_family="gamecraft",
    description="HunyuanGameCraft I2V at 704x1280",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 33,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (GAMECRAFT_I2V, )
