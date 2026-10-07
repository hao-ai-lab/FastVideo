# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-Fast pipeline preset."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Causal-fast denoising pass",
)

LINGBOTWORLD_FAST_I2V = InferencePreset(
    name="lingbotworld_fast_i2v",
    version=1,
    model_family="lingbotworld_fast",
    description="LingBot-World-Fast 14B causal I2V",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "fps": 16,
        "seed": 42,
        "num_frames": 81,
        "height": 480,
        "width": 832,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (LINGBOTWORLD_FAST_I2V, )
