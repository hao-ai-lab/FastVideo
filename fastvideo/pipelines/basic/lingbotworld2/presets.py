# SPDX-License-Identifier: Apache-2.0
"""LingBotWorld2 causal-fast pipeline preset."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Causal-fast denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

LINGBOTWORLD2_CAUSAL_FAST_I2V = InferencePreset(
    name="lingbotworld2_causal_fast_i2v",
    version=1,
    model_family="lingbotworld2",
    description="LingBot World 2 14B causal-fast I2V",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "fps": 16,
        "seed": 42,
        "num_frames": 65,
        "height": 480,
        "width": 832,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (LINGBOTWORLD2_CAUSAL_FAST_I2V, )
