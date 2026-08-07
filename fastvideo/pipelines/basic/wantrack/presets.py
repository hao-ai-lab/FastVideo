# SPDX-License-Identifier: Apache-2.0
"""WanTrack pipeline presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Causal TrackWan Self-Forcing denoising",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

SF_WANTRACK_CAUSAL_I2V = InferencePreset(
    name="sf_wantrack_causal_i2v",
    version=1,
    model_family="wantrack",
    description="Causal WanTrack Self-Forcing I2V (Track-v0)",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 121,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (SF_WANTRACK_CAUSAL_I2V, )
