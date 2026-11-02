# SPDX-License-Identifier: Apache-2.0
"""Inference presets for Pika H3 Flex joint video/audio generation."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

# The six-forward ladder is the adapter's training contract. Exposing a stage
# override would let a request silently sample the adapter outside its
# distribution, so the denoise stage accepts no overrides.
_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Pika's fixed six-forward re-noise ladder",
    allowed_overrides=frozenset(),
)

_SHARED_DEFAULTS = {
    "fps": 24,
    "guidance_scale": 1.0,
    "batch_cfg": False,
    "negative_prompt": "",
    "num_inference_steps": 6,
    "seed": 0,
    "height": 768,
    "width": 1344,
    "num_frames": 124,
}

PIKA_H3_FLEX_T2VA = InferencePreset(
    name="pika_h3_flex_t2va",
    version=1,
    model_family="pika_h3_flex",
    description="Pika H3 Flex text-to-video with synchronized stereo audio at 768p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

PIKA_H3_FLEX_FL2VA = InferencePreset(
    name="pika_h3_flex_fl2va",
    version=1,
    model_family="pika_h3_flex",
    description="Pika H3 Flex first/last-frame-to-video with synchronized stereo audio",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

PIKA_H3_FLEX_REF2VA = InferencePreset(
    name="pika_h3_flex_ref2va",
    version=1,
    model_family="pika_h3_flex",
    description="Pika H3 Flex ordered image/video/audio references to joint video and stereo audio",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

ALL_PRESETS = (PIKA_H3_FLEX_T2VA, PIKA_H3_FLEX_FL2VA, PIKA_H3_FLEX_REF2VA)

__all__ = [
    "ALL_PRESETS",
    "PIKA_H3_FLEX_FL2VA",
    "PIKA_H3_FLEX_REF2VA",
    "PIKA_H3_FLEX_T2VA",
]
