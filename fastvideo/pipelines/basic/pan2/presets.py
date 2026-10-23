# SPDX-License-Identifier: Apache-2.0
"""Inference presets for PAN2 text-to-video and image-to-video."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="PAN2 flow-matching denoising",
    allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
)

# 704x1248, 313 frames (13 s at 24 fps), 50 steps, CFG 3. The preset sends no negative prompt (an empty one is encoded
# as a single space); the reference sampler sends the release's structured negative caption, so pass it as the
# request's ``negative_prompt`` to follow that recipe.
_SHARED_DEFAULTS = {
    "height": 704,
    "width": 1248,
    "num_frames": 313,
    "fps": 24,
    "num_inference_steps": 50,
    "guidance_scale": 3.0,
    "negative_prompt": "",
}

PAN2_T2V = InferencePreset(
    name="pan2_t2v",
    version=1,
    model_family="pan2",
    description="PAN2 text-to-video",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

# The same pipeline runs image-to-video when the request carries a first frame (image_path).
PAN2_I2V = InferencePreset(
    name="pan2_i2v",
    version=1,
    model_family="pan2",
    description="PAN2 image-to-video from a first frame",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

ALL_PRESETS = (PAN2_T2V, PAN2_I2V)

__all__ = ["ALL_PRESETS", "PAN2_I2V", "PAN2_T2V"]
