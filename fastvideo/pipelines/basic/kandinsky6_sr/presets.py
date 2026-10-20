# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 video super-resolution (VSR) presets."""

from dataclasses import asdict

from fastvideo.api.presets import InferencePreset, PresetStageSpec
from fastvideo.configs.pipelines.kandinsky6_sr_options import Kandinsky6SROptions, SR_OPTION_FIELDS

_SR_STAGE = PresetStageSpec(
    name="sr",
    kind="super_resolution",
    description="Tiled latent-upscaler + SR DiT pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        *SR_OPTION_FIELDS,
    }),
)

# Geometry and frame rate come from the source video; height / width / num_frames / fps below only keep the generic
# request validation and logs sensible.  The DiT is text-free, so there is no classifier-free guidance.
_COMMON_DEFAULTS = {
    "height": 512,
    "width": 768,
    "num_frames": 121,
    "fps": 24,
    "seed": 42,
    "guidance_scale": 1.0,
    "save_video": True,
    "return_frames": False,
    **asdict(Kandinsky6SROptions()),
}

KANDINSKY6_SR = InferencePreset(
    name="kandinsky6_sr",
    version=2,
    model_family="kandinsky6_sr",
    description="Kandinsky6 video super-resolution, flow matching (4 Euler steps per tile)",
    workload_type=None,
    stage_schemas=(_SR_STAGE, ),
    defaults={
        **_COMMON_DEFAULTS, "num_inference_steps": 4
    },
)

KANDINSKY6_SR_DISTILLED = InferencePreset(
    name="kandinsky6_sr_distilled",
    version=1,
    model_family="kandinsky6_sr",
    description="Kandinsky6 video super-resolution, 2-step pi-Flow distilled",
    workload_type=None,
    stage_schemas=(_SR_STAGE, ),
    defaults={
        **_COMMON_DEFAULTS, "num_inference_steps": 2
    },
)

ALL_PRESETS = (KANDINSKY6_SR, KANDINSKY6_SR_DISTILLED)
