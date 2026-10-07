# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 model family pipeline presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

# Byte-identical to the diffusers Kandinsky6TI2VAPipeline's
# _DEFAULT_NEGATIVE_PROMPT (and to Kandinsky5's own default negative prompt --
# same lineage, same default).
_NEGATIVE_PROMPT = ("Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, "
                    "deformed, walking backwards")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main Kandinsky6 joint video+audio denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

# Diffusers Kandinsky6TI2VAPipeline defaults
# (height=512, width=768, num_frames=121, num_inference_steps=50,
# guidance_scale=5.0, sample_fps=24.0). One preset serves the single TI2VA
# pipeline whether or not a conditioning image_path is passed at call time.
KANDINSKY6_TI2VA = InferencePreset(
    name="kandinsky6_ti2va",
    version=1,
    model_family="kandinsky6",
    description="Kandinsky6 TI2VA (Pro)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

# pi-Flow distilled checkpoint (Kandinsky-6.0-Pro-distill-5s-Diffusers): distilled for 10 steps (model card; its
# scheduler_config.json leaves nfe unset) and the pi-Flow policy has no classifier-free guidance, so the run is 10
# steps at guidance_scale=1.0 (the diffusers pipeline rejects any other guidance for a PiflowScheduler).
KANDINSKY6_TI2VA_DISTILLED = InferencePreset(
    name="kandinsky6_ti2va_distilled",
    version=1,
    model_family="kandinsky6",
    description="Kandinsky6 TI2VA pi-Flow distilled (Pro-distill)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 10,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (KANDINSKY6_TI2VA, KANDINSKY6_TI2VA_DISTILLED)
