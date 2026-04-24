# SPDX-License-Identifier: Apache-2.0
"""Presets for the daVinci-MagiHuman pipelines.

Scope of this file: the base T2V variant only. Distill / SR variants
live behind their own presets once ported.
"""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

# Mirrors the reference `MagiEvaluator.negative_prompt` (video-side only —
# we do not emit audio in the base T2V pipeline).
_MAGI_HUMAN_NEGATIVE_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, "
                               "works, paintings, images, static, overall gray, worst quality, low "
                               "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                               "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                               "misshapen limbs, fused fingers, still picture, messy background, "
                               "three legs, many people in the background, walking backwards")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Joint video+audio UniPC flow-matching denoise pass.",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

MAGI_HUMAN_BASE_T2V = InferencePreset(
    name="magi_human_base_t2v",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman base text-to-video at 448x256, 4s @ 25 fps. "
                 "Generated audio is discarded in this pipeline; use the upstream "
                 "reference for full audio-visual generation."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 448,
        # num_frames is derived by the pipeline as `seconds*fps + 1`; we
        # surface it here for APIs that expect a concrete default.
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,  # used as video_txt_guidance_scale
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_DISTILL_T2V = InferencePreset(
    name="magi_human_distill_t2v",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman DMD-2 distilled text-to-video at 448x256, 4s @ "
                 "25 fps. 8-step inference, no classifier-free guidance. Generated "
                 "audio is discarded."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 448,
        "num_frames": 101,
        "fps": 25,
        # DMD: cfg=1 at the pipeline level. guidance_scale is kept at 1.0
        # for interop; the DenoisingStage ignores it when cfg_number=1.
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (MAGI_HUMAN_BASE_T2V, MAGI_HUMAN_DISTILL_T2V)
