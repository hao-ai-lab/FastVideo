# SPDX-License-Identifier: Apache-2.0
"""LTX2 model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_LTX2_NEGATIVE_PROMPT = ("blurry, out of focus, overexposed, underexposed, low contrast, "
                         "washed out colors, excessive noise, grainy texture, poor lighting, "
                         "flickering, motion blur, distorted proportions, unnatural skin "
                         "tones, deformed facial features, asymmetrical face, missing facial "
                         "features, extra limbs, disfigured hands, wrong hand count, "
                         "artifacts around text, inconsistent perspective, camera shake, "
                         "incorrect depth of field, background too sharp, background clutter, "
                         "distracting reflections, harsh shadows, inconsistent lighting "
                         "direction, color banding, cartoonish rendering, 3D CGI look, "
                         "unrealistic materials, uncanny valley effect, incorrect ethnicity, "
                         "wrong gender, exaggerated expressions, wrong gaze direction, "
                         "mismatched lip sync, silent or muted audio, distorted voice, "
                         "robotic voice, echo, background noise, off-sync audio, incorrect "
                         "dialogue, added dialogue, repetitive speech, jittery movement, "
                         "awkward pauses, incorrect timing, unnatural transitions, "
                         "inconsistent framing, tilted camera, flat lighting, inconsistent "
                         "tone, cinematic oversaturation, stylized filters, or AI artifacts.")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

_REFINE_STAGE = PresetStageSpec(
    name="refine",
    kind="refinement",
    description="Latent-upsample + second-pass refine",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
        "image_crf",
        "video_position_offset_sec",
    }),
)

LTX2_BASE = InferencePreset(
    name="ltx2_base",
    version=1,
    model_family="ltx2",
    description="LTX-2 base at 512x768",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 3.0,
        "num_inference_steps": 40,
        "negative_prompt": _LTX2_NEGATIVE_PROMPT,
        "ltx2_cfg_scale_video": 3.0,
        "ltx2_cfg_scale_audio": 7.0,
        "ltx2_modality_scale_video": 3.0,
        "ltx2_modality_scale_audio": 3.0,
        "ltx2_rescale_scale": 0.7,
        "ltx2_stg_scale_video": 1.0,
        "ltx2_stg_scale_audio": 1.0,
        "ltx2_stg_blocks_video": [29],
        "ltx2_stg_blocks_audio": [29],
    },
)

LTX2_DISTILLED = InferencePreset(
    name="ltx2_distilled",
    version=1,
    model_family="ltx2",
    description="LTX-2 distilled at 1024x1536",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 1024,
        "width": 1536,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
    },
)

# Two-stage distilled flow: half-res denoise (stage 1) followed by
# spatial-upsample + second denoise (stage 2 "refine"). Stage-2
# sigmas are fixed in fastvideo.pipelines.stages.ltx2_refine — only
# num_inference_steps (2 or 3) and guidance_scale are user-tunable.
# Refine wiring (upsampler weights, optional LoRA, enable/add-noise)
# lives on generator.pipeline.preset_overrides.refine at init-time;
# per-request tuning goes through request.stage_overrides.refine.
LTX2_TWO_STAGE = InferencePreset(
    name="ltx2_two_stage",
    version=1,
    model_family="ltx2",
    description="LTX-2 distilled with 2x spatial refine (stage 1 half-res + stage 2 upsample+denoise)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, _REFINE_STAGE),
    defaults={
        "seed": 10,
        "height": 1024,
        "width": 1536,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
    },
    stage_defaults={
        "refine": {
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
        },
    },
)

ALL_PRESETS = (LTX2_BASE, LTX2_DISTILLED, LTX2_TWO_STAGE)
