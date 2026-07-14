# SPDX-License-Identifier: Apache-2.0
"""Official LingBot-Video T2V inference presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

DEFAULT_NEGATIVE_PROMPT = ('{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", '
                           '"pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", '
                           '"underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], '
                           '"artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", '
                           '"cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", '
                           '"signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in '
                           'landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", '
                           '"motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", '
                           '"unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], '
                           '"material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed '
                           'bottle", "liquid freezing improperly", "distorted reflections"]}}')

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="LingBot-Video batched-CFG denoising",
    allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
)

_REFINE_STAGE = PresetStageSpec(
    name="refine",
    kind="refinement",
    description="LingBot-Video pixel-space resize, VAE re-encode, and refiner denoising",
    allowed_overrides=frozenset({
        "height_sr",
        "width_sr",
        "num_inference_steps_sr",
        "guidance_scale_2",
        "t_thresh",
    }),
)

LINGBOT_VIDEO_DENSE_T2V = InferencePreset(
    name="lingbot_video_dense_t2v",
    version=1,
    model_family="lingbot_video",
    description="LingBot-Video Dense 1.3B text-to-video at 480p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 40,
        "guidance_scale": 3.0,
        "batch_cfg": True,
        "seed": 42,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
)

LINGBOT_VIDEO_MOE_REFINER_T2V = InferencePreset(
    name="lingbot_video_moe_refiner_t2v",
    version=1,
    model_family="lingbot_video",
    description="LingBot-Video MoE 30B-A3B T2V with 1080p refiner",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, _REFINE_STAGE),
    defaults={
        "height": 480,
        "width": 832,
        "height_sr": 1088,
        "width_sr": 1920,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 40,
        "num_inference_steps_sr": 8,
        "guidance_scale": 3.0,
        "guidance_scale_2": 3.0,
        "batch_cfg": True,
        "t_thresh": 0.85,
        "seed": 42,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    stage_defaults={
        "refine": {
            "height_sr": 1088,
            "width_sr": 1920,
            "num_inference_steps_sr": 8,
            "guidance_scale_2": 3.0,
            "t_thresh": 0.85,
        },
    },
)

ALL_PRESETS = (LINGBOT_VIDEO_DENSE_T2V, LINGBOT_VIDEO_MOE_REFINER_T2V)
