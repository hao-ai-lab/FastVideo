# SPDX-License-Identifier: Apache-2.0
"""Inference preset for the official Helios-Distilled checkpoint."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

HELIOS_DISTILLED_NEGATIVE_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, "
                                    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
                                    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
                                    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
                                    "still picture, messy background, three legs, many people in the background, "
                                    "walking backwards")

_PYRAMID_STAGE = PresetStageSpec(
    name="pyramid_denoise",
    kind="denoising",
    description="Three-level autoregressive DMD denoising",
    allowed_overrides=frozenset({
        "guidance_scale",
        "pyramid_num_inference_steps_list",
        "is_amplify_first_chunk",
    }),
)

HELIOS_DISTILLED_T2V = InferencePreset(
    name="helios_distilled_t2v",
    version=1,
    model_family="helios",
    description="Helios-Distilled autoregressive text-to-video",
    workload_type="t2v",
    stage_schemas=(_PYRAMID_STAGE, ),
    defaults={
        "height": 384,
        "width": 640,
        "num_frames": 240,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 2,
        "pyramid_num_inference_steps_list": [2, 2, 2],
        "history_sizes": [16, 2, 1],
        "num_latent_frames_per_chunk": 9,
        "keep_first_frame": True,
        "is_skip_first_chunk": False,
        "use_zero_init": True,
        "zero_steps": 1,
        "is_amplify_first_chunk": True,
        "max_sequence_length": 512,
        "negative_prompt": HELIOS_DISTILLED_NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (HELIOS_DISTILLED_T2V, )
