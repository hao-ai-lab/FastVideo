# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 (Cosmos3-Nano) inference presets.

Defaults track the official ``cosmos-framework`` ``sample_args`` for the video
paths (``text2video`` / ``image2video``: guidance=6.0, num_steps=35, shift=10.0,
fps=24, num_frames=189) and ``text2image`` (guidance=4.0, num_steps=50,
shift=3.0). The default resolution is 16:9 at a VAE-aligned 704x1280 (spatial
compression 16 -> 44x80 latent grid).
"""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Cosmos3 sequential-CFG UniPC denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

# Framework video negative prompt (Cosmos quality prompt).
COSMOS3_VIDEO_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality.")

COSMOS3_NANO = InferencePreset(
    name="cosmos3_nano",
    version=1,
    model_family="cosmos3",
    description="Cosmos3-Nano text-to-video",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 189,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 35,
        "negative_prompt": COSMOS3_VIDEO_NEGATIVE_PROMPT,
    },
)

COSMOS3_NANO_I2V = InferencePreset(
    name="cosmos3_nano_i2v",
    version=1,
    model_family="cosmos3",
    description="Cosmos3-Nano image-to-video",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 189,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 35,
        "negative_prompt": COSMOS3_VIDEO_NEGATIVE_PROMPT,
    },
)

COSMOS3_NANO_T2I = InferencePreset(
    name="cosmos3_nano_t2i",
    version=1,
    model_family="cosmos3",
    description="Cosmos3-Nano text-to-image",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 24,
        "guidance_scale": 4.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (COSMOS3_NANO, COSMOS3_NANO_I2V, COSMOS3_NANO_T2I)
