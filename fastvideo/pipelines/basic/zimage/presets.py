# SPDX-License-Identifier: Apache-2.0
"""Z-Image inference presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Z-Image denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
        "cfg_normalization",
        "cfg_truncation",
    }),
)

ZIMAGE_TURBO = InferencePreset(
    name="zimage_turbo",
    version=1,
    model_family="zimage",
    description="Z-Image-Turbo text-to-image generation",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 42,
        "guidance_scale": 0.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
        "max_sequence_length": 512,
        "cfg_normalization": False,
        "cfg_truncation": 1.0,
    },
)

ALL_PRESETS = (ZIMAGE_TURBO, )
