# SPDX-License-Identifier: Apache-2.0
"""Inference preset for MiniMax-H3-Image."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec
from fastvideo.pipelines.basic.minimax_h3_image.packing import MINIMAX_H3_IMAGE_NUM_STEPS

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Six-step PDD image denoising",
    allowed_overrides=frozenset({"num_inference_steps"}),
)

MINIMAX_H3_IMAGE_T2I = InferencePreset(
    name="minimax_h3_image_t2i",
    version=1,
    model_family="minimax_h3_image",
    description="MiniMax-H3-Image text-to-image and reference editing at a 2048 canvas",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 2048,
        "width": 2048,
        "num_frames": 1,
        "fps": 1,
        "guidance_scale": 1.0,
        "batch_cfg": False,
        "negative_prompt": "",
        "num_inference_steps": MINIMAX_H3_IMAGE_NUM_STEPS,
        "seed": 0,
    },
)

ALL_PRESETS = (MINIMAX_H3_IMAGE_T2I, )

__all__ = ["ALL_PRESETS", "MINIMAX_H3_IMAGE_T2I"]
