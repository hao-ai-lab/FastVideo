# SPDX-License-Identifier: Apache-2.0
from fastvideo.api.presets import InferencePreset, PresetStageSpec

PJIT_21K = InferencePreset(
    name="pjit_21k",
    version=1,
    model_family="pjit",
    description="Pyramid-JiT 21K text-to-image generation",
    workload_type="t2i",
    stage_schemas=(PresetStageSpec(name="denoise",
                                   kind="denoising",
                                   allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"})), ),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "num_videos_per_prompt": 1,
        "fps": 1,
        "num_inference_steps": 50,
        "guidance_scale": 15.0,
        "seed": 0,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (PJIT_21K, )
