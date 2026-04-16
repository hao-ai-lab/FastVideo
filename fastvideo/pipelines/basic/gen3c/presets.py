# SPDX-License-Identifier: Apache-2.0
"""GEN3C model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Camera-controlled denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

GEN3C_COSMOS_7B = InferencePreset(
    name="gen3c_cosmos_7b",
    version=1,
    model_family="gen3c",
    description="GEN3C Cosmos 7B",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 35,
        "trajectory_type": "left",
        "movement_distance": 0.3,
        "camera_rotation": "center_facing",
    },
)

ALL_PRESETS = (GEN3C_COSMOS_7B, )
