# SPDX-License-Identifier: Apache-2.0
"""HYWorld model family pipeline presets."""
import numpy as np

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_HYWORLD_SIGMAS = list(np.linspace(1.0, 0.0, 51).tolist()[:-1])

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Camera-controlled denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

HYWORLD_T2V = InferencePreset(
    name="hyworld_t2v",
    version=1,
    model_family="hyworld",
    description="HY-WorldPlay bidirectional at 480p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 125,
        "fps": 24,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": "",
        "pose": "w-31",
        "sigmas": _HYWORLD_SIGMAS,
    },
)

ALL_PRESETS = (HYWORLD_T2V, )
