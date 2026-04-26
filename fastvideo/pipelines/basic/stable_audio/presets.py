# SPDX-License-Identifier: Apache-2.0
"""Presets for Stable Audio Open 1.0 text-to-audio."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Stable Audio Cosine-DPM++ denoising with text + duration CFG.",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

STABLE_AUDIO_OPEN_1_0_BASE = InferencePreset(
    name="stable_audio_open_1_0_base",
    version=1,
    model_family="stable_audio",
    description=("Stability AI Stable Audio Open 1.0 text-to-audio. Generates up "
                 "to ~47.5s of stereo 44.1 kHz audio per call. Default duration "
                 "is 10s; raise via `audio_end_in_s` up to the model max."),
    workload_type="t2v",  # NOTE: WorkloadType has no T2A variant yet (REVIEW item 28)
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 0,
        "guidance_scale": 7.0,
        "num_inference_steps": 100,
        "negative_prompt": "",
        # audio_end_in_s / audio_start_in_s are pipeline-call kwargs;
        # not surfaced as preset defaults to keep them explicit.
    },
)

ALL_PRESETS = (STABLE_AUDIO_OPEN_1_0_BASE, )
