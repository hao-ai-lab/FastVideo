# SPDX-License-Identifier: Apache-2.0
"""Stable Audio presets.

Sampling defaults track the published HF model card
(https://huggingface.co/stabilityai/stable-audio-open-1.0):
100 steps, CFG=7, dpmpp-3m-sde, sigma_min=0.3, sigma_max=500, rho=1.0.
"""
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

# `audio_start_in_s` / `audio_end_in_s` are call-kwargs, kept off here.
_SHARED_DEFAULTS = {
    "seed": 0,
    "guidance_scale": 7.0,
    "num_inference_steps": 100,
    "negative_prompt": "",
}

STABLE_AUDIO_OPEN_1_0_BASE = InferencePreset(
    name="stable_audio_open_1_0_base",
    version=1,
    model_family="stable_audio",
    description=("Stability AI Stable Audio Open 1.0 text-to-audio. Generates up "
                 "to ~47.5s of stereo 44.1 kHz audio per call. Default duration "
                 "is 10s; raise via `audio_end_in_s` up to the model max."),
    workload_type="t2v",  # NOTE: WorkloadType has no T2A variant yet (REVIEW item 28)
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

# Gated repo `stabilityai/stable-audio-open-small` — same VAE, smaller
# DiT. Registered for users with HF access; load-time state-dict
# mismatch will surface clearly if the DiT sizing differs from the base.
STABLE_AUDIO_OPEN_SMALL = InferencePreset(
    name="stable_audio_open_small",
    version=1,
    model_family="stable_audio",
    description=("Stability AI Stable Audio Open Small. Lower-latency variant of "
                 "Stable Audio Open 1.0 — same Oobleck VAE, smaller/faster DiT. "
                 "Gated repo (`stabilityai/stable-audio-open-small`); request "
                 "access on HF before use."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults=dict(_SHARED_DEFAULTS),
)

ALL_PRESETS = (STABLE_AUDIO_OPEN_1_0_BASE, STABLE_AUDIO_OPEN_SMALL)
