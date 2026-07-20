# SPDX-License-Identifier: Apache-2.0
"""Published MMAudio large-44k-v2 inference defaults."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="MMAudio forward-time Euler flow with multimodal CFG.",
    allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
)

_DEFAULTS = {
    "seed": 42,
    "guidance_scale": 4.5,
    "num_inference_steps": 25,
    "negative_prompt": "",
    "audio_start_in_s": 0.0,
    "audio_end_in_s": 8.0,
    # The shared generator still validates these fields, but MMAudio returns
    # audio metadata rather than materializing the placeholder pixels.
    "height": 8,
    "width": 8,
    "num_frames": 1,
    "fps": 25,
    "return_frames": False,
}

MMAUDIO_LARGE_44K_V2 = InferencePreset(
    name="mmaudio_large_44k_v2",
    version=1,
    model_family="mmaudio",
    description=(
        "MMAudio large-44k-v2 video-to-audio generation with DFN5B CLIP, "
        "Synchformer, a 44.1 kHz audio VAE, and BigVGAN-v2."
    ),
    workload_type="v2a",
    stage_schemas=(_DENOISE_STAGE,),
    defaults=dict(_DEFAULTS),
)

ALL_PRESETS = (MMAUDIO_LARGE_44K_V2,)

