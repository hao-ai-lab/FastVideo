# SPDX-License-Identifier: Apache-2.0
"""Published inference defaults for all official MMAudio variants."""

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


def _mmaudio_preset(
    variant: str,
    description: str,
) -> InferencePreset:
    return InferencePreset(
        name=f"mmaudio_{variant}",
        version=1,
        model_family="mmaudio",
        description=description,
        workload_type="v2a",
        stage_schemas=(_DENOISE_STAGE, ),
        defaults=dict(_DEFAULTS),
    )


MMAUDIO_SMALL_16K = _mmaudio_preset(
    "small_16k",
    ("MMAudio small 16 kHz video-to-audio generation with DFN5B CLIP, "
     "Synchformer, the 16 kHz audio VAE, and the 16 kHz BigVGAN vocoder."),
)
MMAUDIO_SMALL_44K = _mmaudio_preset(
    "small_44k",
    ("MMAudio small 44.1 kHz video-to-audio generation with DFN5B CLIP, "
     "Synchformer, the 44.1 kHz audio VAE, and BigVGAN-v2."),
)
MMAUDIO_MEDIUM_44K = _mmaudio_preset(
    "medium_44k",
    ("MMAudio medium 44.1 kHz video-to-audio generation with DFN5B CLIP, "
     "Synchformer, the 44.1 kHz audio VAE, and BigVGAN-v2."),
)
MMAUDIO_LARGE_44K = _mmaudio_preset(
    "large_44k",
    ("MMAudio large 44.1 kHz video-to-audio generation with DFN5B CLIP, "
     "Synchformer, the 44.1 kHz audio VAE, and BigVGAN-v2."),
)
MMAUDIO_LARGE_44K_V2 = _mmaudio_preset(
    "large_44k_v2",
    ("MMAudio large-44k-v2 video-to-audio generation with DFN5B CLIP, "
     "Synchformer, the 44.1 kHz audio VAE, and BigVGAN-v2."),
)

ALL_PRESETS = (
    MMAUDIO_SMALL_16K,
    MMAUDIO_SMALL_44K,
    MMAUDIO_MEDIUM_44K,
    MMAUDIO_LARGE_44K,
    MMAUDIO_LARGE_44K_V2,
)
