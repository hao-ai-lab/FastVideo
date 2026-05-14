# SPDX-License-Identifier: Apache-2.0
"""Latent-slice regression test for Stable Audio Open 1.0 text-to-audio.

Companion to ``test_ltx2_similarity.py`` — applies the same latent
cosine-distance philosophy to 3-D audio latents ``[B, 64, T_latent]``.

Why latent-space and not waveform-space SSIM:
- ``dpmpp-3m-sde`` (k-diffusion) accumulates per-step bf16 noise; the
  Oobleck VAE then magnifies any residual drift into the time-domain
  waveform. A few mis-rounded accumulators drive sample-wise diff well
  past audible thresholds without indicating a real regression.
- Diffusers' own
  ``tests/pipelines/stable_audio/test_stable_audio.py`` compares
  decoded audio samples via
  ``np.abs(expected - actual).max() < 1.5e-3``; that bound holds for
  CPU dummy components but does not survive cross-architecture bf16
  on our CI pool (L40S/A40/H100/B200).
- Comparing the **pre-VAE latent** moves the assertion upstream of
  the dominant noise source.

Slice spec: ``audio_first_8_timesteps`` returns
``latent[0, :, :8]`` (= 64 channels × 8 latent timesteps = 512
elements). The full latent ``[1, 64, 1024]`` for SA-1.0 is also
compared via cosine distance.
"""

import os

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
)
from fastvideo.tests.ssim.latent_similarity_utils import (
    AUDIO_FIRST_8_TIMESTEPS_SPEC,
    run_text_to_latent_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

STABLE_AUDIO_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/stable-audio-open-1.0-Diffusers",
    # 8/8/1 are the SA-1.0 SamplingParam defaults; the values are
    # placeholders unused by the audio pipeline but must satisfy the
    # shared InputValidationStage divisible-by-8 check.
    "height": 8,
    "width": 8,
    "num_frames": 1,
    "num_inference_steps": 25,
    "guidance_scale": 7.0,
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 24,
}

_SA_FULL_DEFAULTS = SamplingParam.from_pretrained(STABLE_AUDIO_PARAMS["model_path"])
STABLE_AUDIO_FULL_QUALITY_PARAMS = {
    **STABLE_AUDIO_PARAMS,
    "num_inference_steps": _SA_FULL_DEFAULTS.num_inference_steps,
    "guidance_scale": _SA_FULL_DEFAULTS.guidance_scale,
    "seed": _SA_FULL_DEFAULTS.seed,
}

STABLE_AUDIO_MODEL_TO_PARAMS = {
    "stable-audio-open-1.0-Diffusers": STABLE_AUDIO_PARAMS,
}
FULL_QUALITY_STABLE_AUDIO_MODEL_TO_PARAMS = {
    "stable-audio-open-1.0-Diffusers": STABLE_AUDIO_FULL_QUALITY_PARAMS,
}

STABLE_AUDIO_TEST_PROMPTS = [
    "Lo-fi hip hop instrumental with vinyl crackle and gentle piano.",
]

SLICE_COSINE_DISTANCE_THRESHOLD = 5e-3
FULL_COSINE_DISTANCE_THRESHOLD = 1e-2


@pytest.mark.parametrize("prompt", STABLE_AUDIO_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(STABLE_AUDIO_MODEL_TO_PARAMS.keys()))
def test_stable_audio_inference_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    run_text_to_latent_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=STABLE_AUDIO_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_STABLE_AUDIO_MODEL_TO_PARAMS,
        slice_cosine_threshold=SLICE_COSINE_DISTANCE_THRESHOLD,
        full_cosine_threshold=FULL_COSINE_DISTANCE_THRESHOLD,
        slice_spec=AUDIO_FIRST_8_TIMESTEPS_SPEC,
        # FSDP + @torch.inference_mode in StableAudioDenoisingStage hits
        # "Inference tensors do not track version counter" on single-GPU
        # unshard. SA-1.0 fits on one B200 anyway.
        init_kwargs_override={"use_fsdp_inference": False},
    )
