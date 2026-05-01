# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 pipeline (T2A + A2A + RePaint inpainting).

Components are loaded via the standard
`ComposedPipelineBase.load_modules` against the FastVideo-curated
Diffusers-format repo `FastVideo/stable-audio-open-1.0-Diffusers`
(produced by
`scripts/checkpoint_conversion/stable_audio_to_diffusers.py`). The DiT
is a `BaseDiT` subclass loaded by `TransformerLoader`; the VAE is
loaded by `VAELoader`; the multi-conditioner (T5 + NumberConditioners)
is loaded by `ConditionerLoader` (a Stable Audio-specific addition).

Stages:

    InputValidationStage
      → StableAudioConditioningStage      (T5 + NumberConditioner -> cross-attn + global cond, with CFG)
      → StableAudioLatentPreparationStage (initial Gaussian noise; encodes A2A / inpaint refs)
      → StableAudioDenoisingStage         (k-diffusion `dpmpp-3m-sde` over the DiT)
      → StableAudioDecodingStage          (OobleckVAE -> waveform)
"""
from __future__ import annotations

import functools

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.stable_audio.stages import (
    StableAudioConditioningStage,
    StableAudioDecodingStage,
    StableAudioDenoisingStage,
    StableAudioLatentPreparationStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import InputValidationStage

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def _warn_tf32_disabled_for_stable_audio() -> None:
    logger.warning("Stable Audio pipeline is disabling process-global "
                   "torch.backends.{cuda.matmul.allow_tf32, cudnn.allow_tf32, "
                   "cuda.matmul.allow_fp16_reduced_precision_reduction, "
                   "cudnn.benchmark} for A2A renoise determinism. Other models "
                   "loaded into this process will inherit these settings.")


def _disable_tf32_for_stable_audio() -> None:
    """Disable TF32 / cuDNN nondeterminism — A2A renoise-then-denoise SDE
    amplifies per-element drift, and the published parity bounds were
    set with these off. Process-global; the first call logs a warning.
    """
    _warn_tf32_disabled_for_stable_audio()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False


class StableAudioPipeline(ComposedPipelineBase):
    """Stable Audio Open 1.0 pipeline.

    Mode is kwargs-driven on `generate_video()`:

      * Text-to-audio (default) -- `prompt=...`, `audio_end_in_s=...`
      * Audio-to-audio variation -- add `init_audio=ref` (and optionally
        `init_noise_level`, lower = closer to reference)
      * RePaint inpainting / outpainting -- add `inpaint_audio=ref` and
        `inpaint_mask` (1-D, 1 = keep / 0 = regenerate)

    See `examples/inference/basic/basic_stable_audio*.py` for runnable
    examples of each mode.
    """

    _required_config_modules = [
        "vae",
        "transformer",
        "conditioner",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        """Apply Stable Audio's process-global numerics overrides BEFORE
        the standard component loaders run (TF32 off for A2A renoise
        determinism)."""
        _disable_tf32_for_stable_audio()

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        pc = fastvideo_args.pipeline_config

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(
            stage_name="conditioning_stage",
            stage=StableAudioConditioningStage(conditioner=self.get_module("conditioner")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=StableAudioLatentPreparationStage(
                io_channels=64,
                # Fixed-size latent (~47.5s @ 44.1 kHz); requested duration
                # is sliced after decode.
                sample_size=2097152,
                vae=self.get_module("vae"),
                sample_rate=pc.sampling_rate,
                audio_channels=pc.audio_channels,
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=StableAudioDenoisingStage(transformer=self.get_module("transformer")),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=StableAudioDecodingStage(vae=self.get_module("vae")),
        )


EntryClass = StableAudioPipeline
