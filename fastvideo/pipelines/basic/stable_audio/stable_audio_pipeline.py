# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 pipeline (T2A + A2A + RePaint inpainting).

Stages:

    InputValidationStage
      → StableAudioConditioningStage      (T5 + NumberConditioner -> cross-attn + global cond, with CFG)
      → StableAudioLatentPreparationStage (initial Gaussian noise; encodes A2A / inpaint refs)
      → StableAudioDenoisingStage         (k-diffusion `dpmpp-3m-sde` over the DiT)
      → StableAudioDecodingStage          (OobleckVAE -> waveform)
"""
from __future__ import annotations

import functools
import os
from typing import Any

import torch

from fastvideo.distributed.parallel_state import get_local_torch_device
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
from fastvideo.utils import resolve_hf_token

logger = init_logger(__name__)

_HF_REPO_ID = "stabilityai/stable-audio-open-1.0"
_OFFICIAL_WEIGHTS_FILE = "model.safetensors"


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

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the components from the single `model.safetensors`. The
        published checkpoint isn't in Diffusers per-subfolder layout, so
        we skip the standard component loader.
        """
        loaded_modules = loaded_modules or {}
        modules: dict[str, Any] = {}

        device = get_local_torch_device()
        precision = getattr(fastvideo_args.pipeline_config, "precision", "fp32")
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(precision, torch.float32)

        from huggingface_hub import hf_hub_download
        local_weights = os.path.join(self.model_path, _OFFICIAL_WEIGHTS_FILE)
        if os.path.isfile(local_weights):
            weights_path = local_weights
        else:
            weights_path = hf_hub_download(repo_id=_HF_REPO_ID,
                                           filename=_OFFICIAL_WEIGHTS_FILE,
                                           token=resolve_hf_token())

        from safetensors.torch import load_file
        logger.info("Loading Stable Audio checkpoint from %s", weights_path)
        full_state = load_file(weights_path)

        _disable_tf32_for_stable_audio()

        if "vae" in loaded_modules:
            modules["vae"] = loaded_modules["vae"]
        else:
            from fastvideo.configs.models.vaes import OobleckVAEConfig
            from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
            cfg = OobleckVAEConfig()
            local_vae_dir = os.path.join(self.model_path, "vae")
            if os.path.isdir(local_vae_dir):
                cfg.pretrained_path = local_vae_dir
                cfg.pretrained_subfolder = None
            else:
                cfg.pretrained_path = _HF_REPO_ID
                cfg.pretrained_subfolder = "vae"
            modules["vae"] = SAAudioVAEModel(cfg)

        if "transformer" in loaded_modules:
            modules["transformer"] = loaded_modules["transformer"]
        else:
            from fastvideo.models.dits.stable_audio import StableAudioDiT
            modules["transformer"] = StableAudioDiT.from_official_state_dict(full_state)
            modules["transformer"] = modules["transformer"].to(device=device, dtype=torch_dtype).eval()

        if "conditioner" in loaded_modules:
            modules["conditioner"] = loaded_modules["conditioner"]
        else:
            from fastvideo.models.encoders.stable_audio_conditioner import (
                StableAudioMultiConditioner, )
            modules["conditioner"] = StableAudioMultiConditioner.from_official_state_dict(full_state)
            modules["conditioner"] = modules["conditioner"].to(device=device).eval()

        return modules

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
