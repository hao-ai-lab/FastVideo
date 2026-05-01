# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 pipeline (T2A + A2A + RePaint inpainting).

Loads from the FastVideo-curated Diffusers-format converted repo
(`FastVideo/stable-audio-open-1.0-Diffusers`), produced by
`scripts/checkpoint_conversion/stable_audio_to_diffusers.py`. Each
component lives in its own subfolder (`transformer/`, `vae/`,
`conditioner/`) and is loaded via the matching `from_pretrained`
classmethod — same per-subfolder layout the standard
`ComposedPipelineBase` loader assumes, just dispatched here because
`conditioner` is not a standard component type in
`fastvideo.models.loader.component_loader`.

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
from fastvideo.utils import maybe_download_model, set_mixed_precision_policy

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

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load each component from its own subfolder of the
        Diffusers-format converted repo (resolves a HF id to a local
        snapshot first).
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

        # Tell `fastvideo.attention.layer` what dtype the model will run
        # in — without this the attention layer reads
        # `torch.get_default_dtype()` (fp32) and rejects FlashAttention.
        # The standard FSDP loader does this in `fsdp_load.py:89`; we
        # do it here too because we bypass that loader.
        set_mixed_precision_policy(param_dtype=torch_dtype, reduce_dtype=torch_dtype)

        _disable_tf32_for_stable_audio()

        # `self.model_path` may be a HF repo id; resolve to a local dir.
        local_root = (self.model_path if os.path.isdir(self.model_path) else maybe_download_model(self.model_path))
        logger.info("Loading Stable Audio components from %s", local_root)

        if "vae" in loaded_modules:
            modules["vae"] = loaded_modules["vae"]
        else:
            from fastvideo.configs.models.vaes import OobleckVAEConfig
            from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
            cfg = OobleckVAEConfig()
            cfg.pretrained_path = os.path.join(local_root, "vae")
            cfg.pretrained_subfolder = None
            modules["vae"] = SAAudioVAEModel(cfg)

        if "transformer" in loaded_modules:
            modules["transformer"] = loaded_modules["transformer"]
        else:
            from fastvideo.models.dits.stable_audio import StableAudioDiT
            modules["transformer"] = StableAudioDiT.from_pretrained(os.path.join(local_root, "transformer"))
            modules["transformer"] = modules["transformer"].to(device=device, dtype=torch_dtype).eval()

        if "conditioner" in loaded_modules:
            modules["conditioner"] = loaded_modules["conditioner"]
        else:
            from fastvideo.models.encoders.stable_audio_conditioner import (
                StableAudioMultiConditioner, )
            modules["conditioner"] = StableAudioMultiConditioner.from_pretrained(os.path.join(
                local_root, "conditioner"))
            modules["conditioner"] = modules["conditioner"].to(device=device, dtype=torch_dtype).eval()

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
