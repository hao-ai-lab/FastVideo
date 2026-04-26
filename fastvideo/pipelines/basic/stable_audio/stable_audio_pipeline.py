# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 text-to-audio pipeline.

Composes through FastVideo's stage system:

    InputValidationStage
      → TextEncodingStage (T5-base)
      → StableAudioConditioningStage (projection + duration + CFG batch)
      → StableAudioLatentPreparationStage (random latent + 1D RoPE)
      → StableAudioDenoisingStage (Cosine-DPM++ + CFG)
      → StableAudioDecodingStage (Oobleck VAE → waveform)

**Component status (2026-04-25):**

| Component | Source | First-class status |
|---|---|---|
| Text encoder (T5-base) | `transformers.T5EncoderModel` via FastVideo's `T5Config` | Reused (FastVideo-integrated) |
| Tokenizer | `T5TokenizerFast` from the same HF repo | Reused |
| Projection model | `diffusers.StableAudioProjectionModel` | TODO: port (REVIEW item 30) |
| Scheduler | `diffusers.CosineDPMSolverMultistepScheduler` | TODO: port |
| Transformer (DiT) | `diffusers.StableAudioDiTModel` | TODO: port (REVIEW item 30) |
| VAE (Oobleck) | FastVideo's `OobleckVAE` (lazy-loaded via `SAAudioVAEModel`) | First-class (this PR's predecessor commit) |

`load_modules` is overridden because the diffusers components live
under `transformer/`, `projection_model/`, `scheduler/` subdirs of
`stabilityai/stable-audio-open-1.0` and are loaded via diffusers'
`from_pretrained`, not through FastVideo's component loader.
"""
from __future__ import annotations

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
    StableAudioTextEncodingStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import InputValidationStage

logger = init_logger(__name__)

_HF_REPO_ID = "stabilityai/stable-audio-open-1.0"


def _ensure_hf_token_env() -> None:
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            return


class StableAudioPipeline(ComposedPipelineBase):
    """Stable Audio Open 1.0 T2A pipeline."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "projection_model",
    ]

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """All six components live under one HF repo
        (`stabilityai/stable-audio-open-1.0`). T5 + tokenizer go through
        FastVideo's standard component loader path; the
        diffusers-native pieces (transformer / projection / scheduler)
        are loaded directly via `from_pretrained` since they don't have
        a FastVideo-native equivalent yet. Oobleck VAE is loaded via
        our first-class wrapper.
        """
        _ensure_hf_token_env()
        loaded_modules = loaded_modules or {}

        # Drop the diffusers-loaded keys so super() doesn't fail the
        # required-modules check.
        deferred: list[str] = []
        for key in ("transformer", "scheduler", "projection_model", "vae"):
            if key in self.required_config_modules:
                self.required_config_modules.remove(key)
                deferred.append(key)
        try:
            modules = super().load_modules(fastvideo_args, loaded_modules)
        finally:
            for key in deferred:
                if key not in self.required_config_modules:
                    self.required_config_modules.append(key)

        model_path = self.model_path

        # --- VAE: first-class FastVideo OobleckVAE via the lazy wrapper. ---
        if "vae" in loaded_modules:
            modules["vae"] = loaded_modules["vae"]
        else:
            from fastvideo.configs.models.vaes import OobleckVAEConfig
            from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
            cfg = OobleckVAEConfig()
            # Point the wrapper at the local model_path if it has a
            # `vae/` subfolder (Diffusers convention); else fall back
            # to the upstream HF id.
            local_vae_dir = os.path.join(model_path, "vae")
            if os.path.isdir(local_vae_dir):
                cfg.pretrained_path = local_vae_dir
                cfg.pretrained_subfolder = None
            else:
                cfg.pretrained_path = _HF_REPO_ID
                cfg.pretrained_subfolder = "vae"
            logger.info("Building Oobleck VAE (lazy-load from %s)", cfg.pretrained_path)
            modules["vae"] = SAAudioVAEModel(cfg)

        # --- Diffusers-native components (TODO: port to FastVideo native). ---
        # transformer, scheduler, projection_model
        if "transformer" in loaded_modules:
            modules["transformer"] = loaded_modules["transformer"]
        else:
            from diffusers import StableAudioDiTModel
            logger.info("Loading StableAudioDiTModel from %s/transformer", model_path)
            modules["transformer"] = StableAudioDiTModel.from_pretrained(
                model_path,
                subfolder="transformer",
            )

        if "scheduler" in loaded_modules:
            modules["scheduler"] = loaded_modules["scheduler"]
        else:
            from diffusers import CosineDPMSolverMultistepScheduler
            logger.info("Loading CosineDPMSolverMultistepScheduler from %s/scheduler", model_path)
            modules["scheduler"] = CosineDPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder="scheduler",
            )

        if "projection_model" in loaded_modules:
            modules["projection_model"] = loaded_modules["projection_model"]
        else:
            # StableAudioProjectionModel lives under diffusers.pipelines.stable_audio
            from diffusers.pipelines.stable_audio.modeling_stable_audio import (
                StableAudioProjectionModel, )
            logger.info("Loading StableAudioProjectionModel from %s/projection_model", model_path)
            modules["projection_model"] = StableAudioProjectionModel.from_pretrained(
                model_path,
                subfolder="projection_model",
            )

        # Move the diffusers-loaded weights to the worker's GPU. FastVideo's
        # standard component loader handles this for native components but
        # not for the bypass paths above.
        device = get_local_torch_device()
        precision = getattr(fastvideo_args.pipeline_config, "precision", "fp32")
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }.get(precision, torch.float32)
        for key in ("transformer", "projection_model"):
            mod = modules.get(key)
            if mod is not None and hasattr(mod, "to"):
                modules[key] = mod.to(device=device, dtype=torch_dtype)
                if hasattr(modules[key], "eval"):
                    modules[key].eval()

        return modules

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        pc = fastvideo_args.pipeline_config
        transformer = self.get_module("transformer")
        sample_size = int(getattr(transformer.config, "sample_size", 1024))
        attention_head_dim = int(getattr(transformer.config, "attention_head_dim", 64))
        in_channels = int(getattr(transformer.config, "in_channels", 64))
        # Stable Audio uses head_dim // 2 for the rotary half-rotation.
        rotary_embed_dim = attention_head_dim // 2

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=StableAudioTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=StableAudioConditioningStage(
                projection_model=self.get_module("projection_model"),
                sampling_rate=pc.sampling_rate,
                sample_size=sample_size,
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=StableAudioLatentPreparationStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                sample_size=sample_size,
                num_channels_vae=in_channels,
                rotary_embed_dim=rotary_embed_dim,
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=StableAudioDenoisingStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=StableAudioDecodingStage(vae=self.get_module("vae")),
        )


EntryClass = StableAudioPipeline
