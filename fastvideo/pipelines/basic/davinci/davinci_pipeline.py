# SPDX-License-Identifier: Apache-2.0
"""daVinci-MagiHuman text-to-video inference pipeline."""

import glob
import os
import re
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoModel, AutoTokenizer

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TimestepPreparationStage,
)
from fastvideo.pipelines.stages.davinci_denoising import DaVinciDenoisingStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

logger = init_logger(__name__)

# Path where T5Gemma text encoder weights are stored on the pod.
_T5GEMMA_PATH = "/weights/t5gemma-9b"


class DaVinciMagiHumanPipeline(ComposedPipelineBase):
    """daVinci-MagiHuman T2V pipeline.

    Overrides load_modules to bypass Diffusers-format model_index.json since
    the daVinci checkpoint is raw safetensors shards, not Diffusers format.

    Stage order:
      input_validation → text_encoding → conditioning →
      timestep_preparation → latent_preparation →
      denoising → decoding
    """

    # No Diffusers-format modules to discover — we load everything manually.
    _required_config_modules: list[str] = []

    def _load_config(self, model_path: str) -> dict[str, Any]:
        """Override to skip model_index.json check for non-Diffusers checkpoint."""
        return {"_class_name": "DaVinciMagiHumanPipeline", "_diffusers_version": "0.0.0"}

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load all pipeline components directly (non-Diffusers checkpoint)."""
        modules: dict[str, Any] = {}

        # 1. Scheduler
        modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=5.0,
        )
        logger.info("Loaded scheduler (FlowMatchEulerDiscreteScheduler, shift=5.0)")

        # 2. Tokenizer
        logger.info("Loading tokenizer from %s", _T5GEMMA_PATH)
        modules["tokenizer"] = AutoTokenizer.from_pretrained(
            _T5GEMMA_PATH, local_files_only=True)
        logger.info("Loaded tokenizer")

        # 3. Text encoder (T5Gemma — standard transformers T5 API)
        logger.info("Loading T5Gemma text encoder from %s", _T5GEMMA_PATH)
        precision = fastvideo_args.pipeline_config.text_encoder_precisions[0]
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        _full_model = AutoModel.from_pretrained(
            _T5GEMMA_PATH,
            local_files_only=True,
            torch_dtype=dtype,
        )
        # Use encoder submodule only — AutoModel loads the full seq2seq model
        # (T5GemmaForConditionalGeneration) and the decoder has no decoder
        # input_ids in our pipeline, causing a crash.
        text_encoder = _full_model.encoder.cuda().eval()
        del _full_model  # free decoder weights
        modules["text_encoder"] = text_encoder
        logger.info("Loaded T5Gemma text encoder (%s)", dtype)

        # 4. Transformer (DiT) — load safetensors with param remapping
        logger.info("Loading daVinci DiT from %s", self.model_path)
        from fastvideo.configs.models.dits.davinci_magihuman import (
            DaVinciMagiHumanArchConfig, DaVinciMagiHumanConfig)
        from fastvideo.models.dits.davinci_magihuman import DaVinciMagiHuman

        arch_config = DaVinciMagiHumanArchConfig()
        dit_config = DaVinciMagiHumanConfig(arch_config=arch_config)
        dit_precision = fastvideo_args.pipeline_config.dit_precision
        dit_dtype = torch.bfloat16 if dit_precision == "bf16" else torch.float32

        with torch.no_grad():
            transformer = DaVinciMagiHuman(dit_config, hf_config={})

        shard_files = sorted(
            glob.glob(os.path.join(self.model_path, "*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(
                f"No safetensors found in {self.model_path}")
        logger.info("Loading %d safetensors shards", len(shard_files))

        sd: dict[str, torch.Tensor] = {}
        for f in shard_files:
            sd.update(safetensors_load_file(f, device="cpu"))

        # Apply param_names_mapping (FastVideo loader does this; direct
        # load_state_dict does not — must apply manually).
        param_map: dict[str, str] = getattr(transformer, "param_names_mapping",
                                             {})
        if param_map:
            remapped: dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                new_k = k
                for pattern, replacement in param_map.items():
                    new_k = re.sub(pattern, replacement, new_k)
                remapped[new_k] = v
            sd = remapped

        missing, unexpected = transformer.load_state_dict(sd, strict=False)
        logger.info("DiT load_state_dict: %d missing, %d unexpected",
                    len(missing), len(unexpected))
        if missing:
            logger.warning("Missing keys (first 5): %s", missing[:5])
        transformer = transformer.to(dtype=dit_dtype).cuda().eval()
        modules["transformer"] = transformer
        logger.info("Loaded daVinci DiT (%s)", dit_dtype)

        # 5. VAE — Wan2.2 with z_dim=48.
        #    Weights not available on this pod; stub with correct architecture +
        #    random weights so the decoding stage doesn't crash.
        #    TODO: download Wan2.2-TI2V-5B VAE checkpoint.
        logger.warning(
            "VAE: using random-weight Wan stub (no Wan2.2 checkpoint on pod). "
            "Output video will be noise — denoising correctness can still be verified."
        )
        from fastvideo.models.vaes.wanvae import AutoencoderKLWan
        from fastvideo.configs.models.vaes.davinci_vae import DaVinciVAEConfig

        vae_cfg = DaVinciVAEConfig()
        vae_cfg.load_encoder = False
        vae_cfg.load_decoder = True
        vae_dtype = torch.bfloat16
        with torch.no_grad():
            vae = AutoencoderKLWan(vae_cfg).to(dtype=vae_dtype).cuda().eval()
        modules["vae"] = vae
        logger.info("Loaded VAE stub (random weights, z_dim=48)")

        return modules

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        logger.info("Creating daVinci-MagiHuman pipeline stages...")

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        scheduler = self.get_module("scheduler")

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=scheduler),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=scheduler,
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DaVinciDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=scheduler,
                pipeline=self,
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )

        logger.info("daVinci-MagiHuman pipeline stages created")


# Required by pipeline registry discovery
EntryClass = DaVinciMagiHumanPipeline
