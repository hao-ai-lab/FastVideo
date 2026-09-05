# SPDX-License-Identifier: Apache-2.0
"""Cosmos Predict pipeline entry (staged pipeline)."""

import torch
from transformers import AutoTokenizer

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    ConditioningStage, 
    Cosmos25AutoDenoisingStage,
    DecodingStage, 
    InputValidationStage,
    Cosmos25TimestepPreparationStage,
    PipelineStage
)
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class CosmosPredictTextEncodingStage(PipelineStage):
    """Cosmos Predict text encoding stage using CosmosPredictTextEncoder."""
    performance_component_metric = "text_encoder_time_s"

    def __init__(self, text_encoder) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = None

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if self.tokenizer is None:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                fastvideo_args.model_paths["tokenizer"],
                subfolder="tokenizer" if "tokenizer" in fastvideo_args.model_paths else None
            )

        assert batch.prompt is not None
        prompts = [batch.prompt] if isinstance(batch.prompt, str) else batch.prompt

        def _encode(texts):
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.text_encoder.model.device)
            
            with set_forward_context(current_timestep=0, attn_metadata=None):
                embeds = self.text_encoder(input_ids)
            return embeds

        prompt_embeds = _encode(prompts)
        batch.prompt_embeds = [prompt_embeds]

        if batch.do_classifier_free_guidance:
            neg = batch.negative_prompt
            neg_prompts = ([neg] * len(prompts)) if isinstance(neg, str) else neg
            neg_embeds = _encode(neg_prompts)
            batch.negative_prompt_embeds = [neg_embeds]
        else:
            batch.negative_prompt_embeds = []

        return batch


class CosmosPredictLatentPreparationStage(PipelineStage):
    """Latent preparation stage for Cosmos Predict."""

    performance_component_metric = "latent_prep_time_s"

    def __init__(self, scheduler, transformer, vae) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else torch.bfloat16
        device = self.transformer.device if hasattr(self.transformer, "device") else torch.device("cuda")
        
        b, c, t, h, w = batch.batch_size, 16, batch.num_frames, batch.height, batch.width
        # Patching size downsampling (CV8x8x8 VAE tokenizer)
        t = (t - 1) // 8 + 1
        h = h // 8
        w = w // 8
        
        shape = (b, c, t, h, w)
        
        # Generator for reproducible noise
        gen = batch.generator
        if isinstance(gen, list) and len(gen) > 0:
            gen = gen[0]
            
        latents = torch.randn(shape, generator=gen, device=device, dtype=target_dtype)

        # Scale by scheduler init noise sigma
        self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
        latents = latents * self.scheduler.init_noise_sigma
        
        batch.latents = [latents]
        
        # For Text-to-Video, Cosmos Predict expects an explicit zero condition_mask of shape (B, 1, T, H, W)
        # and a padding_mask of shape (B, 1, H, W).
        batch.cond_mask = torch.zeros(b, 1, t, h, w, device=device, dtype=target_dtype)
        batch.padding_mask = torch.zeros(b, 1, h, w, device=device, dtype=target_dtype)

        # We must NOT set `batch.conditioning_latents` or `Cosmos25AutoDenoisingStage` will treat this as V2W
        
        return batch


class CosmosPredictPipeline(ComposedPipelineBase):
    """Cosmos Predict video generation pipeline."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        logger.info("Creating Cosmos Predict pipeline stages...")

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=CosmosPredictTextEncodingStage(text_encoder=self.get_module("text_encoder")),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=Cosmos25TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=CosmosPredictLatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                                 transformer=self.get_module("transformer"),
                                                                 vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=Cosmos25AutoDenoisingStage(transformer=self.get_module("transformer"),
                                                        scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))
        logger.info("Cosmos Predict pipeline stages created")


# Entry point for pipeline registry
EntryClass = CosmosPredictPipeline
