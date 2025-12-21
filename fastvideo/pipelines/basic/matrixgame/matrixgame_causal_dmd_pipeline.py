# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game causal DMD pipeline implementation."""

from fastvideo.fastvideo_args import FastVideoArgs
import torch
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, ForwardBatch, LoRAPipeline

from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        MatrixGameImageEncodingStage,
                                        MatrixGameCausalDenoisingStage)
from fastvideo.pipelines.stages.image_encoding import (
    MatrixGameImageVAEEncodingStage)

logger = init_logger(__name__)


class MatrixGameCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = [
        "vae", "transformer", "scheduler", "image_encoder", "image_processor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        if (self.get_module("text_encoder", None) is not None
                and self.get_module("tokenizer", None) is not None):
            self.add_stage(stage_name="prompt_encoding_stage",
                           stage=TextEncodingStage(
                               text_encoders=[self.get_module("text_encoder")],
                               tokenizers=[self.get_module("tokenizer")],
                           ))

        if (self.get_module("image_encoder", None) is not None
                and self.get_module("image_processor", None) is not None):
            self.add_stage(
                stage_name="image_encoding_stage",
                stage=MatrixGameImageEncodingStage(
                    image_encoder=self.get_module("image_encoder"),
                    image_processor=self.get_module("image_processor"),
                ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=MatrixGameImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=MatrixGameCausalDenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           pipeline=self,
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

        logger.info(
            "MatrixGameCausalDMDPipeline initialized with action support")

    @torch.no_grad()
    def streaming_reset(self, batch: ForwardBatch, 
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not self.post_init_called:
            self.post_init()
            
        # 1. Run Pre-processing stages
        stages_to_run = [
            "input_validation_stage", 
            "prompt_encoding_stage", 
            "image_encoding_stage", 
            "conditioning_stage",
            "latent_preparation_stage",
            "image_latent_preparation_stage"
        ]
        
        for stage_name in stages_to_run:
            if stage_name in self._stage_name_mapping:
                batch = self._stage_name_mapping[stage_name].forward(batch, fastvideo_args)
                
        # 2. Reset Denoising Stage
        denoiser = self._stage_name_mapping["denoising_stage"]
        batch = denoiser.reset_streaming(batch, fastvideo_args)
        
        # 3. Decode the generated block
        end_idx = denoiser._streaming_start_index
        if end_idx > 0:
            current_latents = batch.latents[:, :, :end_idx, :, :]
            decoded_frames = self._stage_name_mapping["decoding_stage"].decode(current_latents, fastvideo_args)
            batch.output = decoded_frames
            
        return batch

    def streaming_step(self, keyboard_action, mouse_action) -> ForwardBatch:
        denoiser = self._stage_name_mapping["denoising_stage"]
        
        start_idx = denoiser._streaming_start_index
        batch = denoiser.step_streaming(keyboard_action, mouse_action)
        end_idx = denoiser._streaming_start_index
        
        # Decode only the new generated block
        if end_idx > start_idx:
            current_latents = batch.latents[:, :, start_idx:end_idx, :, :]
            args = denoiser._streaming_fastvideo_args
            decoded_frames = self._stage_name_mapping["decoding_stage"].decode(current_latents, args)
            batch.output = decoded_frames
        else:
            batch.output = None

        return batch
        
    def streaming_clear(self) -> None:
        denoiser = self._stage_name_mapping.get("denoising_stage")
        if denoiser is not None and hasattr(denoiser, "clear_streaming"):
            denoiser.clear_streaming()

EntryClass = [MatrixGameCausalDMDPipeline]
