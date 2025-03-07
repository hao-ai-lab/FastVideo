"""
HunYuan video diffusion pipeline implementation.

This module contains an implementation of the HunYuan video diffusion pipeline
using the modular pipeline architecture.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.pipelines.composed.text_to_video import TextToVideoPipeline
from fastvideo.pipelines.composed import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    InputValidationStage,
    PromptEncodingStage,
    TimestepPreparationStage,
    LatentPreparationStage,
    ConditioningStage,
    DenoisingStage,
    DecodingStage,
    PostProcessingStage,
)
from fastvideo.pipelines import register_pipeline
from fastvideo.pipelines.stages.prompt_encoding import DualEncoderPromptEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import FlowMatchingTimestepPreparationStage
from fastvideo.inference_args import InferenceArgs
from fastvideo.pipelines.composed.composed_pipeline_base import DiffusionPipelineOutput
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

# class HunyuanLatentPreparationStage(LatentPreparationStage):
#     def _call_implementation(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
#         "custom logic for HunYuan latent preparation"
#         pass


# class hunyuanloader(PipelineLoader):
#     def load_components(self, inference_args: InferenceArgs):
#         pass



@register_pipeline("hunyuan-video")
class HunyuanVideoPipeline(ComposedPipelineBase):
    # def load_components(self, hf_config):
    #     # if we use a different dit class
    #     # or different weight 
    #     pass

    # "text_encoder_2" "encoder2"
    # "transformer" : "dit"

    def setup_pipeline(self, inference_args: InferenceArgs):
        self._stages = []
        self.add_stage(InputValidationStage())
        self.add_stage(DualEncoderPromptEncodingStage())
        self.add_stage(TimestepPreparationStage())
        self.add_stage(LatentPreparationStage())
        self.add_stage(ConditioningStage())
        self.add_stage(DenoisingStage())
        self.add_stage(DecodingStage())


    
    @torch.no_grad()
    def forward(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
        for stage in self._stages:
            batch = stage(batch, inference_args)

        # or 

        # batch = self.input_validation_stage(batch, inference_args)
        # batch = self.prompt_encoding_stage(batch, inference_args)
        # batch = self.timestep_preparation_stage(batch, inference_args)
        # batch = self.latent_preparation_stage(batch, inference_args)
        # batch = self.conditioning_stage(batch, inference_args)
        # batch = self.denoising_stage(batch, inference_args)
        # batch = self.decoding_stage(batch, inference_args)

        return DiffusionPipelineOutput(videos=batch.output) 
