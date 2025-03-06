"""
Text-to-video pipeline implementation.

This module contains a standard text-to-video pipeline implementation that can be
used as a base for more specific implementations.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.pipelines.composed.composed_pipeline_base import ComposedPipelineBase
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
# Import from the main pipelines package to avoid circular imports
from fastvideo.pipelines import register_pipeline


@register_pipeline("text-to-video")
class TextToVideoPipeline(ComposedPipelineBase):
    """
    Standard text-to-video pipeline.
    
    This pipeline implements a standard text-to-video diffusion process with the
    following stages:
    1. Input validation
    2. Prompt encoding
    3. Timestep preparation
    4. Latent preparation
    5. Conditioning (classifier-free guidance)
    6. Denoising
    7. Decoding
    8. Post-processing
    
    This pipeline supports models with either a single text encoder or dual text encoders.
    Different model architectures may use secondary encoders for various purposes such as
    additional conditioning information, temporal control, stylistic aspects, or other
    model-specific features.
    """
    
    is_video_pipeline = True
    
    def __init__(
        self,
        input_validation_stage: InputValidationStage,
        prompt_encoding_stage: PromptEncodingStage,
        timestep_preparation_stage: TimestepPreparationStage,
        latent_preparation_stage: LatentPreparationStage,
        conditioning_stage: ConditioningStage,
        denoising_stage: DenoisingStage,
        decoding_stage: DecodingStage,
        post_processing_stage: Optional[PostProcessingStage] = None,
        secondary_prompt_encoding_stage: Optional[PromptEncodingStage] = None,
    ):
        """
        Initialize the text-to-video pipeline.
        
        Args:
            input_validation_stage: The input validation stage.
            prompt_encoding_stage: The primary prompt encoding stage.
            timestep_preparation_stage: The timestep preparation stage.
            latent_preparation_stage: The latent preparation stage.
            conditioning_stage: The conditioning stage.
            denoising_stage: The denoising stage.
            decoding_stage: The decoding stage.
            post_processing_stage: The post-processing stage (optional).
            secondary_prompt_encoding_stage: Optional secondary prompt encoding stage for models
                with dual text encoders. The purpose of this encoder varies by model architecture
                and may handle additional conditioning aspects such as temporal control, style,
                or other model-specific features.
        """
        super().__init__()
        
        # Add the stages in order
        self.add_stage(input_validation_stage)
        self.add_stage(prompt_encoding_stage)
        
        # Add optional secondary prompt encoding stage if provided
        if secondary_prompt_encoding_stage is not None:
            self.add_stage(secondary_prompt_encoding_stage)
            
        self.add_stage(timestep_preparation_stage)
        self.add_stage(latent_preparation_stage)
        self.add_stage(conditioning_stage)
        self.add_stage(denoising_stage)
        self.add_stage(decoding_stage)
        
        if post_processing_stage is not None:
            self.add_stage(post_processing_stage)
    
    @property
    def components(self) -> Dict[str, Any]:
        """Get all components used by this pipeline."""
        components = {}
        
        # Add components from each stage
        for stage in self._stages:
            for attr_name in dir(stage):
                if attr_name.startswith("__") or attr_name.startswith("_"):
                    continue
                
                attr = getattr(stage, attr_name)
                if isinstance(attr, torch.nn.Module):
                    components[attr_name] = attr
        
        return components 