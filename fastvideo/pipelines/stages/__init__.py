# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.causal_denoising import CausalDMDDenosingStage
from fastvideo.pipelines.stages.conditioning import ConditioningStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.denoising import (CosmosDenoisingStage,
                                                  DenoisingStage,
                                                  DmdDenoisingStage)
from fastvideo.pipelines.stages.encoding import EncodingStage
from fastvideo.pipelines.stages.image_encoding import (ImageEncodingStage,
                                                       RefImageEncodingStage,
                                                       ImageVAEEncodingStage,
                                                       VideoVAEEncodingStage)
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.latent_preparation import (
    CosmosLatentPreparationStage, LatentPreparationStage)
from fastvideo.pipelines.stages.stepvideo_encoding import (
    StepvideoPromptEncodingStage)
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import (
    TimestepPreparationStage)

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "LatentPreparationStage",
    "CosmosLatentPreparationStage",
    "ConditioningStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "CausalDMDDenosingStage",
    "CosmosDenoisingStage",
    "EncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "RefImageEncodingStage",
    "ImageVAEEncodingStage",
    "VideoVAEEncodingStage",
    "TextEncodingStage",
    "StepvideoPromptEncodingStage",
]
