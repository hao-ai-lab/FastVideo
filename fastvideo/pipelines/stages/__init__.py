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
from fastvideo.pipelines.stages.denoising import (Cosmos25DenoisingStage,
                                                  CosmosDenoisingStage,
                                                  DenoisingStage,
                                                  DmdDenoisingStage)
from fastvideo.pipelines.stages.encoding import EncodingStage
from fastvideo.pipelines.stages.image_encoding import (
    ImageEncodingStage, MatrixGameImageEncodingStage, RefImageEncodingStage,
    ImageVAEEncodingStage, VideoVAEEncodingStage, Hy15ImageEncodingStage)
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.latent_preparation import (
    Cosmos25LatentPreparationStage, CosmosLatentPreparationStage,
    LatentPreparationStage)
from fastvideo.pipelines.stages.ltx2_audio_decoding import LTX2AudioDecodingStage
from fastvideo.pipelines.stages.ltx2_denoising import LTX2DenoisingStage
from fastvideo.pipelines.stages.ltx2_latent_preparation import (
    LTX2LatentPreparationStage)
from fastvideo.pipelines.stages.matrixgame_denoising import (
    MatrixGameCausalDenoisingStage)
from fastvideo.pipelines.stages.stepvideo_encoding import (
    StepvideoPromptEncodingStage)
from fastvideo.pipelines.stages.text_encoding import (Cosmos25TextEncodingStage,
                                                      TextEncodingStage)
from fastvideo.pipelines.stages.timestep_preparation import (
    Cosmos25TimestepPreparationStage, TimestepPreparationStage)

# LongCat stages
from fastvideo.pipelines.stages.longcat_video_vae_encoding import LongCatVideoVAEEncodingStage
from fastvideo.pipelines.stages.longcat_kv_cache_init import LongCatKVCacheInitStage
from fastvideo.pipelines.stages.longcat_vc_denoising import LongCatVCDenoisingStage

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "Cosmos25TimestepPreparationStage",
    "LatentPreparationStage",
    "CosmosLatentPreparationStage",
    "Cosmos25LatentPreparationStage",
    "LTX2LatentPreparationStage",
    "LTX2AudioDecodingStage",
    "ConditioningStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "CausalDMDDenosingStage",
    "MatrixGameCausalDenoisingStage",
    "CosmosDenoisingStage",
    "Cosmos25DenoisingStage",
    "LTX2DenoisingStage",
    "EncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "MatrixGameImageEncodingStage",
    "Hy15ImageEncodingStage",
    "RefImageEncodingStage",
    "ImageVAEEncodingStage",
    "VideoVAEEncodingStage",
    "TextEncodingStage",
    "Cosmos25TextEncodingStage",
    "StepvideoPromptEncodingStage",
    # LongCat stages
    "LongCatVideoVAEEncodingStage",
    "LongCatKVCacheInitStage",
    "LongCatVCDenoisingStage",
]
