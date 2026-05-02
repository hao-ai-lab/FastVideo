# SPDX-License-Identifier: Apache-2.0
from fastvideo.pipelines.basic.magi_human.stages.audio_decoding import MagiHumanAudioDecodingStage
from fastvideo.pipelines.basic.magi_human.stages.denoising import MagiHumanDenoisingStage
from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import MagiHumanLatentPreparationStage
from fastvideo.pipelines.basic.magi_human.stages.reference_image import MagiHumanReferenceImageStage

__all__ = [
    "MagiHumanAudioDecodingStage",
    "MagiHumanDenoisingStage",
    "MagiHumanLatentPreparationStage",
    "MagiHumanReferenceImageStage",
]
