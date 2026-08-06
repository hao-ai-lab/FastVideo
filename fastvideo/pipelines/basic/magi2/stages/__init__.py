# SPDX-License-Identifier: Apache-2.0
"""MAGI-2 pipeline components."""

from fastvideo.pipelines.basic.magi2.stages.audio_decoding import (
    MAGI2_AUDIO_TIME_STRETCH,
    Magi2AudioDecodingStage,
    decode_magi2_audio,
    resample_magi2_audio,
)
from fastvideo.pipelines.basic.magi2.stages.conditioning import (
    Magi2ReferenceImageStage,
    Magi2TextEncodingStage,
)
from fastvideo.pipelines.basic.magi2.stages.output import (
    Magi2LatentSavingStage,
    Magi2VideoDecodingStage,
)
from fastvideo.pipelines.basic.magi2.stages.preview_data_proxy import (
    Magi2DataProxy,
    Magi2DataProxyConfig,
    ModelInput,
    Modality,
    VarlenHandler,
)
from fastvideo.pipelines.basic.magi2.stages.refiner_data_proxy import (
    Magi2RefinerDataProxy,
    Magi2RefinerDataProxyConfig,
    RefinerModelInput,
)
from fastvideo.pipelines.basic.magi2.stages.runtime import (
    Magi2InputValidationStage,
    Magi2LatentPreparationStage,
)
from fastvideo.pipelines.basic.magi2.stages.sampling import (
    Magi2PreviewDenoisingStage,
    Magi2RefinerStage,
    ZeroSNRDDPMDiscretization,
)

__all__ = [
    "MAGI2_AUDIO_TIME_STRETCH",
    "Magi2AudioDecodingStage",
    "Magi2DataProxy",
    "Magi2DataProxyConfig",
    "Magi2InputValidationStage",
    "Magi2LatentPreparationStage",
    "Magi2LatentSavingStage",
    "Magi2PreviewDenoisingStage",
    "Magi2ReferenceImageStage",
    "Magi2RefinerStage",
    "Magi2RefinerDataProxy",
    "Magi2RefinerDataProxyConfig",
    "Magi2TextEncodingStage",
    "Magi2VideoDecodingStage",
    "ModelInput",
    "Modality",
    "RefinerModelInput",
    "VarlenHandler",
    "ZeroSNRDDPMDiscretization",
    "decode_magi2_audio",
    "resample_magi2_audio",
]
