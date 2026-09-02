# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from unittest.mock import MagicMock

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.registry import get_model_family, get_default_preset
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.basic.cosmos_predict.pipeline_cosmos_predict import (
    CosmosPredictPipeline,
    CosmosPredictLatentPreparationStage,
)


def test_cosmos_predict_registry_and_preset_resolution():
    """Verify that presets are properly registered and resolvable via SamplingParam."""
    # 7B model check
    param_7b = SamplingParam.from_pretrained("nvidia/Cosmos-1.0-Prompt2World-7B-Video")
    assert param_7b.height == 704
    assert param_7b.width == 1280
    assert param_7b.num_frames == 93
    assert param_7b.num_inference_steps == 35
    assert get_model_family("nvidia/Cosmos-1.0-Prompt2World-7B-Video") == "cosmos_predict"
    assert get_default_preset("nvidia/Cosmos-1.0-Prompt2World-7B-Video") == "cosmos_predict_preset"

    # 14B model check
    param_14b = SamplingParam.from_pretrained("nvidia/Cosmos-1.0-Prompt2World-14B-Video")
    assert param_14b.height == 704
    assert param_14b.width == 1280
    assert param_14b.num_frames == 93
    assert get_model_family("nvidia/Cosmos-1.0-Prompt2World-14B-Video") == "cosmos_predict"
    assert get_default_preset("nvidia/Cosmos-1.0-Prompt2World-14B-Video") == "cosmos_predict_14b_preset"


def test_cosmos_predict_latent_preparation_temporal_downsampling():
    """Verify that temporal compression ratio (CV8x8x8) downsamples t, h, w correctly."""
    mock_scheduler = MagicMock()
    mock_scheduler.init_noise_sigma = 1.0
    mock_transformer = MagicMock()
    mock_transformer.dtype = torch.float32
    mock_transformer.device = torch.device("cpu")
    mock_vae = MagicMock()

    stage = CosmosPredictLatentPreparationStage(
        scheduler=mock_scheduler,
        transformer=mock_transformer,
        vae=mock_vae,
    )

    batch = ForwardBatch(
        data_type="video",
        batch_size=1,
        num_frames=93,
        height=704,
        width=1280,
        num_inference_steps=35,
        generator=None,
    )
    args = FastVideoArgs(model_path="nvidia/Cosmos-1.0-Prompt2World-7B-Video")

    output_batch = stage.forward(batch, args)

    # Expected latent dimensions:
    # t = (93 - 1) // 8 + 1 = 12
    # h = 704 // 8 = 88
    # w = 1280 // 8 = 160
    assert len(output_batch.latents) == 1
    latents = output_batch.latents[0]
    assert latents.shape == (1, 16, 12, 88, 160)
    assert output_batch.cond_mask.shape == (1, 1, 12, 88, 160)
    assert output_batch.padding_mask.shape == (1, 1, 88, 160)
