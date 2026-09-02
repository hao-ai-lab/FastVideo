# SPDX-License-Identifier: Apache-2.0
"""Smoke, registry, preset, and stage-contract tests for Cosmos Predict pipeline."""

import json
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock

from fastvideo.api.presets import get_preset
from fastvideo.configs.pipelines.cosmos_predict import CosmosPredictConfig, CosmosPredict14BConfig
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.registry import get_model_info, get_preset_selection
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.basic.cosmos_predict.pipeline_cosmos_predict import (
    CosmosPredictPipeline,
    CosmosPredictLatentPreparationStage,
    EntryClass,
)


def test_cosmos_predict_registry_and_preset_resolution(tmp_path: Path):
    """Verify exact class resolution, required modules, configs, and official preset defaults."""
    assert EntryClass is CosmosPredictPipeline
    assert CosmosPredictPipeline._required_config_modules == [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    # 7B model preset check
    preset_name_7b, family_7b = get_preset_selection("nvidia/Cosmos-1.0-Prompt2World-7B-Video")
    assert (preset_name_7b, family_7b) == ("cosmos_predict_preset", "cosmos_predict")
    preset_7b = get_preset(preset_name_7b, family_7b)
    assert preset_7b.defaults["height"] == 704
    assert preset_7b.defaults["width"] == 1280
    assert preset_7b.defaults["num_frames"] == 93
    assert preset_7b.defaults["fps"] == 24
    assert preset_7b.defaults["guidance_scale"] == 7.0
    assert preset_7b.defaults["num_inference_steps"] == 35

    # 14B model preset check
    preset_name_14b, family_14b = get_preset_selection("nvidia/Cosmos-1.0-Prompt2World-14B-Video")
    assert (preset_name_14b, family_14b) == ("cosmos_predict_14b_preset", "cosmos_predict")
    preset_14b = get_preset(preset_name_14b, family_14b)
    assert preset_14b.defaults["num_frames"] == 93

    # Local layout model info resolution check
    model_dir = tmp_path / "Cosmos-1.0-Prompt2World-7B-Video"
    model_dir.mkdir()
    model_index = {
        "_class_name": "CosmosPredictPipeline",
        "_diffusers_version": "0.32.0",
        "scheduler": ["diffusers", "EDMEulerScheduler"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "transformer": ["diffusers", "CosmosTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLCosmos"],
    }
    for component in CosmosPredictPipeline._required_config_modules:
        (model_dir / component).mkdir()
    (model_dir / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")

    info_7b = get_model_info(str(model_dir), workload_type=WorkloadType.T2V)
    assert info_7b.pipeline_cls is CosmosPredictPipeline
    assert info_7b.pipeline_config_cls is CosmosPredictConfig


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
