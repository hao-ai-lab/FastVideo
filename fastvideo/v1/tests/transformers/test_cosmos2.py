# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from diffusers import CosmosTransformer3DModel

from fastvideo.v1.configs.pipelines import PipelineConfig
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import TransformerLoader
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.configs.models.dits import CosmosConfig
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29504"

BASE_MODEL_PATH = "nvidia/Cosmos-Predict2-2B-Text2Image"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_cosmos2_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         use_cpu_offload=False,
                         pipeline_config=PipelineConfig(dit_config=CosmosConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, "", args).to(device, dtype=precision)

    model1 = CosmosTransformer3DModel.from_pretrained(
        TRANSFORMER_PATH, device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info("Model 2 weight sum: %s", weight_sum_model2)
    logger.info("Model 2 weight mean: %s", weight_mean_model2)

    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    logger.info("Weight sum difference: %s", weight_sum_diff)
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info("Weight mean difference: %s", weight_mean_diff)

    # Set both models to eval mode
    model1 = model1.eval()
    model2 = model2.eval()

    # Create identical inputs for both models
    batch_size = 1
    seq_len = 30

    # Video latents [B, C, T, H, W] - Cosmos2 specific dimensions
    hidden_states = torch.randn(batch_size,
                                16,
                                1,  # Single frame for image generation
                                32,  # Height patches
                                32,  # Width patches
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D] - Cosmos2 uses T5 embeddings with 1024 dim
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len,
                                        1024,  # T5 embedding dimension
                                        device=device,
                                        dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    # padding mask
    padding_mask = hidden_states.new_zeros(1, 1, 32, 32, device=device, dtype=precision)
    # print(padding_mask.shape)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    with torch.amp.autocast('cuda', dtype=precision):
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            output2 = model2(hidden_states=hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             timestep=timestep,
                             padding_mask=padding_mask)

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-2, f"Mean difference between outputs: {mean_diff.item()}"


@pytest.mark.usefixtures("distributed_setup")
def test_cosmos2_transformer_video2world():
    """Test Cosmos2 Video2World variant"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    
    # Use Video2World model path
    base_model_path = "nvidia/Cosmos-Predict2-2B-Video2World"
    model_path = maybe_download_model(base_model_path,
                                      local_dir=os.path.join(
                                          'data', base_model_path))
    transformer_path = os.path.join(model_path, "transformer")
    
    args = FastVideoArgs(model_path=transformer_path,
                         use_cpu_offload=False,
                         pipeline_config=PipelineConfig(dit_config=CosmosConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(transformer_path, "", args).to(device, dtype=precision)

    model1 = CosmosTransformer3DModel.from_pretrained(
        transformer_path, device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

    # Set both models to eval mode
    model1 = model1.eval()
    model2 = model2.eval()

    # Create identical inputs for both models
    batch_size = 1
    seq_len = 30

    # Video latents [B, C, T, H, W] - Video2World has additional condition channel
    hidden_states = torch.randn(batch_size,
                                17,  # 16 + 1 for condition channel
                                8,   # Multiple frames for video
                                32,  # Height patches
                                32,  # Width patches
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D]
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len,
                                        1024,  # T5 embedding dimension
                                        device=device,
                                        dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    with torch.amp.autocast('cuda', dtype=precision):
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            output2 = model2(hidden_states=hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             timestep=timestep)

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-2, f"Mean difference between outputs: {mean_diff.item()}"
