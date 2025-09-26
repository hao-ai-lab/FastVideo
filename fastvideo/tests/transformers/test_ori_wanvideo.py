# SPDX-License-Identifier: Apache-2.0
import os
import math

import numpy as np
import pytest
import torch
from fastvideo.wan.modules.model import WanModel

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

os.environ["FASTVIDEO_FORCE_ATTN_BF16"] = "1"
BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_ori_wan_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.float32
    precision_str = "fp32"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str, dit_forward_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args)

    model1 = WanModel.from_pretrained(
        "/mnt/weka/home/hao.zhang/wei/Self-Forcing-clean/wan_models/Wan2.1-T2V-1.3B")
    model1.eval()
    model1 = model1.to(device).to(precision)
    model1.requires_grad_(False)

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
    text_seq_len = 120
    seq_len = math.ceil((104 * 60) /
                            (2 * 2) *
                            21)

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                21,
                                16,
                                60,
                                104,
                                device=device,
                                generator=torch.Generator("cuda").manual_seed(1024),
                                dtype=precision)

    logger.info("Hidden states sum: %s, Hidden states shape: %s, Hidden states dtype: %s", hidden_states.float().sum(), hidden_states.shape, hidden_states.dtype)

    # Text embeddings [B, L, D] (including global token)
    # encoder_hidden_states = torch.randn(batch_size,
    #                                     text_seq_len + 1,
    #                                     4096,
    #                                     device=device,
    #                                     dtype=precision)
    encoder_hidden_states = torch.load("../sf_cond_prompt_embeds.pt").to(device, dtype=precision)

    logger.info("Encoder hidden states sum: %s, Encoder hidden states shape: %s, Encoder hidden states dtype: %s", encoder_hidden_states.float().sum(), encoder_hidden_states.shape, encoder_hidden_states.dtype)

    # Timestep
    timestep = torch.tensor([995.7627], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    # with torch.amp.autocast('cuda', dtype=precision):
    output1 = model1(
        x=hidden_states.permute(0, 2, 1, 3, 4),
        context=encoder_hidden_states,
        t=timestep,
        seq_len=seq_len,
    ).permute(0, 2, 1, 3, 4)
    with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
    ):
        output2 = model2(hidden_states=hidden_states.permute(0, 2, 1, 3, 4),
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=timestep).permute(0, 2, 1, 3, 4)

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    logger.info("Output 1 sum: %s", output1.float().sum())
    logger.info("Output 2 sum: %s", output2.float().sum())
    assert max_diff < 1e-4, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-4, f"Mean difference between outputs: {mean_diff.item()}"