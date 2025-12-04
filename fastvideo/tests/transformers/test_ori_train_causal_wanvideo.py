# SPDX-License-Identifier: Apache-2.0
import os
import math

import numpy as np
import pytest
import torch
from fastvideo.tests.third_party.wan.modules.causal_model import CausalWanModel

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

BASE_MODEL_PATH = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_train_ori_causal_wan_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = CausalWanModel.from_pretrained(
        "/mnt/weka/home/hao.zhang/wei/Self-Forcing/wan_models/Wan2.1-T2V-1.3B", device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)
    causal_state_dict = torch.load("/mnt/weka/home/hao.zhang/wei/Self-Forcing/checkpoints/self_forcing_dmd.pt")["generator_ema"]
    new_state_dict = {}
    for k, v in causal_state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
    causal_state_dict = new_state_dict
    model1.load_state_dict(causal_state_dict)

    model1.num_frame_per_block = 3
    model2.num_frame_per_block = 3

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
    text_seq_len = 30
    seq_len = math.ceil((160 * 90) /
                            (2 * 2) *
                            21)

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                16,
                                21,
                                160,
                                90,
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        text_seq_len + 1,
                                        4096,
                                        device=device,
                                        dtype=precision)

    # Timestep
    timestep = torch.randint(0, 1000, (batch_size, 21), device=device, dtype=torch.long)
    logger.info("timestep: %s", timestep)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    # with torch.amp.autocast('cuda', dtype=precision):
    output1 = model1(
        x=hidden_states,
        context=encoder_hidden_states,
        t=timestep,
        seq_len=seq_len,
    )
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
    assert max_diff < 1e-4, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-4, f"Mean difference between outputs: {mean_diff.item()}"
