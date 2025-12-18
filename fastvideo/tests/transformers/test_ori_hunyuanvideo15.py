# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from fastvideo.hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import HunyuanVideo15Config
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")

@pytest.mark.usefixtures("distributed_setup")
def test_hunyuanvideo15_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=HunyuanVideo15Config(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        "/mnt/fast-disks/hao_lab/wei/hy1.5_models/transformer/480p_t2v", device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)
    # model1 = HunyuanVideo15Transformer3DModel.from_pretrained(
    #     TRANSFORMER_PATH, device=device,
    #     torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

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
    seq_len_2 = 70

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                65,
                                9,
                                120,
                                30,
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len + 1,
                                        3584,
                                        device=device,
                                        dtype=precision)
    # Create attention mask for encoder_hidden_states
    encoder_attention_mask = torch.ones(batch_size,
                                        seq_len + 1,
                                        device=device,
                                        dtype=torch.bool)
    encoder_hidden_states[:, 15:] = 0
    encoder_attention_mask[:, 15:] = False

    encoder_hidden_states_2 = torch.randn(batch_size,
                                        seq_len_2 + 1,
                                        1472,
                                        device=device,
                                        dtype=precision)
    encoder_attention_mask_2 = torch.ones(batch_size,
                                        seq_len_2 + 1,
                                        device=device,
                                        dtype=torch.bool)
    encoder_hidden_states_2[:, 39:] = 0
    encoder_attention_mask_2[:, 39:] = False

    image_embeds = torch.zeros(batch_size, 729, 1152, dtype=precision, device=device)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    extra_kwargs = {
        "byt5_text_states": encoder_hidden_states_2,
        "byt5_text_mask": encoder_attention_mask_2
    }

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=precision):
            output1 = model1(
                hidden_states=hidden_states,
                text_states=encoder_hidden_states,
                text_states_2=None,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                vision_states=image_embeds,
                return_dict=False,
                extra_kwargs=extra_kwargs,
            )[0]
            with set_forward_context(
                    current_timestep=0,
                    attn_metadata=None,
                    forward_batch=forward_batch,
            ):
                output2 = model2(hidden_states=hidden_states,
                                encoder_hidden_states=[encoder_hidden_states, encoder_hidden_states_2],
                                encoder_attention_mask=[encoder_attention_mask, encoder_attention_mask_2],
                                encoder_hidden_states_image=[image_embeds],
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