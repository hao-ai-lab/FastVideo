# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from diffusers import FluxTransformer2DModel

from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import TransformerLoader
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.configs.models.dits import FluxImageConfig

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

# BASE_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
BASE_MODEL_PATH = "/home/test/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/"

# MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
#                                   local_dir=os.path.join(
#                                       'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(BASE_MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_flux_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         use_cpu_offload=False,
                         precision=precision_str)
    args.device = device
    args.dit_config = FluxImageConfig()

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, "", args).to(device, dtype=precision)

    model1 = FluxTransformer2DModel.from_pretrained(
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

    def pack_input(hidden_states):
        b, c, _, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, h // 2, 2, w // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(b, (h // 2) * (w // 2), c * 4)
        return hidden_states

    # Video latents [B, C, T, H, W]
    hidden_states_2 = torch.randn(batch_size,
                                 16,
                                 1,
                                 128,
                                 128,
                                 device=device, dtype=precision)
    hidden_states_1 = pack_input(hidden_states_2)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len,
                                        4096,
                                        device=device,
                                        dtype=precision)
    pooled_text_embeds = torch.randn(batch_size,
                                    768,
                                    device=device,
                                    dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)
    guidance = torch.full([1], 3.5, device=device, dtype=torch.float32)

    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    with torch.amp.autocast('cuda', dtype=precision):
        output1 = model1(
            hidden_states=hidden_states_1,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_text_embeds,
            timestep=timestep,
            txt_ids=torch.zeros(encoder_hidden_states.shape[1], 3).to(device=device, dtype=precision),
            img_ids=_prepare_latent_image_ids(batch_size, 64, 64, device, precision),
            guidance=guidance,
            return_dict=False,
        )[0]
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
        ):
            output2 = model2(hidden_states=hidden_states_2,
                             encoder_hidden_states=[pooled_text_embeds,
                                                   encoder_hidden_states],
                             guidance=guidance,
                             timestep=timestep)

    def unpack_output(output):
        output = output.view(1, 64, 64, 16, 2, 2)
        output = output.permute(0, 3, 1, 4, 2, 5)
        output = output.reshape(1, 16, 1, 64 * 2, 64 * 2)
        return output

    # output1 = unpack_output(output1)
    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-1, f"Mean difference between outputs: {mean_diff.item()}"