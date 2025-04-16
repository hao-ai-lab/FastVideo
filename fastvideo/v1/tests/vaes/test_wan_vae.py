# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLWan

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import VAELoader
from fastvideo.v1.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
VAE_PATH = os.path.join(MODEL_PATH, "vae")


@pytest.mark.usefixtures("distributed_setup")
def test_wan_vae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=VAE_PATH, vae_precision=precision_str)
    args.device = device

    loader = VAELoader()
    model2 = loader.load(VAE_PATH, "", args)

    model1 = AutoencoderKLWan.from_pretrained(
        VAE_PATH, torch_dtype=precision).to(device).eval()

    # Create identical inputs for both models
    batch_size = 1

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(batch_size,
                               3,
                               81,
                               32,
                               32,
                               device=device,
                               dtype=precision)
    latent_tensor = torch.randn(batch_size,
                                16,
                                21,
                                32,
                                32,
                                device=device,
                                dtype=precision)

    # Disable gradients for inference
    with torch.no_grad():
        # Test encoding
        logger.info("Testing encoding...")
        latent1 = model1.encode(input_tensor).latent_dist.mean
        print("--------------------------------")
        latent2 = model2.encode(input_tensor).mean
        # Check if latents have the same shape
        assert latent1.shape == latent2.shape, f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        assert latent1.shape == latent2.shape, f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        # Check if latents are similar
        max_diff_encode = torch.max(torch.abs(latent1 - latent2))
        mean_diff_encode = torch.mean(torch.abs(latent1 - latent2))
        logger.info("Maximum difference between encoded latents: %s",
                    max_diff_encode.item())
        logger.info("Mean difference between encoded latents: %s",
                    mean_diff_encode.item())
        assert mean_diff_encode < 5e-1, f"Encoded latents differ significantly: mean diff = {mean_diff_encode.item()}"
        # Test decoding
        logger.info("Testing decoding...")
        latents_mean = (torch.tensor(model1.config.latents_mean).view(
            1, model1.config.z_dim, 1, 1, 1).to(latent_tensor.device,
                                                latent_tensor.dtype))
        latents_std = 1.0 / torch.tensor(model1.config.latents_std).view(
            1, model1.config.z_dim, 1, 1, 1).to(latent_tensor.device,
                                                latent_tensor.dtype)
        latent_tensor = latent_tensor / latents_std + latents_mean
        output2 = model2.decode(latent_tensor)
        output1 = model1.decode(latent_tensor).sample
        # Check if outputs have the same shape
        assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"

        # Check if outputs are similar
        max_diff_decode = torch.max(torch.abs(output1 - output2))
        mean_diff_decode = torch.mean(torch.abs(output1 - output2))
        logger.info("Maximum difference between decoded outputs: %s",
                    max_diff_decode.item())
        logger.info("Mean difference between decoded outputs: %s",
                    mean_diff_decode.item())
        assert mean_diff_decode < 1e-1, f"Decoded outputs differ significantly: mean diff = {mean_diff_decode.item()}"