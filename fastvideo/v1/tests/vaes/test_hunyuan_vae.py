# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from safetensors.torch import load_file

from fastvideo.v1.logger import init_logger
# from fastvideo.v1.models.vaes.hunyuanvae import (
#     AutoencoderKLHunyuanVideo as MyHunyuanVAE)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.models.loader.component_loader import VAELoader
from fastvideo.v1.configs.models.vaes import HunyuanVAEConfig
from fastvideo.v1.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "hunyuanvideo-community/HunyuanVideo"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      "data", BASE_MODEL_PATH))
VAE_PATH = os.path.join(MODEL_PATH, "vae")
CONFIG_PATH = os.path.join(VAE_PATH, "config.json")

# Latent generated on commit 250f0b916cebb18a1c15c4aae1a0b480604d066a with 1 x A40
REFERENCE_LATENT = -105.51324462890625


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan_vae():
    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=VAE_PATH, vae_precision=precision_str)
    args.device = device
    args.vae_config = HunyuanVAEConfig()

    loader = VAELoader()
    model = loader.load(VAE_PATH, "", args)

    model.enable_tiling(tile_sample_min_height=32,
                         tile_sample_min_width=32,
                         tile_sample_min_num_frames=8,
                         tile_sample_stride_height=16,
                         tile_sample_stride_width=16,
                         tile_sample_stride_num_frames=4)

    batch_size = 1

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(batch_size,
                               3,
                               21,
                               64,
                               64,
                               device=device,
                               dtype=torch.bfloat16)

    # Disable gradients for inference
    with torch.no_grad():
        latent = model.encode(input_tensor).mean.double().sum().item()

    # Check if latents are similar
    diff_encoded_latents = abs(REFERENCE_LATENT - latent)
    logger.info(
        f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}"
    )
    assert diff_encoded_latents < 1e-4, f"Encoded latents differ significantly: max diff = {diff_encoded_latents}"

