# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from diffusers import AutoencoderKLHunyuanVideo15

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.logger import init_logger
# from fastvideo.models.vaes.hunyuanvae import (
#     AutoencoderKLHunyuanVideo as MyHunyuanVAE)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.configs.models.vaes import Hunyuan15VAEConfig
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      "data", BASE_MODEL_PATH))
VAE_PATH = os.path.join(MODEL_PATH, "vae")
CONFIG_PATH = os.path.join(VAE_PATH, "config.json")


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan_vae():
    device = torch.device("cuda:0")
    precision = torch.float32
    precision_str = "fp32"
    args = FastVideoArgs(model_path=VAE_PATH, pipeline_config=PipelineConfig(vae_config=Hunyuan15VAEConfig(), vae_precision=precision_str))
    args.device = device
    args.vae_cpu_offload = False

    model1 = AutoencoderKLHunyuanVideo15.from_pretrained(
        VAE_PATH, torch_dtype=precision).to(device).eval()
    model1.enable_tiling()

    loader = VAELoader()
    model2 = loader.load(VAE_PATH, args)

    model2.enable_tiling()

    batch_size = 1

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(batch_size,
                               3,
                               81,
                               512,
                               512,
                               device=device,
                               dtype=precision)

    # Disable gradients for inference
    with torch.no_grad():
        latent1 = model1.encode(input_tensor, return_dict=False)[0].mode()
        latent2 = model2.encode(input_tensor).mode()

    assert latent1.shape == latent2.shape, f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
    max_diff_encode = torch.max(torch.abs(latent1.float() - latent2.float()))
    mean_diff_encode = torch.mean(torch.abs(latent1.float() - latent2.float()))
    logger.info("Maximum difference between encoded latents: %s",
                max_diff_encode.item())
    logger.info("Mean difference between encoded latents: %s",
                mean_diff_encode.item())
    assert max_diff_encode < 1e-5, f"Encoded latents differ significantly: max diff = {max_diff_encode.item()}, mean diff = {mean_diff_encode.item()}"

    # Test decoding
    latent1 = latent1 / model1.config.scaling_factor
    latent2 = latent2 / model2.config.scaling_factor

    with torch.no_grad():
        video1 = model1.decode(latent1, return_dict=False)[0]
        video2 = model2.decode(latent2)

    assert video1.shape == video2.shape, f"Video shapes don't match: {video1.shape} vs {video2.shape}"
    max_diff_decode = torch.max(torch.abs(video1.float() - video2.float()))
    mean_diff_decode = torch.mean(torch.abs(video1.float() - video2.float()))
    logger.info("Maximum difference between decoded videos: %s",
                max_diff_decode.item())
    logger.info("Mean difference between decoded videos: %s",
                mean_diff_decode.item())
    assert max_diff_decode < 1e-5, f"Decoded videos differ significantly: max diff = {max_diff_decode.item()}, mean diff = {mean_diff_decode.item()}"

