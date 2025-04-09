from fastvideo.v1.models.vaes.hunyuanvae import AutoencoderKLHunyuanVideo as MyHunyuanVAE
from diffusers import AutoencoderKLHunyuanVideo as DiffusersHunyuanVAE

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import json
from fastvideo.v1.logger import init_logger
from safetensors.torch import load_file

logger = init_logger(__name__)


def initialize_identical_weights(model1, model2, seed=42):
    """Initialize both models with identical weights using a fixed seed for reproducibility."""
    # Get all parameters from both models
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Initialize each layer with identical values
    with torch.no_grad():
        # Initialize weights
        for name1, param1 in params1.items():
            if 'weight' in name1:
                # Set seed before each weight initialization
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)

        for name2, param2 in params2.items():
            if 'weight' in name2:
                # Reset seed to get same initialization
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.05)

        # Initialize biases
        for name1, param1 in params1.items():
            if 'bias' in name1:
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)
                param1.data = param1.data.to(torch.bfloat16)

        for name2, param2 in params2.items():
            if 'bias' in name2:
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.05)
                param2.data = param2.data.to(torch.bfloat16)

    logger.info("Both models initialized with identical weights in bfloat16")
    return model1, model2


def setup_args():
    parser = argparse.ArgumentParser(description='HunyuanVAE Test')
    parser.add_argument('--in-channels',
                        type=int,
                        default=4,
                        help='Number of input channels')
    parser.add_argument('--out-channels',
                        type=int,
                        default=4,
                        help='Number of output channels')
    parser.add_argument('--latent-channels',
                        type=int,
                        default=4,
                        help='Number of latent channels')
    return parser.parse_args()


def test_hunyuan_vae():
    args = setup_args()

    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Model parameters
    in_channels = args.in_channels
    out_channels = args.out_channels
    latent_channels = args.latent_channels

    device = torch.device("cuda:0")
    print
    # Initialize the two model implementations
    path = "data/hunyuanvideo-community/HunyuanVideo/vae"
    config_path = os.path.join(path, "config.json")
    config = json.load(open(config_path))
    config.pop("_class_name")
    config.pop("_diffusers_version")
    model1 = MyHunyuanVAE(**config).to(torch.bfloat16)

    model2 = DiffusersHunyuanVAE(**config).to(torch.bfloat16)

    loaded = load_file(os.path.join(path,
                                    "diffusion_pytorch_model.safetensors"))
    model1.load_state_dict(loaded)
    model2.load_state_dict(loaded)

    # Set both models to eval mode
    model1.eval()
    model2.eval()

    # Move to GPU
    model1 = model1.to(device)
    model2 = model2.to(device)

    model1.enable_tiling(tile_sample_min_height=32,
                         tile_sample_min_width=32,
                         tile_sample_min_num_frames=8,
                         tile_sample_stride_height=16,
                         tile_sample_stride_width=16,
                         tile_sample_stride_num_frames=4)
    model2.enable_tiling(tile_sample_min_height=32,
                         tile_sample_min_width=32,
                         tile_sample_min_num_frames=8,
                         tile_sample_stride_height=16,
                         tile_sample_stride_width=16,
                         tile_sample_stride_num_frames=4)

    # Create identical inputs for both models
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
        # Test encoding
        logger.info("Testing encoding...")
        latent1 = model1.encode(input_tensor).mean
        print("--------------------------------")
        latent2 = model2.encode(input_tensor).latent_dist.mean
        # Check if latents have the same shape
        assert latent1.shape == latent2.shape, f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        assert latent1.shape == latent2.shape, f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        # Check if latents are similar
        max_diff_encode = torch.max(torch.abs(latent1 - latent2))
        mean_diff_encode = torch.mean(torch.abs(latent1 - latent2))
        logger.info(
            f"Maximum difference between encoded latents: {max_diff_encode.item()}"
        )
        logger.info(
            f"Mean difference between encoded latents: {mean_diff_encode.item()}"
        )
        assert max_diff_encode < 1e-4, f"Encoded latents differ significantly: max diff = {max_diff_encode.item()}"
        # Test decoding
        logger.info("Testing decoding...")
        output1 = model1.decode(latent1)
        output2 = model2.decode(latent2).sample
        # Check if outputs have the same shape
        assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"

        # Check if outputs are similar
        max_diff_decode = torch.max(torch.abs(output1 - output2))
        mean_diff_decode = torch.mean(torch.abs(output1 - output2))
        logger.info(
            f"Maximum difference between decoded outputs: {max_diff_decode.item()}"
        )
        logger.info(
            f"Mean difference between decoded outputs: {mean_diff_decode.item()}"
        )
        assert max_diff_decode < 1e-4, f"Decoded outputs differ significantly: max diff = {max_diff_decode.item()}"

    logger.info(
        "Test passed! Both VAE implementations produce similar outputs.")
    logger.info("Test completed successfully")


if __name__ == "__main__":
    test_hunyuan_vae()
