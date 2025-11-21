import os

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLLTXVideo

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.configs.models.vaes import LTXVAEConfig
from fastvideo.utils import maybe_download_model
from huggingface_hub import snapshot_download


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Lightricks/LTX-Video"


snapshot_download(
    "Lightricks/LTX-Video",
    local_dir="data/Lightricks/LTX-Video",
    allow_patterns=[
        "vae/*.json",
        "vae/*.safetensors",        # Explicitly allow safetensors in vae
        "transformer/*.json",
        "transformer/*.safetensors", # Explicitly allow safetensors in transformer
        "tokenizer/*",
        "scheduler/*",
        "*.json",
        "README.md"
    ]
)

MODEL_PATH = "data/Lightricks/LTX-Video"
VAE_PATH = os.path.join(MODEL_PATH, "vae")
print(f"VAE_PATH {VAE_PATH}")


@pytest.mark.usefixtures("distributed_setup")
def test_ltx_vae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=VAE_PATH, 
        pipeline_config=PipelineConfig(
            vae_config=LTXVAEConfig(), 
            vae_precision=precision_str
        )
    )
    args.device = device
    args.vae_cpu_offload = False

    loader = VAELoader()
    #model2 is the one i implemented
    model2 = loader.load(VAE_PATH, args)

    model1 = AutoencoderKLLTXVideo.from_pretrained(
        VAE_PATH, torch_dtype=precision).to(device).eval()

    # Create identical inputs for both models
    batch_size = 1

    model1_decoder_keys = [k for k in model1.state_dict().keys() if 'decoder' in k]
    model2_decoder_keys = [k for k in model2.state_dict().keys() if 'decoder' in k]

    logger.info(f"Model1 decoder keys sample: {model1_decoder_keys[:5]}")
    logger.info(f"Model2 decoder keys sample: {model2_decoder_keys[:5]}")

    # Check if keys match
    missing_in_model2 = set(model1_decoder_keys) - set(model2_decoder_keys)
    extra_in_model2 = set(model2_decoder_keys) - set(model1_decoder_keys)

    if missing_in_model2:
        logger.warning(f"Keys in model1 but not model2: {list(missing_in_model2)[:5]}")
    if extra_in_model2:
        logger.warning(f"Keys in model2 but not model1: {list(extra_in_model2)[:5]}")

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(batch_size,
                               3,
                               17,  # 17 frames
                               256,  # Height
                               256,  # Width
                               device=device,
                               dtype=precision)

    # Disable gradients for inference
    with torch.no_grad():
        # Test encoding
        logger.info("Testing encoding...")
        latent1_dist = model1.encode(input_tensor).latent_dist
        latent1 = latent1_dist.mean
        
        logger.info("FastVideo encoding...")
        latent2_dist = model2.encode(input_tensor).latent_dist
        latent2 = latent2_dist.mean
        
        # Check if latents have the same shape
        assert latent1.shape == latent2.shape, f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        
        # Check if latents are similar
        max_diff_encode = torch.max(torch.abs(latent1 - latent2))
        mean_diff_encode = torch.mean(torch.abs(latent1 - latent2))
        logger.info("Maximum difference between encoded latents: %s",
                    max_diff_encode.item())
        logger.info("Mean difference between encoded latents: %s",
                    mean_diff_encode.item())
        assert max_diff_encode < 1e-5, f"Encoded latents differ significantly: max diff = {max_diff_encode.item()}"
        
        # Test decoding
        logger.info("Testing decoding...")
        
        # For LTX, we need to use the mode of the distribution
        latent1_tensor = latent1_dist.mode()
        latent2_tensor = latent2_dist.mode()

        latent_diff = torch.max(torch.abs(latent1_tensor - latent2_tensor))
        logger.info(f"Latent difference before decoding: {latent_diff.item()}")
        
        if hasattr(model1.config, 'scaling_factor'):
            latent1_tensor = latent1_tensor * model1.config.scaling_factor
        print(f"model1.config.scaling_factor{model1.config.scaling_factor}")
            
        if hasattr(model2.config.arch_config, 'scaling_factor'):
            latent2_tensor = latent2_tensor * model2.config.arch_config.scaling_factor
        print(f"model2.config.arch_config.scaling_factor{model2.config.arch_config.scaling_factor}")
        output1 = model1.decode(latent1_tensor).sample
        output2 = model2.decode(latent2_tensor).sample
        logger.info(f"Output1 shape: {output1.shape}, range: [{output1.min().item():.4f}, {output1.max().item():.4f}]")
        logger.info(f"Output2 shape: {output2.shape}, range: [{output2.min().item():.4f}, {output2.max().item():.4f}]")

        
        # Check if outputs have the same shape
        assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"

        # Check if outputs are similar
        max_diff_decode = torch.max(torch.abs(output1 - output2))
        mean_diff_decode = torch.mean(torch.abs(output1 - output2))
        logger.info("Maximum difference between decoded outputs: %s",
                    max_diff_decode.item())
        logger.info("Mean difference between decoded outputs: %s",
                    mean_diff_decode.item())
        assert max_diff_decode < 1e-5, f"Decoded outputs differ significantly: max diff = {max_diff_decode.item()}"


# TODO: modify and test this
# @pytest.mark.usefixtures("distributed_setup")
# def test_ltx_vae_tiling():
#     """Test LTX VAE with tiling enabled for large videos."""
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     precision = torch.bfloat16
#     precision_str = "bf16"
#     args = FastVideoArgs(
#         model_path=VAE_PATH, 
#         pipeline_config=PipelineConfig(
#             vae_config=LTXVAEConfig(), 
#             vae_precision=precision_str
#         )
#     )
#     args.device = device
#     args.vae_cpu_offload = False

#     loader = VAELoader()
#     model2 = loader.load(VAE_PATH, args)

#     model1 = AutoencoderKLLTXVideo.from_pretrained(
#         VAE_PATH, torch_dtype=precision).to(device).eval()

#     # Enable tiling for both models
#     model1.enable_tiling()
#     if hasattr(model2, 'enable_tiling'):
#         model2.enable_tiling()

#     # Create larger input that requires tiling
#     batch_size = 1
#     input_tensor = torch.randn(batch_size,
#                                3,
#                                17,   # frames
#                                768,  # height (larger than default tile size)
#                                768,  # width (larger than default tile size)
#                                device=device,
#                                dtype=precision)

#     # Test with tiling
#     with torch.no_grad():
#         logger.info("Testing tiled encoding...")
#         latent1 = model1.encode(input_tensor).latent_dist.mean
#         latent2 = model2.encode(input_tensor).mean
        
#         max_diff = torch.max(torch.abs(latent1 - latent2))
#         logger.info("Max difference with tiling: %s", max_diff.item())
#         assert max_diff < 1e-4, f"Tiled encoding differs: max diff = {max_diff.item()}"