# SPDX-License-Identifier: Apache-2.0
"""Tests for SigLIP vision encoder using HYWorld model."""

import gc
import json
import os

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import SiglipVisionModel as HFSiglipVisionModel

from fastvideo.configs.models.encoders import SiglipVisionConfig
from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"

# HYWorld model path - SigLIP image encoder
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
IMAGE_ENCODER_SUBFOLDER = "image_encoder"


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder():
    """
    Test compatibility between FastVideo SigLIP encoder and HuggingFace implementation.
    
    The test verifies that both implementations:
    - Load models with the same weights and parameters
    - Produce nearly identical outputs for the same input
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading SigLIP models from %s/%s", MODEL_ID, IMAGE_ENCODER_SUBFOLDER)
    
    # Load HuggingFace implementation
    hf_model = HFSiglipVisionModel.from_pretrained(
        MODEL_ID, subfolder=IMAGE_ENCODER_SUBFOLDER
    ).to(torch.float16).to(device).eval()
    
    # Load the config from Hugging Face and extract vision_config
    config_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=f"{IMAGE_ENCODER_SUBFOLDER}/config.json"
    )
    with open(config_path) as f:
        full_config = json.load(f)
    
    # Get vision config from the full config
    vision_config_dict = full_config.get("vision_config", full_config)
    
    # Create FastVideo config with vision-specific settings
    arch_config = SiglipVisionArchConfig(
        hidden_size=vision_config_dict.get("hidden_size", 1152),
        image_size=vision_config_dict.get("image_size", 384),
        intermediate_size=vision_config_dict.get("intermediate_size", 4304),
        num_attention_heads=vision_config_dict.get("num_attention_heads", 16),
        num_hidden_layers=vision_config_dict.get("num_hidden_layers", 27),
        patch_size=vision_config_dict.get("patch_size", 14),
    )
    
    config = SiglipVisionConfig(arch_config=arch_config)
    
    # Create FastVideo model
    from fastvideo.models.encoders.siglip import SiglipVisionModel
    fv_model = SiglipVisionModel(config).to(torch.float16).to(device)
    
    # Load weights from safetensors via Hugging Face
    weights_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=f"{IMAGE_ENCODER_SUBFOLDER}/model.safetensors"
    )
    state_dict = load_file(weights_path)
    # Filter to only vision_model weights (keep the vision_model. prefix)
    vision_weights = [
        (name, weight)
        for name, weight in state_dict.items()
        if name.startswith("vision_model.")
    ]
    fv_model.load_weights(vision_weights)
    fv_model.eval()
    
    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(hf_model.named_parameters())
    params2 = dict(fv_model.named_parameters())
    
    logger.info("HuggingFace model has %d parameters", len(params1))
    logger.info("FastVideo model has %d parameters", len(params2))
    
    # Compare non-stacked parameters
    # Note: HF uses param names directly, FV adds "vision_model." prefix
    for name1, param1 in sorted(params1.items()):
        # Map HF param name to FV param name
        name2 = "vision_model." + name1
        skip = False
        for param_name, weight_name, shard_id in fv_model.config.arch_config.stacked_params_mapping:
            if weight_name in name1:
                skip = True
                break
        # stacked params (qkv) are more troublesome to compare
        if skip:
            continue
        if name2 in params2:
            param2 = params2[name2]
            param2 = param2.to_local().to(device) if isinstance(param2, DTensor) else param2.to(device)
            assert_close(param1, param2, atol=1e-4, rtol=1e-4)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test with sample images
    batch_size = 2
    image_size = fv_model.config.arch_config.image_size
    
    # Create random pixel values
    pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device).to(torch.float16)
    
    logger.info("Testing SigLIP encoder with random pixel values of shape %s", pixel_values.shape)
    
    with torch.no_grad():
        # Get embeddings from HuggingFace implementation
        hf_outputs = hf_model(pixel_values=pixel_values)
        
        # Get embeddings from FastVideo implementation
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fv_outputs = fv_model(pixel_values=pixel_values)
    
    # Compare last hidden states
    hf_hidden_state = hf_outputs.last_hidden_state
    fv_hidden_state = fv_outputs.last_hidden_state
    
    logger.info("HF hidden state shape: %s", hf_hidden_state.shape)
    logger.info("FV hidden state shape: %s", fv_hidden_state.shape)
    
    assert hf_hidden_state.shape == fv_hidden_state.shape, \
        f"Hidden state shapes don't match: {hf_hidden_state.shape} vs {fv_hidden_state.shape}"
    
    # Compare outputs with tolerance for numerical differences
    assert_close(hf_hidden_state, fv_hidden_state, atol=1e-2, rtol=1e-3)
    
    logger.info("SigLIP encoder test passed - outputs match within tolerance")
    
    gc.collect()
    torch.cuda.empty_cache()
