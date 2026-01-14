# SPDX-License-Identifier: Apache-2.0
"""Tests for SigLIP vision encoder."""

import gc
import os

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import SiglipVisionModel as HFSiglipVisionModel

from fastvideo.configs.models.encoders import SiglipVisionConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29504"

# SigLIP model path - using the vision model from google/siglip-so400m-patch14-384
BASE_MODEL_PATH = "google/siglip-so400m-patch14-384"
MODEL_PATH = maybe_download_model(
    BASE_MODEL_PATH,
    local_dir=os.path.join("data", BASE_MODEL_PATH)
)


def create_dummy_pixel_values(batch_size: int = 2, image_size: int = 384, num_channels: int = 3):
    """Create dummy pixel values for testing.
    
    Args:
        batch_size: Number of images in the batch.
        image_size: Size of the image (height and width).
        num_channels: Number of channels in the image.
    
    Returns:
        Tensor of shape [batch_size, num_channels, image_size, image_size]
    """
    return torch.randn(batch_size, num_channels, image_size, image_size)


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_output_shape():
    """Test that SigLIP encoder produces outputs with expected shapes."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create config with default architecture settings
    config = SiglipVisionConfig()
    arch_config = config.arch_config
    
    # Expected output dimensions
    image_size = arch_config.image_size  # 384
    patch_size = arch_config.patch_size  # 14
    hidden_size = arch_config.hidden_size  # 1152
    num_patches = (image_size // patch_size) ** 2  # 729 patches
    
    logger.info("Creating SigLIP vision model with config: %s", config)
    
    from fastvideo.models.encoders.siglip import SiglipVisionModel
    model = SiglipVisionModel(config).to(device).to(torch.float16).eval()
    
    batch_size = 2
    pixel_values = create_dummy_pixel_values(
        batch_size=batch_size,
        image_size=image_size,
        num_channels=arch_config.num_channels
    ).to(device).to(torch.float16)
    
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = model(pixel_values)
    
    # Check output shape
    assert outputs.last_hidden_state is not None, "Expected last_hidden_state to be present"
    expected_shape = (batch_size, num_patches, hidden_size)
    actual_shape = outputs.last_hidden_state.shape
    
    logger.info("Expected shape: %s, Actual shape: %s", expected_shape, actual_shape)
    assert actual_shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"
    
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_vs_huggingface():
    """
    Test compatibility between FastVideo SigLIP encoder and HuggingFace implementation.
    
    The test verifies that both implementations:
    - Load models with the same weights and parameters
    - Produce nearly identical outputs for the same input
    """
    import json
    import os
    from safetensors.torch import load_file
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading SigLIP models from %s", MODEL_PATH)
    
    # Load HuggingFace implementation
    hf_model = HFSiglipVisionModel.from_pretrained(MODEL_PATH).to(torch.float16).to(device).eval()
    
    # Load the config and extract vision_config
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        full_config = json.load(f)
    
    # Get vision config from the full config
    vision_config_dict = full_config.get("vision_config", {})
    
    # Create FastVideo config with vision-specific settings
    from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig
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
    
    # Load weights from safetensors
    weights_path = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(weights_path):
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
    
    # Create random images
    pixel_values = create_dummy_pixel_values(
        batch_size=batch_size,
        image_size=image_size,
        num_channels=3
    ).to(device).to(torch.float16)
    
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


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_embeddings():
    """Test that SigLIP embeddings layer works correctly."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig
    from fastvideo.models.encoders.siglip import SiglipVisionEmbeddings
    
    arch_config = SiglipVisionArchConfig()
    embeddings = SiglipVisionEmbeddings(arch_config).to(device).to(torch.float16)
    
    batch_size = 2
    pixel_values = create_dummy_pixel_values(
        batch_size=batch_size,
        image_size=arch_config.image_size,
        num_channels=arch_config.num_channels
    ).to(device).to(torch.float16)
    
    with torch.no_grad():
        output = embeddings(pixel_values)
    
    # Check output shape: [batch_size, num_patches, hidden_size]
    expected_num_patches = (arch_config.image_size // arch_config.patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches, arch_config.hidden_size)
    
    assert output.shape == expected_shape, \
        f"Embeddings output shape mismatch: expected {expected_shape}, got {output.shape}"
    
    logger.info("SigLIP embeddings test passed with shape %s", output.shape)
    
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_attention():
    """Test that SigLIP attention layer works correctly."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig
    from fastvideo.models.encoders.siglip import SiglipAttention
    
    arch_config = SiglipVisionArchConfig()
    attention = SiglipAttention(arch_config).to(device).to(torch.float16)
    
    batch_size = 2
    seq_len = 729  # 27x27 patches for 384x384 image with 14x14 patches
    hidden_size = arch_config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device).to(torch.float16)
    
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            output, _ = attention(hidden_states)
    
    # Output should have the same shape as input
    assert output.shape == hidden_states.shape, \
        f"Attention output shape mismatch: expected {hidden_states.shape}, got {output.shape}"
    
    logger.info("SigLIP attention test passed with shape %s", output.shape)
    
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_mlp():
    """Test that SigLIP MLP layer works correctly."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig
    from fastvideo.models.encoders.siglip import SiglipMLP
    
    arch_config = SiglipVisionArchConfig()
    mlp = SiglipMLP(arch_config).to(device).to(torch.float16)
    
    batch_size = 2
    seq_len = 729
    hidden_size = arch_config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device).to(torch.float16)
    
    with torch.no_grad():
        output = mlp(hidden_states)
    
    # Output should have the same shape as input
    assert output.shape == hidden_states.shape, \
        f"MLP output shape mismatch: expected {hidden_states.shape}, got {output.shape}"
    
    logger.info("SigLIP MLP test passed with shape %s", output.shape)
    
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_layer():
    """Test that SigLIP encoder layer (attention + MLP) works correctly."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig
    from fastvideo.models.encoders.siglip import SiglipEncoderLayer
    
    arch_config = SiglipVisionArchConfig()
    encoder_layer = SiglipEncoderLayer(arch_config).to(device).to(torch.float16)
    
    batch_size = 2
    seq_len = 729
    hidden_size = arch_config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device).to(torch.float16)
    
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            output = encoder_layer(hidden_states)
    
    # Output should have the same shape as input
    assert output.shape == hidden_states.shape, \
        f"Encoder layer output shape mismatch: expected {hidden_states.shape}, got {output.shape}"
    
    logger.info("SigLIP encoder layer test passed with shape %s", output.shape)
    
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_with_feature_sample_layers():
    """Test SigLIP encoder with feature_sample_layers to return intermediate hidden states."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create config with fewer layers for faster testing
    config = SiglipVisionConfig(num_hidden_layers_override=4)
    
    from fastvideo.models.encoders.siglip import SiglipVisionModel
    model = SiglipVisionModel(config).to(device).to(torch.float16).eval()
    
    batch_size = 2
    image_size = config.arch_config.image_size
    
    pixel_values = create_dummy_pixel_values(
        batch_size=batch_size,
        image_size=image_size,
        num_channels=3
    ).to(device).to(torch.float16)
    
    # Request intermediate hidden states
    feature_sample_layers = [0, 1, 2, 3]
    
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = model(pixel_values, feature_sample_layers=feature_sample_layers)
    
    # When feature_sample_layers is specified, last_hidden_state should be a list
    assert isinstance(outputs.last_hidden_state, list), \
        "Expected list of hidden states when feature_sample_layers is specified"
    
    # Should have len(layers) + 1 hidden states (including input embeddings)
    expected_num_states = 5  # 4 layers + initial embeddings
    assert len(outputs.last_hidden_state) == expected_num_states, \
        f"Expected {expected_num_states} hidden states, got {len(outputs.last_hidden_state)}"
    
    logger.info("SigLIP feature sample layers test passed with %d hidden states", 
                len(outputs.last_hidden_state))
    
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("distributed_setup")
def test_siglip_encoder_num_hidden_layers_override():
    """Test that num_hidden_layers_override correctly limits the number of layers."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create model with override
    num_layers_override = 4
    config = SiglipVisionConfig(num_hidden_layers_override=num_layers_override)
    
    from fastvideo.models.encoders.siglip import SiglipVisionModel
    model = SiglipVisionModel(config).to(device).to(torch.float16).eval()
    
    # Check that the model has the expected number of layers
    actual_num_layers = len(model.vision_model.encoder.layers)
    assert actual_num_layers == num_layers_override, \
        f"Expected {num_layers_override} layers, got {actual_num_layers}"
    
    # Verify it can still process images
    batch_size = 1
    image_size = config.arch_config.image_size
    
    pixel_values = create_dummy_pixel_values(
        batch_size=batch_size,
        image_size=image_size,
        num_channels=3
    ).to(device).to(torch.float16)
    
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = model(pixel_values)
    
    assert outputs.last_hidden_state is not None
    logger.info("SigLIP num_hidden_layers_override test passed with %d layers", num_layers_override)
    
    gc.collect()
    torch.cuda.empty_cache()
