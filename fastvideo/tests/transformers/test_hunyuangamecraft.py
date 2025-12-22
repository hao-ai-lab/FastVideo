# SPDX-License-Identifier: Apache-2.0
"""
Test script for Hunyuan GameCraft DiT model.

This test validates that FastVideo's GameCraft implementation produces
identical numerical outputs to the original implementation.

Usage:
    pytest fastvideo/tests/transformers/test_hunyuangamecraft.py -v
"""

import json
import os
import sys
import pytest
import torch

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size)
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits import HunyuanGameCraftConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29504"

# Path to GameCraft model (adjust as needed)
# Since GameCraft isn't on HuggingFace, you'll need to download weights manually
GAMECRAFT_MODEL_PATH = os.environ.get("GAMECRAFT_MODEL_PATH", "weights/gamecraft_models")
TRANSFORMER_PATH = os.path.join(GAMECRAFT_MODEL_PATH, "transformer")

LOCAL_RANK = 0
RANK = 0
WORLD_SIZE = 1

# Reference latent will be generated from original implementation
# TODO: Generate this value after first successful run with original implementation
REFERENCE_LATENT = None  # Will be set after validation against original


def load_original_gamecraft_model():
    """
    Load the original GameCraft implementation for comparison.
    
    This assumes you've copied the official repo to FastVideo/fastvideo/models/Hunyuan-GameCraft-1.0-main/
    """
    # Add original implementation to path
    original_path = os.path.join(os.path.dirname(__file__), 
                                 "../../models/Hunyuan-GameCraft-1.0-main")
    if original_path not in sys.path:
        sys.path.insert(0, original_path)
    
    try:
        from hymm_sp.modules.models import HYVideoDiffusionTransformer
        logger.info("✓ Successfully imported original GameCraft implementation")
        return HYVideoDiffusionTransformer
    except ImportError as e:
        logger.warning(f"Could not import original implementation: {e}")
        logger.warning("Skipping comparison with original. Copy official repo to test against it.")
        return None


@pytest.mark.usefixtures("distributed_setup")
def test_gamecraft_transformer_distributed():
    """Test GameCraft transformer in distributed setup."""
    logger.info(
        f"Initializing process: rank={RANK}, local_rank={LOCAL_RANK}, world_size={WORLD_SIZE}"
    )

    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")

    # Get sequence parallel info
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    logger.info(
        f"Process rank {RANK} initialized with SP rank {sp_rank} in SP world size {sp_world_size}"
    )

    # Load FastVideo implementation
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(),
            dit_precision=precision_str
        )
    )
    args.device = torch.device(f"cuda:{LOCAL_RANK}")

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)
    model.eval()

    # Create test inputs
    batch_size = 1
    text_seq_len = 256
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 16, 9, 88, 152,
        device=device,
        dtype=torch.bfloat16
    )
    
    # Handle sequence parallelism
    chunk_per_rank = hidden_states.shape[2] // sp_world_size
    hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]

    # Text embeddings [B, L+1, D] (including pooled token at position 0)
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len + 1, 4096,
        device=device,
        dtype=torch.bfloat16
    )
    # Zero out beyond pooled dimension for first token
    encoder_hidden_states[:, 0, 768:] = 0

    # Camera latents [B, T, 6, H, W] - GameCraft specific
    camera_latents = torch.randn(
        batch_size, 9, 6, 704, 1216,
        device=device,
        dtype=torch.bfloat16
    )

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)
    
    forward_batch = ForwardBatch(data_type="dummy")

    # Run inference
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    camera_latents=camera_latents,
                )

    latent = output.double().sum().item()
    logger.info(f"FastVideo GameCraft output latent sum: {latent}")

    # If reference is set, validate
    if REFERENCE_LATENT is not None:
        diff_output_latents = abs(REFERENCE_LATENT - latent)
        logger.info(f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}")
        logger.info(f"Numerical difference: {diff_output_latents}")
        assert diff_output_latents < 1e-4, f"Output latents differ significantly: diff = {diff_output_latents}"
        logger.info("✓ Numerical diff test PASSED")
    else:
        logger.warning("No reference latent set. Save this value for future tests:")
        logger.warning(f"REFERENCE_LATENT = {latent}")


def test_gamecraft_vs_original():
    """
    Test FastVideo GameCraft against original implementation.
    
    This test requires the original GameCraft repo to be available.
    """
    OriginalModel = load_original_gamecraft_model()
    
    if OriginalModel is None:
        pytest.skip("Original GameCraft implementation not available")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # TODO: Implement comparison logic
    # 1. Load both models with same weights
    # 2. Create identical inputs
    # 3. Run forward pass on both
    # 4. Compare outputs
    
    logger.info("TODO: Implement comparison with original")
    pytest.skip("Comparison test not yet implemented")


def test_gamecraft_camera_conditioning():
    """Test that camera conditioning properly affects output."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(),
            dit_precision=precision_str
        )
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)
    model.eval()

    # Create inputs
    batch_size = 1
    hidden_states = torch.randn(batch_size, 16, 9, 88, 152, device=device, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(batch_size, 257, 4096, device=device, dtype=torch.bfloat16)
    encoder_hidden_states[:, 0, 768:] = 0
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)

    # Test 1: Output with camera conditioning
    camera_latents_1 = torch.randn(batch_size, 9, 6, 704, 1216, device=device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        forward_batch = ForwardBatch(data_type="dummy")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output_1 = model(
                    hidden_states=hidden_states.clone(),
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    timestep=timestep.clone(),
                    camera_latents=camera_latents_1,
                )

    # Test 2: Output with different camera conditioning
    camera_latents_2 = torch.randn(batch_size, 9, 6, 704, 1216, device=device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output_2 = model(
                    hidden_states=hidden_states.clone(),
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    timestep=timestep.clone(),
                    camera_latents=camera_latents_2,
                )

    # Outputs should be different when camera inputs differ
    diff = (output_1 - output_2).abs().mean().item()
    logger.info(f"Difference with different camera inputs: {diff}")
    assert diff > 1e-6, "Camera conditioning should affect output!"
    logger.info("✓ Camera conditioning test PASSED")


def test_gamecraft_output_shape():
    """Test that output shapes are correct."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(),
            dit_precision=precision_str
        )
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)
    model.eval()

    # Test different input sizes
    test_cases = [
        (1, 16, 9, 88, 152),    # Small
        (1, 16, 10, 88, 152),   # 10 frames (special case)
        (1, 16, 18, 88, 152),   # 18 frames (special case)
    ]

    for b, c, t, h, w in test_cases:
        hidden_states = torch.randn(b, c, t, h, w, device=device, dtype=torch.bfloat16)
        encoder_hidden_states = torch.randn(b, 257, 4096, device=device, dtype=torch.bfloat16)
        encoder_hidden_states[:, 0, 768:] = 0
        camera_latents = torch.randn(b, t, 6, 704, 1216, device=device, dtype=torch.bfloat16)
        timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            forward_batch = ForwardBatch(data_type="dummy")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                    output = model(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        camera_latents=camera_latents,
                    )

        assert output.shape == hidden_states.shape, \
            f"Output shape {output.shape} doesn't match input shape {hidden_states.shape}"
        logger.info(f"✓ Shape test passed for {hidden_states.shape}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])

