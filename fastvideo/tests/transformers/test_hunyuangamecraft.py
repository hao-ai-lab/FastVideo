# SPDX-License-Identifier: Apache-2.0
"""
Numerical parity test for HunyuanGameCraft transformer.

This test compares the FastVideo implementation against the official
Hunyuan-GameCraft implementation to ensure numerical alignment.

Requirements:
- Official HunyuanGameCraft weights downloaded to official_weights/HunyuanGameCraft/
- Or converted weights in converted_weights/HunyuanGameCraft/

Usage:
    pytest fastvideo/tests/transformers/test_hunyuangamecraft.py -v

Environment variables:
    GAMECRAFT_WEIGHTS_PATH: Path to official/converted GameCraft weights
    GAMECRAFT_OFFICIAL_REPO: Path to cloned Hunyuan-GameCraft-1.0 repo (for parity test)
"""

import os
import sys

import pytest
import torch

from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29506"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Paths - customize these for your setup
DEFAULT_WEIGHTS_PATH = "converted_weights/HunyuanGameCraft/transformer"
WEIGHTS_PATH = os.environ.get("GAMECRAFT_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)
OFFICIAL_REPO_PATH = os.environ.get(
    "GAMECRAFT_OFFICIAL_REPO", 
    "Hunyuan-GameCraft-1.0"
)


def _skip_if_weights_missing():
    """Skip test if weights are not available."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip(f"Weights not found at {WEIGHTS_PATH}")


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_forward():
    """
    Test that FastVideo HunyuanGameCraft model can do a forward pass.
    
    This is a basic smoke test that verifies the model can be loaded
    and run inference without errors.
    """
    _skip_if_weights_missing()
    
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    
    logger.info(f"Loading model from {WEIGHTS_PATH}")
    
    args = FastVideoArgs(
        model_path=WEIGHTS_PATH,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(), 
            dit_precision=precision_str
        ),
    )
    args.device = device
    
    loader = TransformerLoader()
    model = loader.load(WEIGHTS_PATH, args).to(device, dtype=precision)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Test inputs
    torch.manual_seed(42)
    
    batch_size = 1
    latent_frames = 9  # Standard GameCraft frame count
    latent_height = 44  # 704 / 16
    latent_width = 80   # 1280 / 16
    text_seq_len = 256
    text_dim = 4096
    pooled_dim = 768
    
    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 16, latent_frames, latent_height, latent_width,
        device=device, dtype=precision
    )
    
    if sp_world_size > 1:
        chunk_per_rank = hidden_states.shape[2] // sp_world_size
        hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]
    
    # Text embeddings [B, L+1, D] (with global token prepended)
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len + 1, text_dim,
        device=device, dtype=precision
    )
    # Override first token for pooled projection
    encoder_hidden_states[:, 0, :pooled_dim] = torch.randn(
        batch_size, pooled_dim, device=device, dtype=precision
    )
    
    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)
    
    # Guidance
    guidance = torch.tensor([6016.0], device=device, dtype=precision)
    
    # Camera states (Plücker coordinates) [B, F, 6, H, W]
    # Note: These are in pixel space before downsampling
    num_frames_pixel = (latent_frames - 1) * 4 + 1  # Approximate based on VAE temporal compression
    camera_height = 704  # Original pixel height
    camera_width = 1280   # Original pixel width
    camera_states = torch.randn(
        batch_size, num_frames_pixel, 6, camera_height, camera_width,
        device=device, dtype=precision
    )
    
    forward_batch = ForwardBatch(data_type="dummy")
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    guidance=guidance,
                    camera_states=camera_states,
                )
    
    # Verify output shape
    expected_shape = (batch_size, 16, latent_frames, latent_height, latent_width)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    latent_sum = output.double().sum().item()
    logger.info(f"Output latent sum: {latent_sum}")
    
    # Basic sanity check - output should not be all zeros or NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    assert output.abs().sum() > 0, "Output is all zeros"
    
    logger.info("Forward pass test PASSED")


@pytest.mark.skip(reason="Requires official repo setup and matching weights")
@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_parity():
    """
    Numerical parity test comparing FastVideo vs Official implementation.
    
    This test loads both implementations with the same weights and inputs,
    then compares the outputs to ensure they match numerically.
    
    Requirements:
    - Official Hunyuan-GameCraft-1.0 repo cloned at GAMECRAFT_OFFICIAL_REPO
    - Weights available at GAMECRAFT_WEIGHTS_PATH
    """
    _skip_if_weights_missing()
    
    if not os.path.exists(OFFICIAL_REPO_PATH):
        pytest.skip(f"Official repo not found at {OFFICIAL_REPO_PATH}")
    
    # Add official repo to path
    sys.path.insert(0, OFFICIAL_REPO_PATH)
    
    try:
        from hymm_sp.modules.models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
    except ImportError as e:
        pytest.skip(f"Could not import official model: {e}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    
    # ============= Load Official Model =============
    logger.info("Loading official HunyuanGameCraft model...")
    
    # Create mock args for official model
    class MockArgs:
        def __init__(self):
            self.text_projection = "single_refiner"
            self.text_states_dim = 4096
            self.text_states_dim_2 = 768
            self.use_attention_mask = False
    
    mock_args = MockArgs()
    config = HUNYUAN_VIDEO_CONFIG["HYVideo-T/2"]
    
    official_model = HYVideoDiffusionTransformer(
        args=mock_args,
        **config,
        guidance_embed=True,
        dtype=precision,
        device=device,
    ).to(device)
    
    # ============= Load FastVideo Model =============
    logger.info("Loading FastVideo HunyuanGameCraft model...")
    
    args = FastVideoArgs(
        model_path=WEIGHTS_PATH,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(),
            dit_precision="bf16"
        ),
    )
    args.device = device
    
    loader = TransformerLoader()
    fastvideo_model = loader.load(WEIGHTS_PATH, args).to(device, dtype=precision)
    
    # Set both to eval mode
    official_model.eval()
    fastvideo_model.eval()
    
    # ============= Test Inputs =============
    torch.manual_seed(42)
    
    batch_size = 1
    latent_frames = 9
    latent_height = 44
    latent_width = 80
    text_seq_len = 256
    
    # Create identical inputs
    x = torch.randn(
        batch_size, 16, latent_frames, latent_height, latent_width,
        device=device, dtype=precision
    )
    
    text_states = torch.randn(
        batch_size, text_seq_len, 4096,
        device=device, dtype=precision
    )
    text_states_2 = torch.randn(
        batch_size, 768,
        device=device, dtype=precision
    )
    
    timestep = torch.tensor([500], device=device, dtype=precision)
    guidance = torch.tensor([6016.0], device=device, dtype=precision)
    
    # Camera states
    num_frames_pixel = (latent_frames - 1) * 4 + 1
    camera_states = torch.randn(
        batch_size, num_frames_pixel, 6, 704, 1280,
        device=device, dtype=precision
    )
    
    # RoPE frequencies (need to compute for official model)
    from hymm_sp.modules.posemb_layers import get_nd_rotary_pos_embed
    
    tt = latent_frames
    th = latent_height
    tw = latent_width
    rope_dim_list = config["rope_dim_list"]
    
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        (tt, th, tw),
        theta=256,
        use_real=True,
        theta_rescale_factor=1.0,
    )
    freqs_cos = freqs_cos.to(device, dtype=precision)
    freqs_sin = freqs_sin.to(device, dtype=precision)
    
    # ============= Forward Pass =============
    with torch.no_grad():
        # Official model
        official_output = official_model(
            x=x,
            t=timestep,
            text_states=text_states,
            text_mask=None,
            text_states_2=text_states_2,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=guidance,
            cam_latents=camera_states,
            return_dict=False,
        )
        
        # FastVideo model (combine text states for input format)
        encoder_hidden_states = torch.cat([
            text_states_2.unsqueeze(1),  # Add global token
            text_states,
        ], dim=1)
        encoder_hidden_states[:, 0, :768] = text_states_2
        
        forward_batch = ForwardBatch(data_type="dummy")
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
            fastvideo_output = fastvideo_model(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                camera_states=camera_states,
            )
    
    # ============= Compare Outputs =============
    official_sum = official_output.double().sum().item()
    fastvideo_sum = fastvideo_output.double().sum().item()
    
    diff = abs(official_sum - fastvideo_sum)
    relative_diff = diff / abs(official_sum) if official_sum != 0 else diff
    
    logger.info(f"Official output sum: {official_sum}")
    logger.info(f"FastVideo output sum: {fastvideo_sum}")
    logger.info(f"Absolute diff: {diff}")
    logger.info(f"Relative diff: {relative_diff * 100:.4f}%")
    
    # Use torch.testing.assert_close for detailed comparison
    try:
        torch.testing.assert_close(
            fastvideo_output, 
            official_output,
            atol=1e-2,  # Start with relaxed tolerance
            rtol=1e-2,
        )
        logger.info("Outputs match within tolerance!")
    except AssertionError as e:
        logger.error(f"Output mismatch: {e}")
        # Get max difference
        max_diff = (fastvideo_output - official_output).abs().max().item()
        logger.info(f"Max absolute difference: {max_diff}")
        raise
    
    logger.info("Parity test PASSED")


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_config():
    """Test that the HunyuanGameCraft config is properly set up."""
    config = HunyuanGameCraftConfig()
    arch = config.arch_config
    
    # Verify config values
    assert arch.hidden_size == 3072
    assert arch.num_attention_heads == 24
    assert arch.num_layers == 20  # double stream blocks
    assert arch.num_single_layers == 40  # single stream blocks
    assert arch.guidance_embeds is True
    assert arch.camera_in_channels == 6  # Plücker coordinates
    assert arch.camera_downscale_coef == 8
    
    # Verify param_names_mapping has CameraNet entries
    mapping = arch.param_names_mapping
    camera_patterns = [k for k in mapping.keys() if 'camera' in k]
    assert len(camera_patterns) > 0, "CameraNet mapping patterns missing"
    
    logger.info("Config test PASSED")
