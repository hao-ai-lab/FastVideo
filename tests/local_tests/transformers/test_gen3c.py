# SPDX-License-Identifier: Apache-2.0
"""
Usage:
    # Basic tests (no weights required)
    pytest tests/local_tests/transformers/test_gen3c.py -v
    
    # Full parity tests (requires converted weights)
    GEN3C_FASTVIDEO_PATH=./gen3c_converted/transformer pytest tests/local_tests/transformers/test_gen3c.py -v
"""

import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29514")

repo_root = Path(__file__).resolve().parents[3]
gen3c_path = repo_root / "GEN3C"
if gen3c_path.exists() and str(gen3c_path) not in sys.path:
    sys.path.insert(0, str(gen3c_path))

from fastvideo.configs.models.dits.gen3c import Gen3CVideoConfig, Gen3CArchConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.gen3c import Gen3CTransformer3DModel


def create_test_config(
    frame_buffer_max: int = 2,
    num_layers: int = 2,  # Reduced for faster testing
    max_size: tuple[int, int, int] = (16, 32, 32),
) -> Gen3CVideoConfig:
    """Create a minimal GEN3C config for testing."""
    config = Gen3CVideoConfig()
    config.arch_config = Gen3CArchConfig()
    
    # Override for faster testing
    config.arch_config.num_layers = num_layers
    config.arch_config.max_size = max_size
    config.arch_config.frame_buffer_max = frame_buffer_max
    
    # Ensure buffer channels are recalculated
    config.arch_config.__post_init__()
    
    return config


def test_gen3c_model_instantiation():
    """Test that GEN3C model can be instantiated."""
    config = create_test_config()
    
    model = Gen3CTransformer3DModel(config=config, hf_config={})
    
    # Verify model attributes
    assert model.hidden_size == config.arch_config.num_attention_heads * config.arch_config.attention_head_dim
    assert model.in_channels == config.arch_config.in_channels
    assert model.out_channels == config.arch_config.out_channels
    assert len(model.transformer_blocks) == config.arch_config.num_layers
    
    print(f"[GEN3C TEST] Model instantiated successfully")
    print(f"  - Hidden size: {model.hidden_size}")
    print(f"  - Num layers: {len(model.transformer_blocks)}")
    print(f"  - Frame buffer max: {model.frame_buffer_max}")
    print(f"  - Buffer channels: {model.buffer_channels}")


def test_gen3c_forward_pass_shapes():
    """Test that GEN3C forward pass produces correct output shapes.
    
    Note: This test requires distributed initialization for DistributedAttention.
    It will be skipped if run without proper distributed setup.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GEN3C forward pass test")
    
    # Check if distributed is initialized
    try:
        from fastvideo.distributed.parallel_state import get_sp_group
        get_sp_group()
    except (AssertionError, ImportError):
        pytest.skip("Distributed parallel state not initialized. Run with TransformerLoader or full pipeline.")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    config = create_test_config()
    model = Gen3CTransformer3DModel(config=config, hf_config={}).to(device=device, dtype=dtype)
    model.eval()
    
    # Create test inputs
    batch_size = 1
    num_frames = 8  # Divisible by patch_t=1
    height = 16  # Divisible by patch_h=2
    width = 16  # Divisible by patch_w=2
    
    hidden_states = torch.randn(
        batch_size,
        config.arch_config.in_channels,
        num_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    
    encoder_hidden_states = torch.randn(
        batch_size,
        16,  # seq_len
        config.arch_config.text_embed_dim,
        device=device,
        dtype=dtype,
    )
    
    # GEN3C-specific inputs
    condition_video_input_mask = torch.zeros(
        batch_size, 1, num_frames, height, width,
        device=device, dtype=dtype,
    )
    
    condition_video_pose = torch.zeros(
        batch_size, config.arch_config.buffer_channels, num_frames, height, width,
        device=device, dtype=dtype,
    )
    
    condition_video_augment_sigma = torch.zeros(batch_size, device=device, dtype=dtype)
    
    padding_mask = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
    
    with torch.no_grad():
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                condition_video_input_mask=condition_video_input_mask,
                condition_video_pose=condition_video_pose,
                condition_video_augment_sigma=condition_video_augment_sigma,
                padding_mask=padding_mask,
            )
    
    # Verify output shape matches input latent shape
    assert output.shape == hidden_states.shape, (
        f"Output shape {output.shape} does not match input shape {hidden_states.shape}"
    )
    
    print(f"[GEN3C TEST] Forward pass successful")
    print(f"  - Input shape: {hidden_states.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output dtype: {output.dtype}")


def test_gen3c_forward_pass_no_conditioning():
    """Test GEN3C forward pass without explicit conditioning inputs.
    
    Note: This test requires distributed initialization for DistributedAttention.
    It will be skipped if run without proper distributed setup.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GEN3C forward pass test")
    
    # Check if distributed is initialized
    try:
        from fastvideo.distributed.parallel_state import get_sp_group
        get_sp_group()
    except (AssertionError, ImportError):
        pytest.skip("Distributed parallel state not initialized. Run with TransformerLoader or full pipeline.")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    config = create_test_config()
    model = Gen3CTransformer3DModel(config=config, hf_config={}).to(device=device, dtype=dtype)
    model.eval()
    
    batch_size = 1
    num_frames = 8
    height = 16
    width = 16
    
    hidden_states = torch.randn(
        batch_size,
        config.arch_config.in_channels,
        num_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    
    encoder_hidden_states = torch.randn(
        batch_size,
        16,
        config.arch_config.text_embed_dim,
        device=device,
        dtype=dtype,
    )
    
    with torch.no_grad():
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            # Call without conditioning inputs - model should use defaults
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )
    
    assert output.shape == hidden_states.shape
    print(f"[GEN3C TEST] Forward pass without conditioning successful")


def test_gen3c_buffer_channels_calculation():
    """Test that buffer channels are calculated correctly."""
    for frame_buffer_max in [1, 2, 3, 4]:
        config = create_test_config(frame_buffer_max=frame_buffer_max)
        
        expected_buffer_channels = frame_buffer_max * 32  # Each buffer: 16 (frame) + 16 (mask)
        actual_buffer_channels = config.arch_config.buffer_channels
        
        assert actual_buffer_channels == expected_buffer_channels, (
            f"Buffer channels mismatch for frame_buffer_max={frame_buffer_max}: "
            f"expected {expected_buffer_channels}, got {actual_buffer_channels}"
        )
        
        print(f"[GEN3C TEST] frame_buffer_max={frame_buffer_max} -> buffer_channels={actual_buffer_channels}")


def test_gen3c_patch_embed_channels():
    """Test that patch embedding has correct input channels."""
    config = create_test_config()
    model = Gen3CTransformer3DModel(config=config, hf_config={})
    
    # Expected input channels:
    # - in_channels (16): VAE latent
    # - condition_video_input_mask (1): Conditioning mask
    # - condition_video_pose (frame_buffer_max * 32): 3D cache buffers
    # - padding_mask (1 if concat_padding_mask): Padding mask
    expected_channels = (
        config.arch_config.in_channels +  # 16
        1 +  # condition_video_input_mask
        config.arch_config.buffer_channels +  # 64 for frame_buffer_max=2
        (1 if config.arch_config.concat_padding_mask else 0)  # padding_mask
    )
    
    actual_channels = model.patch_embed.dim // (
        model.patch_size[0] * model.patch_size[1] * model.patch_size[2]
    )
    
    assert actual_channels == expected_channels, (
        f"Patch embed input channels mismatch: expected {expected_channels}, got {actual_channels}"
    )
    
    print(f"[GEN3C TEST] Patch embed input channels: {actual_channels}")


@pytest.mark.skipif(
    not os.getenv("GEN3C_FASTVIDEO_PATH"),
    reason="GEN3C_FASTVIDEO_PATH not set - skip weight loading test"
)
def test_gen3c_weight_loading():
    """Test loading converted GEN3C weights."""
    from fastvideo.models.loader.component_loader import TransformerLoader
    
    fastvideo_path = Path(os.getenv("GEN3C_FASTVIDEO_PATH"))
    if not fastvideo_path.exists():
        pytest.skip(f"GEN3C converted weights not found at {fastvideo_path}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    config = Gen3CVideoConfig()
    args = FastVideoArgs(
        model_path=str(fastvideo_path),
        dit_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=config,
            dit_precision="bf16" if torch.cuda.is_available() else "fp32"
        ),
    )
    args.device = device
    
    loader = TransformerLoader()
    model = loader.load(str(fastvideo_path), args).to(device=device, dtype=dtype)
    
    print(f"[GEN3C TEST] Model loaded successfully from {fastvideo_path}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[GEN3C TEST] Total parameters: {total_params:,}")
    
    # Verify model is usable
    model.eval()
    assert model is not None


@pytest.mark.skipif(
    not os.getenv("GEN3C_OFFICIAL_PATH") or not os.getenv("GEN3C_FASTVIDEO_PATH"),
    reason="GEN3C_OFFICIAL_PATH and GEN3C_FASTVIDEO_PATH required for parity test"
)
def test_gen3c_transformer_parity():
    """Test parity between FastVideo and official GEN3C implementation."""
    from fastvideo.models.loader.component_loader import TransformerLoader
    
    official_path = Path(os.getenv("GEN3C_OFFICIAL_PATH"))
    fastvideo_path = Path(os.getenv("GEN3C_FASTVIDEO_PATH"))
    
    if not official_path.exists():
        pytest.skip(f"Official GEN3C weights not found at {official_path}")
    if not fastvideo_path.exists():
        pytest.skip(f"FastVideo converted weights not found at {fastvideo_path}")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for parity test")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # Load FastVideo model
    config = Gen3CVideoConfig()
    args = FastVideoArgs(
        model_path=str(fastvideo_path),
        dit_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=config, dit_precision="bf16"),
    )
    args.device = device
    
    loader = TransformerLoader()
    fastvideo_model = loader.load(str(fastvideo_path), args).to(device=device, dtype=dtype)
    fastvideo_model.eval()
    
    # Load official GEN3C model
    try:
        from cosmos_predict1.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
        
        official_checkpoint = torch.load(official_path, map_location="cpu")
        official_state_dict = official_checkpoint.get("state_dict", official_checkpoint)
        
        # Create official model with matching config
        official_model = VideoExtendGeneralDIT(
            max_img_h=720,
            max_img_w=1280,
            max_frames=128,
            in_channels=16 + 16 * 4 + 1,  # latent + buffers + mask
            out_channels=16,
            patch_spatial=(2, 2),
            patch_temporal=1,
            model_channels=2048,
            num_blocks=28,
            num_heads=16,
            mlp_ratio=4.0,
            crossattn_emb_channels=1024,
            use_adaln_lora=True,
            adaln_lora_dim=256,
            add_augment_sigma_embedding=True,
        )
        official_model.load_state_dict(official_state_dict, strict=False)
        official_model = official_model.to(device=device, dtype=dtype)
        official_model.eval()
        
    except ImportError as e:
        pytest.skip(f"Failed to import official GEN3C: {e}")
    except Exception as e:
        pytest.skip(f"Failed to load official GEN3C: {e}")
    
    # Create test inputs
    batch_size = 1
    num_frames = 8
    height = 16
    width = 16
    
    hidden_states = torch.randn(
        batch_size,
        16,
        num_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    
    encoder_hidden_states = torch.randn(
        batch_size,
        16,
        1024,
        device=device,
        dtype=dtype,
    )
    
    condition_video_input_mask = torch.zeros(
        batch_size, 1, num_frames, height, width,
        device=device, dtype=dtype,
    )
    
    condition_video_pose = torch.zeros(
        batch_size, 64, num_frames, height, width,
        device=device, dtype=dtype,
    )
    
    with torch.no_grad():
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            fastvideo_out = fastvideo_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                condition_video_input_mask=condition_video_input_mask,
                condition_video_pose=condition_video_pose,
            )
        
        # Prepare inputs for official model
        x_official = torch.cat([
            hidden_states,
            condition_video_input_mask,
            condition_video_pose,
        ], dim=1)
        
        padding_mask = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        
        official_out = official_model(
            x=x_official,
            timesteps=timestep,
            crossattn_emb=encoder_hidden_states,
            padding_mask=padding_mask,
        )
    
    # Compare outputs
    assert fastvideo_out.shape == official_out.shape, (
        f"Shape mismatch: FastVideo {fastvideo_out.shape} vs Official {official_out.shape}"
    )
    
    assert_close(fastvideo_out, official_out, atol=1e-4, rtol=1e-4)
    print("[GEN3C TEST] Parity test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
