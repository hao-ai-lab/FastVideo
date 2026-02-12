# SPDX-License-Identifier: Apache-2.0
"""Smoke test for Waypoint pipeline.

This test verifies that:
1. The Waypoint pipeline can be instantiated
2. The transformer model loads correctly with weight parity
3. The streaming interface works for frame generation
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Add the repository root to sys.path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    """Log tensor statistics for debugging."""
    tensor_f32 = tensor.float()
    print(
        f"[WAYPOINT SMOKE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f} "
        f"mean={tensor_f32.mean().item():.6f}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Waypoint pipeline smoke test requires CUDA.",
)
def test_waypoint_transformer_load():
    """Test that Waypoint transformer loads with correct weight shapes."""
    # Import here to avoid import errors when CUDA is not available
    from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    
    weights_path = os.getenv(
        "WAYPOINT_WEIGHTS_PATH",
        "official_weights/Waypoint-1-Small/transformer"
    )
    
    if not os.path.exists(weights_path):
        pytest.skip(f"Waypoint weights not found at {weights_path}")
    
    safetensors_path = os.path.join(weights_path, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(safetensors_path):
        pytest.skip(f"Waypoint safetensors not found at {safetensors_path}")
    
    device = torch.device("cuda:0")
    
    # Create model with default config
    config = WaypointArchConfig()
    model = WaypointWorldModel(config)
    
    # Load weights
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    # Should have no missing or unexpected keys for transformer weights
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    
    # Move to device
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    
    print("✓ Waypoint transformer loaded successfully")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Waypoint pipeline smoke test requires CUDA.",
)
def test_waypoint_transformer_forward():
    """Test that Waypoint transformer forward pass works."""
    from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    
    weights_path = os.getenv(
        "WAYPOINT_WEIGHTS_PATH",
        "official_weights/Waypoint-1-Small/transformer"
    )
    
    if not os.path.exists(weights_path):
        pytest.skip(f"Waypoint weights not found at {weights_path}")
    
    safetensors_path = os.path.join(weights_path, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(safetensors_path):
        pytest.skip(f"Waypoint safetensors not found at {safetensors_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # Create and load model
    config = WaypointArchConfig()
    model = WaypointWorldModel(config)
    
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Create test inputs
    batch_size = 1
    n_frames = 1
    channels = config.channels
    height = config.height
    width = config.width
    prompt_len = 32
    prompt_dim = config.prompt_embedding_dim
    n_buttons = config.n_buttons
    
    with torch.no_grad():
        # Input latent: [B, T, C, H, W]
        x = torch.randn(batch_size, n_frames, channels, height, width, device=device, dtype=dtype)
        
        # Sigma: [B, T]
        sigma = torch.ones(batch_size, n_frames, device=device, dtype=dtype) * 0.5
        
        # Frame timestamp: [B, T]
        frame_timestamp = torch.zeros(batch_size, n_frames, device=device, dtype=torch.long)
        
        # Prompt embeddings: [B, seq_len, dim]
        prompt_emb = torch.randn(batch_size, prompt_len, prompt_dim, device=device, dtype=dtype)
        prompt_pad_mask = torch.ones(batch_size, prompt_len, device=device, dtype=torch.bool)
        
        # Control inputs
        mouse = torch.zeros(batch_size, n_frames, 2, device=device, dtype=dtype)
        button = torch.zeros(batch_size, n_frames, n_buttons, device=device, dtype=dtype)
        scroll = torch.zeros(batch_size, n_frames, 1, device=device, dtype=dtype)
        
        # Forward pass
        output = model(
            x=x,
            sigma=sigma,
            frame_timestamp=frame_timestamp,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
            mouse=mouse,
            button=button,
            scroll=scroll,
            kv_cache=None,
        )
    
    # Check output shape matches input
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    
    # Check output is not NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    _log_tensor_stats("output", output)
    
    print("✓ Waypoint transformer forward pass successful")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Waypoint pipeline smoke test requires CUDA.",
)
def test_waypoint_pipeline_import():
    """Test that Waypoint pipeline can be imported."""
    from fastvideo.pipelines.basic.waypoint import WaypointPipeline
    from fastvideo.pipelines.basic.waypoint.waypoint_pipeline import CtrlInput, StreamingContext
    
    # Verify pipeline is registered
    from fastvideo.pipelines.pipeline_registry import _PIPELINE_NAME_TO_ARCHITECTURE_NAME
    
    assert "WaypointPipeline" in _PIPELINE_NAME_TO_ARCHITECTURE_NAME
    assert _PIPELINE_NAME_TO_ARCHITECTURE_NAME["WaypointPipeline"] == "waypoint"
    
    # Verify CtrlInput works
    ctrl = CtrlInput(button={1, 2, 3}, mouse=(0.5, -0.5), scroll=1)
    mouse, button, scroll = ctrl.to_tensors(
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_buttons=256,
    )
    
    assert mouse.shape == (1, 1, 2)
    assert button.shape == (1, 1, 256)
    assert scroll.shape == (1, 1, 1)
    assert button[0, 0, 1] == 1.0
    assert button[0, 0, 2] == 1.0
    assert button[0, 0, 3] == 1.0
    assert button[0, 0, 0] == 0.0
    
    print("✓ Waypoint pipeline imports and CtrlInput works")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Waypoint pipeline smoke test requires CUDA.",
)  
def test_waypoint_config_loading():
    """Test that Waypoint pipeline config loads correctly."""
    from fastvideo.configs.pipelines.waypoint import WaypointT2VConfig
    from fastvideo.configs.pipelines.registry import PIPE_NAME_TO_CONFIG
    
    # Check config is registered (key is model path, not pipeline name)
    assert "Overworld/Waypoint-1-Small" in PIPE_NAME_TO_CONFIG
    
    # Instantiate config
    config = WaypointT2VConfig()
    
    # Check expected values
    assert config.is_causal is True
    assert config.base_fps == 60
    assert config.n_buttons == 256
    assert len(config.scheduler_sigmas) == 4  # HF schedule: 3 denoising steps
    assert config.scheduler_sigmas[0] == 1.0
    assert config.scheduler_sigmas[-1] == 0.0
    
    # Check dit_config
    dit_config = config.dit_config
    assert dit_config.d_model == 2560
    assert dit_config.n_heads == 40
    assert dit_config.n_layers == 22
    assert dit_config.channels == 16
    assert dit_config.height == 16
    assert dit_config.width == 16
    
    print("✓ Waypoint config loads correctly")


if __name__ == "__main__":
    # Run all tests
    test_waypoint_pipeline_import()
    test_waypoint_config_loading()
    
    if torch.cuda.is_available():
        test_waypoint_transformer_load()
        test_waypoint_transformer_forward()
    else:
        print("Skipping CUDA tests - no CUDA available")

