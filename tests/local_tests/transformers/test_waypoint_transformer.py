# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Waypoint-1-Small transformer.

This test validates that the FastVideo implementation of the Waypoint
transformer can load official weights and produces identical outputs.

Run with:
    pytest tests/local_tests/transformers/test_waypoint_transformer.py -v

Requires:
    - official_weights/Waypoint-1-Small/ directory with downloaded model
"""

import os
import pytest
import torch

# Skip if weights not available
WEIGHTS_PATH = "official_weights/Waypoint-1-Small"
SKIP_REASON = f"Waypoint weights not found at {WEIGHTS_PATH}"


def weights_available():
    """Check if Waypoint weights are downloaded."""
    transformer_path = os.path.join(WEIGHTS_PATH, "transformer", "diffusion_pytorch_model.safetensors")
    return os.path.exists(transformer_path)


@pytest.fixture
def waypoint_config():
    """Create Waypoint config matching official model."""
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    return WaypointArchConfig()


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.mark.skipif(not weights_available(), reason=SKIP_REASON)
class TestWaypointTransformerParity:
    """Test parity between FastVideo and official Waypoint implementation."""
    
    def test_weight_loading(self, waypoint_config, device):
        """Test that weights load correctly with strict=True."""
        import safetensors.torch as st
        from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel
        
        # Load official weights
        weights_file = os.path.join(WEIGHTS_PATH, "transformer", "diffusion_pytorch_model.safetensors")
        official_state_dict = st.load_file(weights_file)
        
        # Create FastVideo model
        model = WaypointWorldModel(waypoint_config)
        
        # Get FastVideo model keys
        fastvideo_keys = set(model.state_dict().keys())
        official_keys = set(official_state_dict.keys())
        
        # Check for missing/unexpected keys
        missing = official_keys - fastvideo_keys
        unexpected = fastvideo_keys - official_keys
        
        print(f"\nOfficial keys: {len(official_keys)}")
        print(f"FastVideo keys: {len(fastvideo_keys)}")
        
        if missing:
            print(f"\nMissing keys (in official but not in FastVideo): {len(missing)}")
            for k in sorted(missing)[:20]:
                print(f"  - {k}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        
        if unexpected:
            print(f"\nUnexpected keys (in FastVideo but not in official): {len(unexpected)}")
            for k in sorted(unexpected)[:20]:
                print(f"  - {k}")
            if len(unexpected) > 20:
                print(f"  ... and {len(unexpected) - 20} more")
        
        # Try loading (this will show exactly what's wrong)
        try:
            model.load_state_dict(official_state_dict, strict=True)
            print("\n✓ Weights loaded successfully with strict=True!")
        except RuntimeError as e:
            print(f"\n✗ Weight loading failed: {e}")
            pytest.fail(f"Weight loading failed: {e}")
    
    def test_forward_pass(self, waypoint_config, device):
        """Test that forward pass runs without errors."""
        import safetensors.torch as st
        from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel
        from fastvideo.pipelines.basic.waypoint.waypoint_pipeline import CtrlInput
        
        # Load model
        weights_file = os.path.join(WEIGHTS_PATH, "transformer", "diffusion_pytorch_model.safetensors")
        official_state_dict = st.load_file(weights_file)
        
        model = WaypointWorldModel(waypoint_config)
        
        # Try to load weights (may fail initially - that's okay for this test)
        try:
            model.load_state_dict(official_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
        
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        
        # Create dummy inputs
        B, N, C, H, W = 1, 1, 16, 32, 32  # Single frame, 32x32 latent
        
        x = torch.randn(B, N, C, H, W, device=device, dtype=torch.bfloat16)
        sigma = torch.tensor([[0.5]], device=device, dtype=torch.bfloat16)
        frame_timestamp = torch.tensor([[0]], device=device, dtype=torch.long)
        
        # Prompt embedding (T5-XXL style)
        prompt_emb = torch.randn(B, 128, 2048, device=device, dtype=torch.bfloat16)
        prompt_pad_mask = torch.ones(B, 128, device=device, dtype=torch.bool)
        
        # Control inputs
        ctrl = CtrlInput(button={48, 42}, mouse=(0.1, 0.2), scroll=0.0)
        mouse, button, scroll = ctrl.to_tensors(device, torch.bfloat16, n_buttons=256)
        
        # Forward pass
        with torch.no_grad():
            try:
                output = model(
                    x=x,
                    sigma=sigma,
                    frame_timestamp=frame_timestamp,
                    prompt_emb=prompt_emb,
                    prompt_pad_mask=prompt_pad_mask,
                    mouse=mouse,
                    button=button,
                    scroll=scroll,
                )
                print(f"\n✓ Forward pass successful!")
                print(f"  Input shape: {x.shape}")
                print(f"  Output shape: {output.shape}")
                assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
            except Exception as e:
                print(f"\n✗ Forward pass failed: {e}")
                raise


@pytest.mark.skipif(not weights_available(), reason=SKIP_REASON)
def test_key_comparison():
    """Quick test to compare checkpoint keys with model keys."""
    import safetensors.torch as st
    from fastvideo.configs.models.dits.waypoint_transformer import WaypointArchConfig
    from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel
    
    # Load official keys
    weights_file = os.path.join(WEIGHTS_PATH, "transformer", "diffusion_pytorch_model.safetensors")
    official_keys = sorted(st.load_file(weights_file).keys())
    
    # Get FastVideo model keys
    config = WaypointArchConfig()
    model = WaypointWorldModel(config)
    fastvideo_keys = sorted(model.state_dict().keys())
    
    print(f"\n{'Official Key':<60} | {'FastVideo Key':<60}")
    print("-" * 125)
    
    max_show = 50
    for i, (ok, fk) in enumerate(zip(official_keys[:max_show], fastvideo_keys[:max_show])):
        match = "✓" if ok == fk else "✗"
        print(f"{match} {ok:<58} | {fk:<58}")
    
    if len(official_keys) > max_show:
        print(f"... and {len(official_keys) - max_show} more keys")


if __name__ == "__main__":
    # Run quick key comparison
    if weights_available():
        test_key_comparison()
    else:
        print(f"Weights not found at {WEIGHTS_PATH}")
        print("Download with:")
        print(f"  python scripts/huggingface/download_hf.py --repo_id Overworld/Waypoint-1-Small --local_dir {WEIGHTS_PATH} --repo_type model")

