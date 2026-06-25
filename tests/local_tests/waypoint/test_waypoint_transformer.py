# SPDX-License-Identifier: Apache-2.0
"""
Structural parity test for the Waypoint-1-Small transformer.

Loads the official checkpoint into FastVideo's ``WaypointWorldModel`` with
``strict=True`` to prove the architecture (key names and tensor shapes) matches
the published weights exactly, and runs a forward pass for output validity.

Run with:
    pytest tests/local_tests/waypoint/test_waypoint_transformer.py -v

Requires the model weights (set ``WAYPOINT_MODEL_PATH`` to override):
    - models/Waypoint-1-Small-Diffusers/transformer/diffusion_pytorch_model.safetensors
"""

import os
import pytest
import torch

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.forward_context import set_forward_context

# Skip if weights not available
WEIGHTS_PATH = os.environ.get("WAYPOINT_MODEL_PATH",
                              "models/Waypoint-1-Small-Diffusers")
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
        """Official checkpoint loads into the native model with strict=True.

        Applies the config's ``param_names_mapping`` (the same checkpoint->FastVideo
        key remap the production loader uses, e.g. mlp.fc1->mlp.fc_in) so this is a
        true structural-parity assertion on key names and tensor shapes.
        """
        import re

        import safetensors.torch as st
        from fastvideo.models.dits.waypoint_transformer import WaypointWorldModel

        weights_file = os.path.join(WEIGHTS_PATH, "transformer", "diffusion_pytorch_model.safetensors")
        official_state_dict = st.load_file(weights_file)

        mapping = waypoint_config.param_names_mapping
        remapped = {}
        for key, tensor in official_state_dict.items():
            new_key = key
            for pat, repl in mapping.items():
                new_key = re.sub(pat, repl, new_key)
            remapped[new_key] = tensor

        model = WaypointWorldModel(waypoint_config)
        fastvideo_keys = set(model.state_dict().keys())
        official_keys = set(remapped.keys())
        missing = official_keys - fastvideo_keys
        unexpected = fastvideo_keys - official_keys
        print(f"\nOfficial keys: {len(official_keys)}  FastVideo keys: {len(fastvideo_keys)}")
        print(f"missing (in ckpt, not model): {sorted(missing)[:8]}")
        print(f"unexpected (in model, not ckpt): {sorted(unexpected)[:8]}")

        model.load_state_dict(remapped, strict=True)
        print("\n✓ Official weights loaded with strict=True after param_names_mapping.")
    
    def test_forward_pass(self, waypoint_config, device, distributed_setup):
        """Test that forward pass runs without errors.
        
        Uses distributed_setup fixture to initialize distributed environment (SP=1).
        """
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
        
        # Forward pass with forward context (required for attention layers)
        attn_metadata = SDPAMetadata(current_timestep=0, attn_mask=None)
        with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=attn_metadata,
            forward_batch=None,
        ):
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

