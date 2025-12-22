#!/usr/bin/env python3
"""
Standalone script to compare FastVideo GameCraft with original implementation.

This script loads both implementations in the same environment and compares outputs.

Usage:
    python compare_with_original.py
    
Requirements:
    1. Original repo at: fastvideo/models/Hunyuan-GameCraft-1.0-main/
    2. Model weights available (for loading state dicts)
"""

import os
import sys
import torch

# Add original implementation to path
ORIGINAL_PATH = os.path.join(os.path.dirname(__file__), 
                             "fastvideo/models/Hunyuan-GameCraft-1.0-main")
if ORIGINAL_PATH not in sys.path:
    sys.path.insert(0, ORIGINAL_PATH)

print("="*70)
print("FastVideo vs Original GameCraft Comparison")
print("="*70)
print()

# Try to import both implementations
print("Step 1: Loading implementations...")
try:
    # Original
    from hymm_sp.modules.models import HYVideoDiffusionTransformer
    print("  ✓ Original GameCraft implementation loaded")
except ImportError as e:
    print(f"  ✗ Could not load original implementation: {e}")
    print(f"    Make sure the official repo is at: {ORIGINAL_PATH}")
    sys.exit(1)

try:
    # FastVideo
    from fastvideo.models.dits.hunyuangamecraft import HunyuanGameCraftTransformer3DModel
    from fastvideo.configs.models.dits import HunyuanGameCraftConfig
    print("  ✓ FastVideo GameCraft implementation loaded")
except ImportError as e:
    print(f"  ✗ Could not load FastVideo implementation: {e}")
    sys.exit(1)

print()

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Step 2: Using device: {device}")
print()

# Create models
print("Step 3: Initializing models...")

# Original model
class OriginalArgs:
    text_states_dim = 4096
    text_states_dim_2 = 768
    text_projection = "single_refiner"
    use_attention_mask = False

original_args = OriginalArgs()

try:
    original_model = HYVideoDiffusionTransformer(
        args=original_args,
        patch_size=[1, 2, 2],
        in_channels=16,
        out_channels=16,
        hidden_size=3072,
        mlp_width_ratio=4.0,
        num_heads=24,
        depth_double_blocks=20,
        depth_single_blocks=40,
        rope_dim_list=[16, 56, 56],
        qkv_bias=True,
        qk_norm=True,
        qk_norm_type='rms',
        guidance_embed=False,
        camera_in_channels=6,
        camera_down_coef=8,
    ).to(device)
    original_model.eval()
    print("  ✓ Original model initialized")
except Exception as e:
    print(f"  ✗ Failed to initialize original model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# FastVideo model
try:
    config = HunyuanGameCraftConfig()
    fastvideo_model = HunyuanGameCraftTransformer3DModel(config, hf_config={}).to(device)
    fastvideo_model.eval()
    print("  ✓ FastVideo model initialized")
except Exception as e:
    print(f"  ✗ Failed to initialize FastVideo model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Create test inputs
print("Step 4: Creating test inputs...")
batch_size = 1
text_seq_len = 256
torch.manual_seed(42)

# Video latents
hidden_states = torch.randn(batch_size, 16, 9, 88, 152, device=device, dtype=torch.float32)

# Text embeddings for original
text_states_orig = torch.randn(batch_size, text_seq_len, 4096, device=device, dtype=torch.float32)

# Text embeddings for FastVideo (with pooled token at position 0)
text_states_fv = torch.randn(batch_size, text_seq_len + 1, 4096, device=device, dtype=torch.float32)
text_states_fv[:, 0, 768:] = 0

# Pooled embeddings
text_states_2 = torch.randn(batch_size, 768, device=device, dtype=torch.float32)

# Camera latents
camera_latents = torch.randn(batch_size, 9, 6, 704, 1216, device=device, dtype=torch.float32)

# Timestep
timestep = torch.tensor([500.0], device=device, dtype=torch.float32)

# Rotary embeddings
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
tt, th, tw = 9, 88, 152
freqs_cos, freqs_sin = get_rotary_pos_embed((tt, th, tw), 3072, 24, [16, 56, 56], 256)
freqs_cos = freqs_cos.to(device)
freqs_sin = freqs_sin.to(device)

print(f"  Video latents: {hidden_states.shape}")
print(f"  Text states: {text_states_orig.shape}")
print(f"  Camera latents: {camera_latents.shape}")
print()

# Run forward passes
print("Step 5: Running forward passes...")

# Original
print("  Running original implementation...")
with torch.no_grad():
    try:
        output_original = original_model(
            x=hidden_states.clone(),
            t=timestep.clone(),
            text_states=text_states_orig.clone(),
            text_states_2=text_states_2.clone(),
            text_mask=None,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=None,
            cam_latents=camera_latents.clone(),
            use_sage=False,
        )
        
        if isinstance(output_original, dict):
            output_original = output_original['x']
        
        print(f"    ✓ Original output: {output_original.shape}")
    except Exception as e:
        print(f"    ✗ Original forward failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# FastVideo
print("  Running FastVideo implementation...")
with torch.no_grad():
    try:
        output_fastvideo = fastvideo_model(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=text_states_fv.clone(),
            timestep=timestep.clone(),
            camera_latents=camera_latents.clone(),
        )
        print(f"    ✓ FastVideo output: {output_fastvideo.shape}")
    except Exception as e:
        print(f"    ✗ FastVideo forward failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print()

# Compare
print("Step 6: Comparing outputs...")
abs_diff = (output_original - output_fastvideo).abs()
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()

rel_error = abs_diff / (output_original.abs() + 1e-8)
max_rel_error = rel_error.max().item()
mean_rel_error = rel_error.mean().item()

print()
print("="*70)
print("RESULTS")
print("="*70)
print(f"Original output range: [{output_original.min():.6f}, {output_original.max():.6f}]")
print(f"FastVideo output range: [{output_fastvideo.min():.6f}, {output_fastvideo.max():.6f}]")
print()
print("Absolute Difference:")
print(f"  Max:  {max_diff:.2e}")
print(f"  Mean: {mean_diff:.2e}")
print()
print("Relative Error:")
print(f"  Max:  {max_rel_error:.2e}")
print(f"  Mean: {mean_rel_error:.2e}")
print("="*70)
print()

# Verdict
tolerance = 1e-5
if max_diff < tolerance:
    print(f"✓✓✓ PASSED: Max diff {max_diff:.2e} < {tolerance:.2e}")
    print("✓ FastVideo GameCraft matches original implementation!")
    sys.exit(0)
else:
    print(f"⚠ Max diff {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
    print()
    print("Possible reasons:")
    print("  1. Random initialization (expected without loading same weights)")
    print("  2. Implementation differences in some layers")
    print("  3. Numerical precision differences")
    print()
    print("Next steps:")
    print("  1. Load same checkpoint weights into both models")
    print("  2. Debug layer-by-layer to find discrepancies")
    print("  3. Check parameter name mappings")
    sys.exit(1)

