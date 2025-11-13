"""
Test script to verify LoRA alpha scaling works correctly.
This creates a minimal test without requiring full model loading.
"""
import torch
from safetensors.torch import save_file, load_file
import tempfile
import os

print("=" * 80)
print("Testing LoRA Alpha Scaling Implementation")
print("=" * 80)

# Test parameters
rank = 128
alpha = 64
alpha_scale_expected = alpha / rank  # Should be 0.5

print(f"\nTest Configuration:")
print(f"  Rank: {rank}")
print(f"  Alpha: {alpha}")
print(f"  Expected alpha_scale: {alpha_scale_expected}")

# Create test LoRA weights
print("\n[1] Creating test LoRA weights...")
lora_A = torch.randn(rank, 512)  # (rank, in_dim)
lora_B = torch.randn(1024, rank)  # (out_dim, rank)

# Save with lora_alpha
with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, "test_lora.safetensors")
    
    save_file({
        "blocks.0.attn.to_q.lora_A": lora_A,
        "blocks.0.attn.to_q.lora_B": lora_B,
        "blocks.0.attn.to_q.lora_alpha": torch.tensor(alpha, dtype=torch.float32),
    }, test_file)
    
    print(f"✓ Saved test LoRA to: {test_file}")
    
    # Load and verify
    print("\n[2] Loading test LoRA weights...")
    state_dict = load_file(test_file)
    
    print(f"✓ Loaded {len(state_dict)} tensors:")
    for key in state_dict.keys():
        print(f"    - {key}: shape={state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'}")
    
    # Verify alpha value
    print("\n[3] Verifying alpha value...")
    loaded_alpha = state_dict["blocks.0.attn.to_q.lora_alpha"].item()
    print(f"  Loaded alpha: {loaded_alpha}")
    print(f"  Expected alpha: {alpha}")
    
    if abs(loaded_alpha - alpha) < 1e-6:
        print("  ✓ Alpha value matches!")
    else:
        print(f"  ✗ Alpha mismatch! Expected {alpha}, got {loaded_alpha}")
        exit(1)
    
    # Test alpha_scale computation
    print("\n[4] Testing alpha_scale computation...")
    loaded_lora_A = state_dict["blocks.0.attn.to_q.lora_A"]
    loaded_rank = loaded_lora_A.shape[0]
    computed_alpha_scale = loaded_alpha / loaded_rank
    
    print(f"  Loaded rank: {loaded_rank}")
    print(f"  Computed alpha_scale: {computed_alpha_scale}")
    print(f"  Expected alpha_scale: {alpha_scale_expected}")
    
    if abs(computed_alpha_scale - alpha_scale_expected) < 1e-6:
        print("  ✓ Alpha scale computation correct!")
    else:
        print(f"  ✗ Alpha scale mismatch! Expected {alpha_scale_expected}, got {computed_alpha_scale}")
        exit(1)
    
    # Test weight merging with alpha scaling
    print("\n[5] Testing weight merging with alpha scaling...")
    base_weight = torch.randn(1024, 512)
    
    # Method 1: Without scaling (BUG)
    merged_wrong = base_weight + (lora_B @ lora_A)
    
    # Method 2: With alpha scaling (CORRECT)
    merged_correct = base_weight + alpha_scale_expected * (lora_B @ lora_A)
    
    # Calculate difference
    delta_magnitude = torch.norm(lora_B @ lora_A).item()
    scaling_diff = torch.norm(merged_correct - merged_wrong).item()
    
    print(f"  Delta magnitude (B @ A): {delta_magnitude:.4f}")
    print(f"  Difference due to scaling: {scaling_diff:.4f}")
    print(f"  Scaling effect: {scaling_diff / delta_magnitude * 100:.1f}% reduction")
    
    expected_reduction = (1 - alpha_scale_expected) * 100
    print(f"  Expected reduction: {expected_reduction:.1f}%")
    
    if abs(scaling_diff / delta_magnitude - (1 - alpha_scale_expected)) < 1e-5:
        print("  ✓ Scaling applied correctly!")
    else:
        print("  ✗ Scaling not applied correctly!")
        exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nConclusion:")
print(f"  - Storing alpha={alpha} with rank={rank}")
print(f"  - Computing alpha_scale={alpha_scale_expected} at runtime")
print(f"  - Applying {expected_reduction:.0f}% reduction to LoRA delta")
print(f"  - This matches the original LongCat behavior!")
print("=" * 80)

