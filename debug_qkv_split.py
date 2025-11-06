"""
Verify that the QKV splitting in weight conversion is correct.
"""

import torch
from safetensors.torch import load_file
from pathlib import Path
import json

def check_qkv_split():
    print("="*80)
    print("QKV Weight Splitting Verification")
    print("="*80)
    
    # Load original weights (sharded)
    original_dir = Path("/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit")
    converted_path = Path("/mnt/fast-disks/hao_lab/shao/FastVideo/weights/longcat-native/transformer/model.safetensors")
    
    print("\n[1] Loading original weights...")
    with open(original_dir / "diffusion_pytorch_model.safetensors.index.json") as f:
        index = json.load(f)
    
    original_weights = {}
    weight_map = index["weight_map"]
    unique_files = set(weight_map.values())
    
    for shard_file in sorted(unique_files):
        shard_path = original_dir / shard_file
        shard_weights = load_file(shard_path)
        original_weights.update(shard_weights)
    
    print(f"  Loaded {len(original_weights)} original keys")
    
    print("\n[2] Loading converted weights...")
    converted_weights = load_file(converted_path)
    print(f"  Loaded {len(converted_weights)} converted keys")
    
    print("\n[3] Checking QKV split for block 0...")
    
    # Original fused QKV
    orig_qkv_weight = original_weights["blocks.0.attn.qkv.weight"]
    orig_qkv_bias = original_weights["blocks.0.attn.qkv.bias"]
    
    print(f"\n  Original fused QKV:")
    print(f"    Weight shape: {orig_qkv_weight.shape}")  # Should be [dim*3, dim]
    print(f"    Bias shape: {orig_qkv_bias.shape}")      # Should be [dim*3]
    
    # Converted split Q, K, V
    conv_q_weight = converted_weights["blocks.0.self_attn.to_q.weight"]
    conv_k_weight = converted_weights["blocks.0.self_attn.to_k.weight"]
    conv_v_weight = converted_weights["blocks.0.self_attn.to_v.weight"]
    conv_q_bias = converted_weights["blocks.0.self_attn.to_q.bias"]
    conv_k_bias = converted_weights["blocks.0.self_attn.to_k.bias"]
    conv_v_bias = converted_weights["blocks.0.self_attn.to_v.bias"]
    
    print(f"\n  Converted split Q/K/V:")
    print(f"    Q weight shape: {conv_q_weight.shape}")
    print(f"    K weight shape: {conv_k_weight.shape}")
    print(f"    V weight shape: {conv_v_weight.shape}")
    print(f"    Q bias shape: {conv_q_bias.shape}")
    print(f"    K bias shape: {conv_k_bias.shape}")
    print(f"    V bias shape: {conv_v_bias.shape}")
    
    # Verify the split
    print(f"\n[4] Verifying split correctness...")
    
    dim = orig_qkv_weight.shape[0] // 3
    print(f"    Embedding dim: {dim}")
    
    # Split the original QKV
    orig_q, orig_k, orig_v = torch.chunk(orig_qkv_weight, 3, dim=0)
    orig_q_bias, orig_k_bias, orig_v_bias = torch.chunk(orig_qkv_bias, 3, dim=0)
    
    # Compare with converted
    q_match = torch.allclose(orig_q, conv_q_weight, atol=1e-6)
    k_match = torch.allclose(orig_k, conv_k_weight, atol=1e-6)
    v_match = torch.allclose(orig_v, conv_v_weight, atol=1e-6)
    q_bias_match = torch.allclose(orig_q_bias, conv_q_bias, atol=1e-6)
    k_bias_match = torch.allclose(orig_k_bias, conv_k_bias, atol=1e-6)
    v_bias_match = torch.allclose(orig_v_bias, conv_v_bias, atol=1e-6)
    
    print(f"\n  Weight comparison:")
    print(f"    Q weight match: {q_match} {'✓' if q_match else '✗'}")
    print(f"    K weight match: {k_match} {'✓' if k_match else '✗'}")
    print(f"    V weight match: {v_match} {'✓' if v_match else '✗'}")
    print(f"    Q bias match: {q_bias_match} {'✓' if q_bias_match else '✗'}")
    print(f"    K bias match: {k_bias_match} {'✓' if k_bias_match else '✗'}")
    print(f"    V bias match: {v_bias_match} {'✓' if v_bias_match else '✗'}")
    
    if all([q_match, k_match, v_match, q_bias_match, k_bias_match, v_bias_match]):
        print(f"\n  ✓ QKV splitting is CORRECT!")
    else:
        print(f"\n  ✗ QKV splitting has ERRORS!")
        
        # Show differences
        if not q_match:
            diff = torch.abs(orig_q - conv_q_weight)
            print(f"    Q weight max diff: {diff.max().item():.2e}")
        if not k_match:
            diff = torch.abs(orig_k - conv_k_weight)
            print(f"    K weight max diff: {diff.max().item():.2e}")
        if not v_match:
            diff = torch.abs(orig_v - conv_v_weight)
            print(f"    V weight max diff: {diff.max().item():.2e}")
    
    print("\n[5] Testing QKV application...")
    # Create a test input
    batch_size = 2
    seq_len = 100
    hidden_dim = orig_qkv_weight.shape[1]
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Original fused way
    qkv_fused = torch.nn.functional.linear(x, orig_qkv_weight, orig_qkv_bias)
    q_fused, k_fused, v_fused = torch.chunk(qkv_fused, 3, dim=-1)
    
    # Converted split way
    q_split = torch.nn.functional.linear(x, conv_q_weight, conv_q_bias)
    k_split = torch.nn.functional.linear(x, conv_k_weight, conv_k_bias)
    v_split = torch.nn.functional.linear(x, conv_v_weight, conv_v_bias)
    
    # Compare
    q_output_match = torch.allclose(q_fused, q_split, atol=1e-5)
    k_output_match = torch.allclose(k_fused, k_split, atol=1e-5)
    v_output_match = torch.allclose(v_fused, v_split, atol=1e-5)
    
    print(f"  Q output match: {q_output_match} {'✓' if q_output_match else '✗'}")
    print(f"  K output match: {k_output_match} {'✓' if k_output_match else '✗'}")
    print(f"  V output match: {v_output_match} {'✓' if v_output_match else '✗'}")
    
    if all([q_output_match, k_output_match, v_output_match]):
        print(f"\n  ✓ QKV application produces IDENTICAL outputs!")
    else:
        print(f"\n  ✗ QKV application produces DIFFERENT outputs!")
        if not q_output_match:
            diff = torch.abs(q_fused - q_split)
            print(f"    Q max diff: {diff.max().item():.2e}, mean: {diff.mean().item():.2e}")
        if not k_output_match:
            diff = torch.abs(k_fused - k_split)
            print(f"    K max diff: {diff.max().item():.2e}, mean: {diff.mean().item():.2e}")
        if not v_output_match:
            diff = torch.abs(v_fused - v_split)
            print(f"    V max diff: {diff.max().item():.2e}, mean: {diff.mean().item():.2e}")

if __name__ == "__main__":
    check_qkv_split()


