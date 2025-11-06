"""
Check if split Q/K/V weights correctly combine to match original QKV.
"""
import torch
from safetensors.torch import load_file

print("="*100)
print("QKV SPLIT → FUSED CONVERSION CHECK")
print("="*100)

# Load original weights
print("\n[1] Loading original weights...")
orig_state = load_file('/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit/diffusion_pytorch_model-00001-of-00006.safetensors')
orig_qkv_weight = orig_state['blocks.0.attn.qkv.weight']  # [12288, 4096]
orig_qkv_bias = orig_state['blocks.0.attn.qkv.bias']  # [12288]
print(f"  Original QKV weight: {orig_qkv_weight.shape}")
print(f"  Original QKV bias: {orig_qkv_bias.shape}")

# Load native split weights
print("\n[2] Loading native split weights...")
native_state = load_file('/mnt/fast-disks/hao_lab/shao/FastVideo/weights/longcat-native/transformer/model.safetensors')
q_weight = native_state['blocks.0.self_attn.to_q.weight']  # [4096, 4096]
k_weight = native_state['blocks.0.self_attn.to_k.weight']  # [4096, 4096]
v_weight = native_state['blocks.0.self_attn.to_v.weight']  # [4096, 4096]
q_bias = native_state['blocks.0.self_attn.to_q.bias']
k_bias = native_state['blocks.0.self_attn.to_k.bias']
v_bias = native_state['blocks.0.self_attn.to_v.bias']

print(f"  Native Q weight: {q_weight.shape}")
print(f"  Native K weight: {k_weight.shape}")
print(f"  Native V weight: {v_weight.shape}")

# Combine native weights (as done in _load_from_state_dict)
print("\n[3] Combining native weights...")
native_qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
native_qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
print(f"  Combined QKV weight: {native_qkv_weight.shape}")
print(f"  Combined QKV bias: {native_qkv_bias.shape}")

# Compare
print("\n[4] Comparing...")
weight_match = torch.allclose(orig_qkv_weight, native_qkv_weight, atol=1e-6)
bias_match = torch.allclose(orig_qkv_bias, native_qkv_bias, atol=1e-6)

print(f"  Weight match: {weight_match}")
if not weight_match:
    diff = (orig_qkv_weight - native_qkv_weight).abs()
    print(f"    Max diff: {diff.max().item():.6e}")
    print(f"    Mean diff: {diff.mean().item():.6e}")
    print(f"    Fraction different (>1e-6): {(diff > 1e-6).float().mean().item():.2%}")
    
    # Check each component
    C = 4096
    q_diff = (orig_qkv_weight[:C] - q_weight).abs()
    k_diff = (orig_qkv_weight[C:2*C] - k_weight).abs()
    v_diff = (orig_qkv_weight[2*C:] - v_weight).abs()
    print(f"    Q component max diff: {q_diff.max().item():.6e}")
    print(f"    K component max diff: {k_diff.max().item():.6e}")
    print(f"    V component max diff: {v_diff.max().item():.6e}")

print(f"  Bias match: {bias_match}")
if not bias_match:
    diff = (orig_qkv_bias - native_qkv_bias).abs()
    print(f"    Max diff: {diff.max().item():.6e}")
    print(f"    Mean diff: {diff.mean().item():.6e}")

print(f"\n✅ Comparison complete!")

