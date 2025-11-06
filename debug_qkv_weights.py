"""
Check if QKV weights match between original and converted.
"""
import torch
from safetensors.torch import load_file

print("="*100)
print("QKV WEIGHT COMPARISON")
print("="*100)

# Load original weights
print("\n[1] Loading original weights...")
orig_state = load_file('/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit/diffusion_pytorch_model-00001-of-00006.safetensors')
print(f"  Loaded {len(orig_state)} keys")

# Load native/converted weights  
print("\n[2] Loading converted weights...")
native_state = load_file('/mnt/fast-disks/hao_lab/shao/FastVideo/weights/longcat-native/transformer/model.safetensors')
print(f"  Loaded {len(native_state)} keys")

# Check Block 0 self-attention QKV
print("\n[3] Block 0 Self-Attention QKV:")
print(f"  Original qkv.weight shape: {orig_state['blocks.0.attn.qkv.weight'].shape}")
print(f"  Original qkv.bias shape: {orig_state['blocks.0.attn.qkv.bias'].shape}")

# Native has split weights that should be combined
native_qkv_weight = native_state['blocks.0.self_attn.qkv.weight']
native_qkv_bias = native_state['blocks.0.self_attn.qkv.bias']
print(f"  Native qkv.weight shape: {native_qkv_weight.shape}")
print(f"  Native qkv.bias shape: {native_qkv_bias.shape}")

# Compare
orig_qkv_weight = orig_state['blocks.0.attn.qkv.weight']
orig_qkv_bias = orig_state['blocks.0.attn.qkv.bias']

weight_match = torch.allclose(orig_qkv_weight, native_qkv_weight, atol=1e-6)
bias_match = torch.allclose(orig_qkv_bias, native_qkv_bias, atol=1e-6)

print(f"\n[4] Weight comparison:")
print(f"  QKV weight match: {weight_match}")
if not weight_match:
    diff = (orig_qkv_weight - native_qkv_weight).abs()
    print(f"    Max diff: {diff.max().item():.6e}")
    print(f"    Mean diff: {diff.mean().item():.6e}")
    print(f"    Num different: {(diff > 1e-6).sum().item()} / {diff.numel()}")

print(f"  QKV bias match: {bias_match}")
if not bias_match:
    diff = (orig_qkv_bias - native_qkv_bias).abs()
    print(f"    Max diff: {diff.max().item():.6e}")
    print(f"    Mean diff: {diff.mean().item():.6e}")
    print(f"    Num different: {(diff > 1e-6).sum().item()} / {diff.numel()}")

# Check RMSNorm weights
print(f"\n[5] Block 0 Self-Attention RMSNorm:")
print(f"  Original q_norm.weight: {orig_state['blocks.0.attn.q_norm.weight'].shape}")
print(f"  Original k_norm.weight: {orig_state['blocks.0.attn.k_norm.weight'].shape}")
print(f"  Native q_norm.weight: {native_state['blocks.0.self_attn.q_norm.weight'].shape}")
print(f"  Native k_norm.weight: {native_state['blocks.0.self_attn.k_norm.weight'].shape}")

q_norm_match = torch.allclose(orig_state['blocks.0.attn.q_norm.weight'], 
                               native_state['blocks.0.self_attn.q_norm.weight'], atol=1e-6)
k_norm_match = torch.allclose(orig_state['blocks.0.attn.k_norm.weight'],
                               native_state['blocks.0.self_attn.k_norm.weight'], atol=1e-6)

print(f"  Q norm weight match: {q_norm_match}")
print(f"  K norm weight match: {k_norm_match}")

print(f"\n✅ Comparison complete!")

