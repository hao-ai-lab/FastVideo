"""
Debug internal operations within a single transformer block.
"""
import torch
import sys
import os
import numpy as np

sys.path.insert(0, '/mnt/fast-disks/hao_lab/shao/LongCat-Video')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_grad_enabled(False)

print("="*100)
print("DETAILED BLOCK 0 INTERNAL COMPARISON")
print("="*100)

# Load saved activations
output_dir = '/mnt/fast-disks/hao_lab/shao/FastVideo/outputs/debug_layers'
orig_acts = torch.load(f'{output_dir}/orig_activations.pt')
native_acts = torch.load(f'{output_dir}/native_activations.pt')

print(f"\n[1] Input to Block 0:")
x_orig = orig_acts['patch_embed'].cuda()
x_native = native_acts['patch_embed'].cuda()
print(f"  Match: {torch.allclose(x_orig, x_native, atol=1e-5)}")
print(f"  Max diff: {(x_orig - x_native).abs().max():.6e}")

print(f"\n[2] Timestep embedding:")
t_orig = orig_acts['time_embed'].cuda()
t_native = native_acts['time_embed'].cuda()
print(f"  Original shape: {t_orig.shape}, mean={t_orig.mean():.6f}, std={t_orig.std():.6f}")
print(f"  Native shape: {t_native.shape}, mean={t_native.mean():.6f}, std={t_native.std():.6f}")
print(f"  Max diff: {(t_orig - t_native).abs().max():.6e}")
print(f"  Mean diff: {(t_orig - t_native).abs().mean():.6e}")

print(f"\n[3] Caption embedding:")
cap_orig = orig_acts['caption_embed'].cuda()
cap_native = native_acts['caption_embed'].cuda()
print(f"  Original shape: {cap_orig.shape}")
print(f"  Native shape: {cap_native.shape}")
if cap_orig.shape != cap_native.shape:
    print(f"  ⚠️  Shape mismatch! Original has extra batch dim")
    cap_orig = cap_orig.squeeze(0)
    print(f"  After squeeze: {cap_orig.shape}")
print(f"  Match: {torch.allclose(cap_orig, cap_native, atol=1e-5)}")
print(f"  Max diff: {(cap_orig - cap_native).abs().max():.6e}")

print(f"\n[4] Block 0 Self-Attention output:")
sa_orig = orig_acts['block_0_self_attn'].cuda()
sa_native = native_acts['block_0_self_attn'].cuda()
print(f"  Original: mean={sa_orig.mean():.6f}, std={sa_orig.std():.6f}")
print(f"  Native:   mean={sa_native.mean():.6f}, std={sa_native.std():.6f}")
print(f"  Max diff: {(sa_orig - sa_native).abs().max():.6e}")
print(f"  Mean diff: {(sa_orig - sa_native).abs().mean():.6e}")
print(f"  Relative mean diff: {((sa_orig - sa_native).abs().mean() / sa_orig.abs().mean()):.2%}")

print(f"\n[5] Block 0 Cross-Attention output:")
ca_orig = orig_acts['block_0_cross_attn'].cuda()
ca_native = native_acts['block_0_cross_attn'].cuda()
print(f"  Original: mean={ca_orig.mean():.6f}, std={ca_orig.std():.6f}")
print(f"  Native:   mean={ca_native.mean():.6f}, std={ca_native.std():.6f}")
print(f"  Max diff: {(ca_orig - ca_native).abs().max():.6e}")
print(f"  Mean diff: {(ca_orig - ca_native).abs().mean():.6e}")

print(f"\n[6] Block 0 FFN output:")
ffn_orig = orig_acts['block_0_ffn'].cuda()
ffn_native = native_acts['block_0_ffn'].cuda()
print(f"  Original: mean={ffn_orig.mean():.6f}, std={ffn_orig.std():.6f}")
print(f"  Native:   mean={ffn_native.mean():.6f}, std={ffn_native.std():.6f}")
print(f"  Max diff: {(ffn_orig - ffn_native).abs().max():.6e}")
print(f"  Mean diff: {(ffn_orig - ffn_native).abs().mean():.6e}")

print(f"\n[7] Block 0 Final output:")
b0_orig = orig_acts['block_0'].cuda()
b0_native = native_acts['block_0'].cuda()
print(f"  Original: mean={b0_orig.mean():.6f}, std={b0_orig.std():.6f}, range=[{b0_orig.min():.2f}, {b0_orig.max():.2f}]")
print(f"  Native:   mean={b0_native.mean():.6f}, std={b0_native.std():.6f}, range=[{b0_native.min():.2f}, {b0_native.max():.2f}]")
print(f"  Max diff: {(b0_orig - b0_native).abs().max():.6e}")
print(f"  Mean diff: {(b0_orig - b0_native).abs().mean():.6e}")

# Check if magnitudes are growing unexpectedly
print(f"\n[8] Magnitude growth through Block 0:")
print(f"  Input magnitude: {x_orig.abs().mean():.6f}")
print(f"  Self-attn out:   Orig={sa_orig.abs().mean():.6f}, Native={sa_native.abs().mean():.6f}")
print(f"  Cross-attn out:  Orig={ca_orig.abs().mean():.6f}, Native={ca_native.abs().mean():.6f}")
print(f"  FFN out:         Orig={ffn_orig.abs().mean():.6f}, Native={ffn_native.abs().mean():.6f}")
print(f"  Block out:       Orig={b0_orig.abs().mean():.6f}, Native={b0_native.abs().mean():.6f}")

print(f"\n✅ Analysis complete!")

