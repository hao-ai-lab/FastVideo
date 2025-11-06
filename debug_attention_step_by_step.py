"""
Debug self-attention step-by-step to find exact divergence point.
"""
import torch
import numpy as np
import os
import sys

# Add LongCat to path
sys.path.insert(0, '/mnt/fast-disks/hao_lab/shao/LongCat-Video')

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_grad_enabled(False)

print("="*100)
print("STEP-BY-STEP SELF-ATTENTION DEBUG")
print("="*100)

# Load the saved inputs from layer comparison
native_acts = torch.load('/mnt/fast-disks/hao_lab/shao/FastVideo/outputs/debug_layers/native_activations.pt')
original_acts = torch.load('/mnt/fast-disks/hao_lab/shao/FastVideo/outputs/debug_layers/orig_activations.pt')

# Get patch embeddings (input to block 0)
x_orig = original_acts['patch_embed'].cuda()
x_native = native_acts['patch_embed'].cuda()

print(f"\n[1] Input to Block 0 Self-Attention:")
print(f"  Original: mean={x_orig.mean():.6f}, std={x_orig.std():.6f}")
print(f"  Native:   mean={x_native.mean():.6f}, std={x_native.std():.6f}")
print(f"  Match: {torch.allclose(x_orig, x_native, atol=1e-5)}")

# Load models
print(f"\n[2] Loading models...")

# Original model
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalModel
original_model = OriginalModel.from_pretrained(
    '/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit',
    torch_dtype='bfloat16',
    cp_split_hw=[1, 1],
).cuda().eval()
print("  ✓ Original loaded")

# Native model
from fastvideo.models.dits.longcat import LongCatTransformer3DModel
from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from safetensors.torch import load_file
import json

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

init_distributed_environment()
initialize_model_parallel()

transformer_path = '/mnt/fast-disks/hao_lab/shao/FastVideo/weights/longcat-native/transformer'
with open(f'{transformer_path}/config.json') as f:
    config_dict = json.load(f)

model_config = LongCatVideoConfig()
native_model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
state_dict = load_file(f'{transformer_path}/model.safetensors')
native_model.load_state_dict(state_dict, strict=False)
native_model = native_model.cuda().eval()
print("  ✓ Native loaded")

# Get block 0 self-attention modules
orig_attn = original_model.blocks[0].attn
native_attn = native_model.blocks[0].self_attn

latent_shape = (9, 30, 52)
B, N, C = x_orig.shape

print(f"\n[3] QKV Projection:")
# Original
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    qkv_orig = orig_attn.qkv(x_orig)  # [B, N, 3*C]
    qkv_orig = qkv_orig.view(B, N, 3, orig_attn.num_heads, orig_attn.head_dim)
    qkv_orig = qkv_orig.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
    q_orig, k_orig, v_orig = qkv_orig.unbind(0)

print(f"  Original QKV shape: Q={q_orig.shape}, K={k_orig.shape}, V={v_orig.shape}")
print(f"  Original Q: mean={q_orig.mean():.6f}, std={q_orig.std():.6f}")

# Native
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    qkv_native, _ = native_attn.qkv(x_native)  # [B, N, 3*C]
    qkv_native = qkv_native.view(B, N, 3, native_attn.num_heads, native_attn.head_dim)
    qkv_native = qkv_native.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
    q_native, k_native, v_native = qkv_native.unbind(0)

print(f"  Native QKV shape: Q={q_native.shape}, K={k_native.shape}, V={v_native.shape}")
print(f"  Native Q: mean={q_native.mean():.6f}, std={q_native.std():.6f}")

print(f"\n  Q match: {torch.allclose(q_orig, q_native, atol=1e-3, rtol=1e-2)}")
print(f"  Q diff: max={(q_orig - q_native).abs().max():.6f}, mean={(q_orig - q_native).abs().mean():.6f}")

print(f"\n[4] Q/K Normalization:")
# Original
q_norm_orig = orig_attn.q_norm(q_orig)
k_norm_orig = orig_attn.k_norm(k_orig)
print(f"  Original Q after norm: mean={q_norm_orig.mean():.6f}, std={q_norm_orig.std():.6f}")

# Native  
q_norm_native = native_attn.q_norm(q_native)
k_norm_native = native_attn.k_norm(k_native)
print(f"  Native Q after norm: mean={q_norm_native.mean():.6f}, std={q_norm_native.std():.6f}")

print(f"\n  Q_norm match: {torch.allclose(q_norm_orig, q_norm_native, atol=1e-3, rtol=1e-2)}")
print(f"  Q_norm diff: max={(q_norm_orig - q_norm_native).abs().max():.6f}, mean={(q_norm_orig - q_norm_native).abs().mean():.6f}")

print(f"\n[5] RoPE Application:")
# Original
q_rope_orig, k_rope_orig = orig_attn.rope_3d(q_norm_orig, k_norm_orig, latent_shape)
print(f"  Original Q after RoPE: mean={q_rope_orig.mean():.6f}, std={q_rope_orig.std():.6f}")

# Native
q_rope_native, k_rope_native = native_attn.rope_3d(q_norm_native, k_norm_native, grid_size=latent_shape)
print(f"  Native Q after RoPE: mean={q_rope_native.mean():.6f}, std={q_rope_native.std():.6f}")

print(f"\n  Q_rope match: {torch.allclose(q_rope_orig, q_rope_native, atol=1e-3, rtol=1e-2)}")
print(f"  Q_rope diff: max={(q_rope_orig - q_rope_native).abs().max():.6f}, mean={(q_rope_orig - q_rope_native).abs().mean():.6f}")

print(f"\n[6] Attention computation:")
print(f"  Original backend: FlashAttention2")
print(f"  Native backend: {native_attn.attn.backend}")

# Skip actual attention computation (too complex to debug here)
print(f"\n✅ Debug complete! Check where divergence starts.")

