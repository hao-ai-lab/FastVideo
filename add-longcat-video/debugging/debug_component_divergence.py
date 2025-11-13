"""
Detailed component-level divergence analysis for a single transformer block.

This script compares every single component within a transformer block:
- Q/K/V projections separately
- Q/K normalization
- RoPE application
- Attention scores
- Attention output
- Cross-attention components
- FFN components (gate, up, down)
- Normalization layers
- Residual connections
"""

import torch
import sys
import os
import numpy as np

sys.path.insert(0, '/mnt/fast-disks/hao_lab/shao/LongCat-Video')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_grad_enabled(False)

def setup_environment():
    """Setup distributed environment for single GPU."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

def compare_tensors(name, orig, native, indent="  "):
    """Compare two tensors and return divergence metrics."""
    # Handle tuple returns (extract first element)
    if isinstance(orig, tuple):
        orig = orig[0]
    if isinstance(native, tuple):
        native = native[0]
    
    if orig.shape != native.shape:
        print(f"{indent}❌ {name}: SHAPE MISMATCH! Orig={orig.shape}, Native={native.shape}")
        return {'max_diff': float('inf'), 'mse': float('inf'), 'mean_diff': float('inf'), 'mean_rel': float('inf')}
    
    orig_f = orig.float()
    native_f = native.float()
    
    diff = torch.abs(orig_f - native_f)
    rel_diff = diff / (orig_f.abs() + 1e-8)
    
    # Calculate metrics
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    
    # Calculate MSE
    mse = torch.mean((orig_f - native_f) ** 2).item()
    rmse = torch.sqrt(torch.mean((orig_f - native_f) ** 2)).item()
    
    # Calculate cosine similarity
    orig_flat = orig_f.reshape(-1)
    native_flat = native_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat.unsqueeze(0), native_flat.unsqueeze(0)).item()
    
    # Determine status based on MSE
    if mse < 1e-8:
        status = "✅ IDENTICAL"
    elif mse < 1e-4:
        status = "✅ VERY CLOSE"
    elif mse < 1e-2:
        status = "⚠️  MODERATE DIFF"
    else:
        status = "❌ DIVERGED"
    
    print(f"{indent}{status} {name:40s} | MSE: {mse:.4e} | RMSE: {rmse:.4e} | max: {max_diff:.4e} | cos_sim: {cos_sim:.6f}")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'mse': mse,
        'rmse': rmse,
        'mean_rel': mean_rel,
        'cos_sim': cos_sim
    }

def detailed_block_comparison(block_idx=0):
    """Compare all components within a single transformer block."""
    print("="*100)
    print(f"DETAILED COMPONENT DIVERGENCE ANALYSIS - BLOCK {block_idx}")
    print("="*100)
    
    setup_environment()
    
    # ========================================================================
    # Load models
    # ========================================================================
    print("\n[1] Loading models...")
    
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalModel
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from safetensors.torch import load_file
    import json
    
    dit_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit"
    orig_model = OriginalModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
        cp_split_hw=[1, 1],
    ).cuda().eval()
    print("  ✓ Original model loaded")
    
    init_distributed_environment()
    initialize_model_parallel()
    
    transformer_path = "weights/longcat-native/transformer"
    with open(f"{transformer_path}/config.json") as f:
        config_dict = json.load(f)
    
    model_config = LongCatVideoConfig()
    native_model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    state_dict = load_file(f"{transformer_path}/model.safetensors")
    native_model.load_state_dict(state_dict, strict=False)
    native_model = native_model.cuda().to(torch.bfloat16).eval()
    print("  ✓ Native model loaded")
    
    # ========================================================================
    # Create test inputs
    # ========================================================================
    print("\n[2] Creating test inputs...")
    
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 100  # Small for testing
    hidden_dim = 4096
    
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device='cuda')
    timestep = torch.tensor([500.0], device='cuda')
    text_emb_orig = torch.randn(batch_size, 1, 256, 4096, dtype=torch.bfloat16, device='cuda')
    text_emb_native = text_emb_orig.squeeze(1)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Timestep: {timestep.item()}")
    
    # ========================================================================
    # Get block modules
    # ========================================================================
    orig_block = orig_model.blocks[block_idx]
    native_block = native_model.blocks[block_idx]
    
    # ========================================================================
    # [A] SELF-ATTENTION COMPONENT-BY-COMPONENT
    # ========================================================================
    print(f"\n[3] SELF-ATTENTION COMPONENTS (Block {block_idx})")
    print("-" * 100)
    
    divergences = {}
    
    # Get timestep embeddings for AdaLN
    with torch.no_grad():
        t_emb_orig = orig_model.t_embedder(timestep, dtype=torch.bfloat16)
        t_emb_native = native_model.time_embedder(timestep)
    
    # Handle shape differences (original might be [T, C], native [T, C])
    if t_emb_orig.shape != t_emb_native.shape:
        print(f"  Time embedding shape mismatch: orig={t_emb_orig.shape}, native={t_emb_native.shape}")
        if len(t_emb_orig.shape) == 3 and t_emb_orig.shape[0] == 1:
            t_emb_orig = t_emb_orig.squeeze(0)
    
    divergences['time_embedding'] = compare_tensors("Time Embedding", t_emb_orig, t_emb_native)
    
    # Apply AdaLN to get modulation parameters
    with torch.no_grad():
        adaln_orig = orig_block.adaLN_modulation[1](torch.nn.functional.silu(t_emb_orig))
        adaln_native_out = native_block.adaln_linear_1(torch.nn.functional.silu(t_emb_native))
    
    # Extract tensor from tuple if needed
    if isinstance(adaln_native_out, tuple):
        adaln_native = adaln_native_out[0]
    else:
        adaln_native = adaln_native_out
    
    divergences['adaln_modulation'] = compare_tensors("AdaLN Modulation", adaln_orig, adaln_native)
    
    # Split modulation parameters
    shift_msa_orig, scale_msa_orig, gate_msa_orig, shift_mlp_orig, scale_mlp_orig, gate_mlp_orig = adaln_orig.chunk(6, dim=1)
    shift_msa_native, scale_msa_native, gate_msa_native, shift_mlp_native, scale_mlp_native, gate_mlp_native = adaln_native.chunk(6, dim=1)
    
    # Apply modulation to input (for self-attention)
    x_norm_orig = orig_block.attn.mod_norm_attn(x)
    x_modulated_orig = x_norm_orig * (1 + scale_msa_orig.unsqueeze(1)) + shift_msa_orig.unsqueeze(1)
    
    x_norm_native = native_block.self_attn.norm(x)
    x_modulated_native = x_norm_native * (1 + scale_msa_native.unsqueeze(1)) + shift_msa_native.unsqueeze(1)
    
    divergences['input_norm'] = compare_tensors("Input Norm (pre-attn)", x_norm_orig, x_norm_native)
    divergences['input_modulated'] = compare_tensors("Modulated Input", x_modulated_orig, x_modulated_native)
    
    # QKV projections
    print("\n  === QKV Projections ===")
    with torch.no_grad():
        # Original: fused QKV
        qkv_orig = orig_block.attn.qkv(x_modulated_orig)  # [B, N, 3*C]
        B, N, _ = qkv_orig.shape
        qkv_orig = qkv_orig.view(B, N, 3, orig_block.attn.num_heads, orig_block.attn.head_dim)
        qkv_orig = qkv_orig.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q_orig, k_orig, v_orig = qkv_orig.unbind(0)
        
        # Native: separate Q, K, V
        qkv_native, _ = native_block.self_attn.qkv(x_modulated_native)
        qkv_native = qkv_native.view(B, N, 3, native_block.self_attn.num_heads, native_block.self_attn.head_dim)
        qkv_native = qkv_native.permute(2, 0, 3, 1, 4)
        q_native, k_native, v_native = qkv_native.unbind(0)
    
    divergences['q_projection'] = compare_tensors("Q Projection", q_orig, q_native, indent="    ")
    divergences['k_projection'] = compare_tensors("K Projection", k_orig, k_native, indent="    ")
    divergences['v_projection'] = compare_tensors("V Projection", v_orig, v_native, indent="    ")
    
    # Q/K normalization
    print("\n  === Q/K Normalization ===")
    with torch.no_grad():
        q_norm_orig = orig_block.attn.q_norm(q_orig)
        k_norm_orig = orig_block.attn.k_norm(k_orig)
        
        q_norm_native = native_block.self_attn.q_norm(q_native)
        k_norm_native = native_block.self_attn.k_norm(k_native)
    
    divergences['q_norm'] = compare_tensors("Q after Norm", q_norm_orig, q_norm_native, indent="    ")
    divergences['k_norm'] = compare_tensors("K after Norm", k_norm_orig, k_norm_native, indent="    ")
    
    # RoPE application
    print("\n  === RoPE Application ===")
    latent_shape = (9, 30, 52)
    with torch.no_grad():
        q_rope_orig, k_rope_orig = orig_block.attn.rope_3d(q_norm_orig, k_norm_orig, latent_shape)
        q_rope_native, k_rope_native = native_block.self_attn.rope_3d(q_norm_native, k_norm_native, grid_size=latent_shape)
    
    divergences['q_rope'] = compare_tensors("Q after RoPE", q_rope_orig, q_rope_native, indent="    ")
    divergences['k_rope'] = compare_tensors("K after RoPE", k_rope_orig, k_rope_native, indent="    ")
    
    # Attention output (skip actual attention computation for now, just compare final attn output)
    print("\n  === Self-Attention Output ===")
    with torch.no_grad():
        attn_out_orig = orig_block.attn(x_modulated_orig, shape=latent_shape, num_cond_latents=0)
        attn_out_native = native_block.self_attn(x_modulated_native, grid_size=latent_shape)
    
    divergences['self_attn_output'] = compare_tensors("Self-Attn Output", attn_out_orig, attn_out_native, indent="    ")
    
    # ========================================================================
    # [B] CROSS-ATTENTION COMPONENTS  
    # ========================================================================
    print(f"\n[4] CROSS-ATTENTION COMPONENTS (Block {block_idx})")
    print("-" * 100)
    
    # Prepare cross-attention input
    caption_emb_orig = orig_model.y_embedder(text_emb_orig)
    caption_emb_native = native_model.caption_embedder(text_emb_native)
    
    divergences['caption_embedding'] = compare_tensors("Caption Embedding", caption_emb_orig.squeeze(1), caption_emb_native[0])
    
    # Apply normalization
    x_ca_norm_orig = orig_block.cross_attn.pre_crs_attn_norm(x + gate_msa_orig.unsqueeze(1) * attn_out_orig)
    x_ca_norm_native = native_block.norm_cross(x + gate_msa_native.unsqueeze(1) * attn_out_native)
    
    divergences['cross_attn_input_norm'] = compare_tensors("Cross-Attn Input Norm", x_ca_norm_orig, x_ca_norm_native)
    
    # Cross-attention output
    print("\n  === Cross-Attention Output ===")
    with torch.no_grad():
        ca_out_orig = orig_block.cross_attn(x_ca_norm_orig, caption_emb_orig, shape=latent_shape, num_cond_latents=caption_emb_orig.shape[2])
        ca_out_native = native_block.cross_attn(x_ca_norm_native, caption_emb_native[0], grid_size=latent_shape)
    
    divergences['cross_attn_output'] = compare_tensors("Cross-Attn Output", ca_out_orig, ca_out_native, indent="    ")
    
    # ========================================================================
    # [C] FFN COMPONENTS
    # ========================================================================
    print(f"\n[5] FFN COMPONENTS (Block {block_idx})")
    print("-" * 100)
    
    # FFN input
    x_ffn_in = x + gate_msa_orig.unsqueeze(1) * attn_out_orig + ca_out_orig
    
    x_ffn_norm_orig = orig_block.ffn.mod_norm_ffn(x_ffn_in)
    x_ffn_modulated_orig = x_ffn_norm_orig * (1 + scale_mlp_orig.unsqueeze(1)) + shift_mlp_orig.unsqueeze(1)
    
    x_ffn_norm_native = native_block.norm_ffn(x + gate_msa_native.unsqueeze(1) * attn_out_native + ca_out_native)
    x_ffn_modulated_native = x_ffn_norm_native * (1 + scale_mlp_native.unsqueeze(1)) + shift_mlp_native.unsqueeze(1)
    
    divergences['ffn_input_norm'] = compare_tensors("FFN Input Norm", x_ffn_norm_orig, x_ffn_norm_native)
    divergences['ffn_input_modulated'] = compare_tensors("FFN Modulated Input", x_ffn_modulated_orig, x_ffn_modulated_native)
    
    # FFN forward
    print("\n  === FFN Layers ===")
    with torch.no_grad():
        # Original
        w1_orig = orig_block.ffn.w1(x_ffn_modulated_orig)
        w3_orig = orig_block.ffn.w3(x_ffn_modulated_orig)
        ffn_gate_orig = torch.nn.functional.silu(w1_orig) * w3_orig
        ffn_out_orig = orig_block.ffn.w2(ffn_gate_orig)
        
        # Native
        w1_native = native_block.ffn.w1(x_ffn_modulated_native)
        w3_native = native_block.ffn.w3(x_ffn_modulated_native)
        ffn_gate_native = torch.nn.functional.silu(w1_native) * w3_native
        ffn_out_native = native_block.ffn.w2(ffn_gate_native)
    
    divergences['ffn_w1'] = compare_tensors("FFN W1 (gate proj)", w1_orig, w1_native, indent="    ")
    divergences['ffn_w3'] = compare_tensors("FFN W3 (up proj)", w3_orig, w3_native, indent="    ")
    divergences['ffn_gate'] = compare_tensors("FFN Gate (SiLU * W3)", ffn_gate_orig, ffn_gate_native, indent="    ")
    divergences['ffn_w2'] = compare_tensors("FFN W2 (down proj)", ffn_out_orig, ffn_out_native, indent="    ")
    
    # ========================================================================
    # [D] BLOCK OUTPUT
    # ========================================================================
    print(f"\n[6] BLOCK OUTPUT (Block {block_idx})")
    print("-" * 100)
    
    block_out_orig = x + gate_msa_orig.unsqueeze(1) * attn_out_orig + ca_out_orig + gate_mlp_orig.unsqueeze(1) * ffn_out_orig
    block_out_native = x + gate_msa_native.unsqueeze(1) * attn_out_native + ca_out_native + gate_mlp_native.unsqueeze(1) * ffn_out_native
    
    divergences['block_output'] = compare_tensors("Final Block Output", block_out_orig, block_out_native)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("DIVERGENCE SUMMARY (sorted by MSE)")
    print("="*100)
    
    sorted_divs = sorted(divergences.items(), key=lambda x: x[1]['mse'], reverse=True)
    
    print(f"\n{'Component':<40} {'MSE':>12} {'RMSE':>12} {'Max Diff':>12} {'Cos Sim':>10}")
    print("-" * 90)
    for name, metrics in sorted_divs:
        mse = metrics['mse']
        if mse < 1e-8:
            icon = "✅"
        elif mse < 1e-4:
            icon = "✅"
        elif mse < 1e-2:
            icon = "⚠️ "
        else:
            icon = "❌"
        print(f"{icon} {name:<40} {metrics['mse']:>12.4e} {metrics['rmse']:>12.4e} {metrics['max_diff']:>12.4e} {metrics['cos_sim']:>10.6f}")
    
    # Find first significant divergence
    print("\n" + "="*100)
    print("FIRST SIGNIFICANT DIVERGENCE")
    print("="*100)
    
    mse_threshold = 1e-4
    for name, metrics in sorted_divs[::-1]:  # Check in order
        if metrics['mse'] > mse_threshold:
            print(f"\n❌ First component with MSE > {mse_threshold}: {name}")
            print(f"   MSE: {metrics['mse']:.6e}")
            print(f"   RMSE: {metrics['rmse']:.6e}")
            print(f"   Max difference: {metrics['max_diff']:.6e}")
            print(f"   Cosine similarity: {metrics['cos_sim']:.6f}")
            break
    else:
        print(f"\n✅ All components have MSE < {mse_threshold}")
    
    print("\n✅ Component-level analysis complete!")
    
    return divergences


if __name__ == "__main__":
    divergences = detailed_block_comparison(block_idx=0)

