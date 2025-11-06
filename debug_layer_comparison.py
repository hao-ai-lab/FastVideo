"""
Layer-by-layer comparison between original LongCat and native implementation.

This script runs a single forward pass through both models and compares
intermediate outputs at each layer to identify where they diverge.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

# Create output directory
output_dir = Path("outputs/debug_layers")
output_dir.mkdir(parents=True, exist_ok=True)

def setup_environment():
    """Setup distributed environment for single GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

def save_comparison(name, orig, native, step=""):
    """Compare two tensors and save statistics."""
    orig_f = orig.float().cpu()
    native_f = native.float().cpu()
    
    diff = torch.abs(orig_f - native_f)
    rel_diff = diff / (orig_f.abs() + 1e-8)
    
    print(f"\n{step} {name}")
    print("="*80)
    print(f"  Shape: {orig.shape}")
    print(f"  Original: mean={orig_f.mean():.6f}, std={orig_f.std():.6f}, range=[{orig_f.min():.6f}, {orig_f.max():.6f}]")
    print(f"  Native:   mean={native_f.mean():.6f}, std={native_f.std():.6f}, range=[{native_f.min():.6f}, {native_f.max():.6f}]")
    print(f"  Abs diff: max={diff.max():.6e}, mean={diff.mean():.6e}, median={diff.median():.6e}")
    print(f"  Rel diff: max={rel_diff.max():.6e}, mean={rel_diff.mean():.6e}")
    
    # Check if similar
    if diff.max() < 1e-4:
        print("  ✅ VERY SIMILAR (max diff < 1e-4)")
    elif diff.max() < 1e-2:
        print("  ✅ Similar (max diff < 1e-2)")
    elif diff.max() < 0.1:
        print("  ⚠️  Moderate difference (max diff < 0.1)")
    else:
        print("  ❌ SIGNIFICANT DIFFERENCE (max diff >= 0.1)")
    
    return diff.max().item()

def test_layer_by_layer():
    """Compare original and native implementations layer by layer."""
    setup_environment()
    
    print("="*100)
    print("LAYER-BY-LAYER COMPARISON: ORIGINAL vs NATIVE")
    print("="*100)
    
    # ========================================================================
    # 1. LOAD BOTH MODELS
    # ========================================================================
    print("\n[1] LOADING MODELS")
    print("="*100)
    
    # Load original model
    print("\n[1a] Loading Original LongCat...")
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalModel
    
    dit_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit"
    orig_model = OriginalModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
    )
    orig_model = orig_model.to("cuda").eval()
    print("  ✓ Original model loaded")
    
    # Load native model
    print("\n[1b] Loading Native LongCat...")
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from safetensors.torch import load_file
    import json
    
    init_distributed_environment()
    initialize_model_parallel()
    
    transformer_path = "weights/longcat-native/transformer"
    with open(f"{transformer_path}/config.json") as f:
        config_dict = json.load(f)
    
    model_config = LongCatVideoConfig()
    # Free memory from original model first
    del orig_model
    torch.cuda.empty_cache()
    print("  Cleared original model from memory")
    
    native_model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    
    state_dict = load_file(f"{transformer_path}/model.safetensors")
    native_model.load_state_dict(state_dict, strict=False)
    native_model = native_model.to("cuda").to(torch.bfloat16).eval()
    print("  ✓ Native model loaded")
    
    # Reload original for comparison
    print("\n[1c] Reloading original model...")
    orig_model = OriginalModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
    )
    orig_model = orig_model.to("cuda").eval()
    print("  ✓ Original model reloaded")
    
    # ========================================================================
    # 2. CREATE IDENTICAL TEST INPUTS
    # ========================================================================
    print("\n[2] CREATING TEST INPUTS")
    print("="*100)
    
    torch.manual_seed(42)
    
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    # Input latents
    latents = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.18215
    
    # Timestep
    timestep = torch.tensor([500.0], device="cuda")
    
    # Text embeddings - original expects [B, 1, N, C], native expects [B, N, C]
    text_emb_orig = torch.randn(batch_size, 1, 256, 4096, dtype=torch.bfloat16, device="cuda")
    text_emb_native = text_emb_orig.squeeze(1)
    
    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text (original): {text_emb_orig.shape}")
    print(f"  Text (native): {text_emb_native.shape}")
    
    # ========================================================================
    # 3. HOOK TO CAPTURE INTERMEDIATE OUTPUTS
    # ========================================================================
    print("\n[3] SETTING UP HOOKS TO CAPTURE INTERMEDIATE OUTPUTS")
    print("="*100)
    
    orig_activations = {}
    native_activations = {}
    
    def make_hook(name, storage):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            storage[name] = output.detach()
        return hook
    
    # Register hooks on original model
    orig_hooks = []
    
    # Patch embedding
    orig_hooks.append(orig_model.x_embedder.register_forward_hook(
        make_hook("patch_embed", orig_activations)))
    
    # Timestep embedding
    orig_hooks.append(orig_model.t_embedder.register_forward_hook(
        make_hook("time_embed", orig_activations)))
    
    # Caption embedding
    orig_hooks.append(orig_model.y_embedder.register_forward_hook(
        make_hook("caption_embed", orig_activations)))
    
    # Transformer blocks (sample first, middle, last)
    for i in [0, 23, 47]:
        orig_hooks.append(orig_model.blocks[i].register_forward_hook(
            make_hook(f"block_{i}", orig_activations)))
        
        # Self-attention within block
        orig_hooks.append(orig_model.blocks[i].attn.register_forward_hook(
            make_hook(f"block_{i}_self_attn", orig_activations)))
        
        # Cross-attention within block
        orig_hooks.append(orig_model.blocks[i].cross_attn.register_forward_hook(
            make_hook(f"block_{i}_cross_attn", orig_activations)))
        
        # FFN within block
        orig_hooks.append(orig_model.blocks[i].ffn.register_forward_hook(
            make_hook(f"block_{i}_ffn", orig_activations)))
    
    # Final layer
    orig_hooks.append(orig_model.final_layer.register_forward_hook(
        make_hook("final_layer", orig_activations)))
    
    # Register hooks on native model
    native_hooks = []
    
    # Patch embedding
    native_hooks.append(native_model.patch_embed.register_forward_hook(
        make_hook("patch_embed", native_activations)))
    
    # Timestep embedding
    native_hooks.append(native_model.time_embedder.register_forward_hook(
        make_hook("time_embed", native_activations)))
    
    # Caption embedding
    native_hooks.append(native_model.caption_embedder.register_forward_hook(
        make_hook("caption_embed", native_activations)))
    
    # Transformer blocks (same indices)
    for i in [0, 23, 47]:
        native_hooks.append(native_model.blocks[i].register_forward_hook(
            make_hook(f"block_{i}", native_activations)))
        
        # Self-attention within block
        native_hooks.append(native_model.blocks[i].self_attn.register_forward_hook(
            make_hook(f"block_{i}_self_attn", native_activations)))
        
        # Cross-attention within block
        native_hooks.append(native_model.blocks[i].cross_attn.register_forward_hook(
            make_hook(f"block_{i}_cross_attn", native_activations)))
        
        # FFN within block
        native_hooks.append(native_model.blocks[i].ffn.register_forward_hook(
            make_hook(f"block_{i}_ffn", native_activations)))
    
    # Final layer
    native_hooks.append(native_model.final_layer.register_forward_hook(
        make_hook("final_layer", native_activations)))
    
    print(f"  ✓ Registered {len(orig_hooks)} hooks on original model")
    print(f"  ✓ Registered {len(native_hooks)} hooks on native model")
    
    # ========================================================================
    # 4. RUN FORWARD PASSES
    # ========================================================================
    print("\n[4] RUNNING FORWARD PASSES")
    print("="*100)
    
    with torch.no_grad():
        print("\n[4a] Original model forward pass...")
        orig_output = orig_model(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_emb_orig,
        )
        print(f"  ✓ Original output: {orig_output.shape}")
        
        print("\n[4b] Native model forward pass...")
        dummy_batch = ForwardBatch(data_type="t2v")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=dummy_batch,
        ):
            native_output = native_model(
                hidden_states=latents,
                encoder_hidden_states=text_emb_native,
                timestep=timestep,
            )
        print(f"  ✓ Native output: {native_output.shape}")
    
    # ========================================================================
    # 5. COMPARE INTERMEDIATE OUTPUTS
    # ========================================================================
    print("\n[5] COMPARING INTERMEDIATE OUTPUTS")
    print("="*100)
    
    max_diffs = {}
    
    # Compare embeddings
    print("\n--- EMBEDDINGS ---")
    max_diffs["patch_embed"] = save_comparison(
        "Patch Embedding", 
        orig_activations["patch_embed"], 
        native_activations["patch_embed"],
        "[5.1]"
    )
    
    max_diffs["time_embed"] = save_comparison(
        "Timestep Embedding",
        orig_activations["time_embed"],
        native_activations["time_embed"],
        "[5.2]"
    )
    
    # Caption embeddings - need to handle compaction
    print("\n[5.3] Caption Embedding")
    print("  Note: Native uses compaction, may have different format")
    orig_cap = orig_activations["caption_embed"]
    native_cap = native_activations["caption_embed"]
    print(f"  Original: {orig_cap[0].shape if isinstance(orig_cap, tuple) else orig_cap.shape}")
    print(f"  Native: {native_cap[0].shape if isinstance(native_cap, tuple) else native_cap.shape}")
    
    # Compare transformer blocks
    print("\n--- TRANSFORMER BLOCKS ---")
    for i in [0, 23, 47]:
        print(f"\n--- Block {i} ---")
        
        max_diffs[f"block_{i}_self_attn"] = save_comparison(
            f"Block {i} Self-Attention",
            orig_activations[f"block_{i}_self_attn"],
            native_activations[f"block_{i}_self_attn"],
            f"[5.{4+i}a]"
        )
        
        max_diffs[f"block_{i}_cross_attn"] = save_comparison(
            f"Block {i} Cross-Attention",
            orig_activations[f"block_{i}_cross_attn"],
            native_activations[f"block_{i}_cross_attn"],
            f"[5.{4+i}b]"
        )
        
        max_diffs[f"block_{i}_ffn"] = save_comparison(
            f"Block {i} FFN",
            orig_activations[f"block_{i}_ffn"],
            native_activations[f"block_{i}_ffn"],
            f"[5.{4+i}c]"
        )
        
        max_diffs[f"block_{i}"] = save_comparison(
            f"Block {i} Output",
            orig_activations[f"block_{i}"],
            native_activations[f"block_{i}"],
            f"[5.{4+i}d]"
        )
    
    # Compare final layer
    print("\n--- FINAL LAYER ---")
    max_diffs["final_layer"] = save_comparison(
        "Final Layer Output",
        orig_activations["final_layer"],
        native_activations["final_layer"],
        "[5.6]"
    )
    
    # Compare final outputs
    print("\n--- FINAL MODEL OUTPUTS ---")
    max_diffs["final_output"] = save_comparison(
        "Final Model Output",
        orig_output,
        native_output,
        "[5.7]"
    )
    
    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    print("\n[6] SUMMARY OF DIFFERENCES")
    print("="*100)
    
    print("\nMax differences by layer:")
    sorted_diffs = sorted(max_diffs.items(), key=lambda x: x[1], reverse=True)
    
    for name, diff in sorted_diffs:
        status = "✅" if diff < 1e-2 else ("⚠️" if diff < 0.1 else "❌")
        print(f"  {status} {name:30s}: {diff:.6e}")
    
    # Find where divergence starts
    print("\n" + "="*100)
    print("DIVERGENCE ANALYSIS")
    print("="*100)
    
    divergence_threshold = 1e-2
    first_divergence = None
    
    check_order = [
        "patch_embed",
        "time_embed",
        "block_0_self_attn",
        "block_0_cross_attn",
        "block_0_ffn",
        "block_0",
        "block_23_self_attn",
        "block_23_cross_attn",
        "block_23_ffn",
        "block_23",
        "block_47_self_attn",
        "block_47_cross_attn",
        "block_47_ffn",
        "block_47",
        "final_layer",
        "final_output",
    ]
    
    for name in check_order:
        if name in max_diffs:
            if max_diffs[name] > divergence_threshold:
                if first_divergence is None:
                    first_divergence = name
                    print(f"\n❌ First significant divergence at: {name}")
                    print(f"   Max difference: {max_diffs[name]:.6e}")
                    break
    
    if first_divergence is None:
        print("\n✅ No significant divergence found! Models produce very similar outputs.")
    
    # Clean up hooks
    for hook in orig_hooks:
        hook.remove()
    for hook in native_hooks:
        hook.remove()
    
    print("\n" + "="*100)
    print("✅ Layer-by-layer comparison complete!")
    print("="*100)

if __name__ == "__main__":
    test_layer_by_layer()

