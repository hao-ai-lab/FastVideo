#!/usr/bin/env python3
"""
Compare original LongCat pipeline with LoRA vs FastVideo native pipeline with LoRA.

This uses the full pipelines from both implementations to ensure LoRA is loaded correctly.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

# Create output directory
output_dir = Path("outputs/debug_lora_comparison")
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
    
    # Compute MSE
    mse = torch.mean((orig_f - native_f) ** 2)
    
    print(f"\n{step} {name}")
    print("="*80)
    print(f"  Shape: {orig.shape}")
    print(f"  Original: mean={orig_f.mean():.6f}, std={orig_f.std():.6f}, range=[{orig_f.min():.6f}, {orig_f.max():.6f}]")
    print(f"  Native:   mean={native_f.mean():.6f}, std={native_f.std():.6f}, range=[{native_f.min():.6f}, {native_f.max():.6f}]")
    print(f"  MSE: {mse:.6e}")
    print(f"  Abs diff: max={diff.max():.6e}, mean={diff.mean():.6e}, median={diff.median():.6e}")
    print(f"  Rel diff: max={rel_diff.max():.6e}, mean={rel_diff.mean():.6e}")
    
    # Check if similar
    if mse < 1e-8:
        print("  ✅ IDENTICAL (MSE < 1e-8)")
    elif mse < 1e-6:
        print("  ✅ VERY SIMILAR (MSE < 1e-6)")
    elif mse < 1e-4:
        print("  ✅ Similar (MSE < 1e-4)")
    elif mse < 1e-2:
        print("  ⚠️  Moderate difference (MSE < 1e-2)")
    else:
        print("  ❌ SIGNIFICANT DIFFERENCE (MSE >= 1e-2)")
    
    return mse.item()

def test_lora_pipeline_comparison():
    """Compare original and native pipelines with LoRA applied via pipelines."""
    setup_environment()
    
    print("="*100)
    print("LORA PIPELINE COMPARISON: ORIGINAL vs NATIVE (using full pipelines)")
    print("="*100)
    
    # ========================================================================
    # 1. LOAD BOTH PIPELINES WITH LORA
    # ========================================================================
    print("\n[1] LOADING PIPELINES WITH LORA")
    print("="*100)
    
    # Load original pipeline with LoRA
    print("\n[1a] Loading Original LongCat Pipeline with LoRA...")
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline as OriginalPipeline
    
    model_path_orig = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video"
    lora_path_orig = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/lora/cfg_step_lora.safetensors"
    
    orig_pipe = OriginalPipeline.from_pretrained(
        model_path_orig,
        torch_dtype=torch.bfloat16,
        lora_path=lora_path_orig,
        lora_weight=1.0,
    ).to("cuda")
    
    orig_model = orig_pipe.dit
    orig_model.eval()
    print("  ✓ Original pipeline with LoRA loaded")
    print(f"  Model type: {type(orig_model).__name__}")
    
    # Load native pipeline with LoRA
    print("\n[1b] Loading Native FastVideo Pipeline with LoRA...")
    from fastvideo import VideoGenerator
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    
    init_distributed_environment()
    initialize_model_parallel()
    
    model_path_native = "weights/longcat-native"
    lora_path_native = "weights/longcat-native/lora/distilled"
    
    native_generator = VideoGenerator.from_pretrained(
        model_path_native,
        lora_path=lora_path_native,
        lora_nickname="distilled",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
    )
    
    # Access the transformer through the pipeline
    native_pipe = native_generator.pipeline
    native_model = native_pipe.modules["transformer"]
    native_model.eval()
    print("  ✓ Native pipeline with LoRA loaded")
    print(f"  Model type: {type(native_model).__name__}")
    
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
    
    # Timestep - use a timestep relevant for distilled model
    timestep = torch.tensor([250.0], device="cuda")
    
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
    
    # Transformer blocks (sample every 6th block)
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    for i in sample_blocks:
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
    for i in sample_blocks:
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
    
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
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
    
    mse_values = {}
    
    # Compare embeddings
    print("\n--- EMBEDDINGS ---")
    mse_values["patch_embed"] = save_comparison(
        "Patch Embedding", 
        orig_activations["patch_embed"], 
        native_activations["patch_embed"],
        "[5.1]"
    )
    
    mse_values["time_embed"] = save_comparison(
        "Timestep Embedding",
        orig_activations["time_embed"],
        native_activations["time_embed"],
        "[5.2]"
    )
    
    # Caption embeddings
    print("\n[5.3] Caption Embedding")
    print("  Note: Native uses compaction, may have different format")
    orig_cap = orig_activations["caption_embed"]
    native_cap = native_activations["caption_embed"]
    if isinstance(orig_cap, tuple):
        print(f"  Original: {orig_cap[0].shape}")
    else:
        print(f"  Original: {orig_cap.shape}")
    if isinstance(native_cap, tuple):
        print(f"  Native: {native_cap[0].shape}")
    else:
        print(f"  Native: {native_cap.shape}")
    
    # Compare transformer blocks
    print("\n--- TRANSFORMER BLOCKS (with LoRA) ---")
    for idx, i in enumerate(sample_blocks):
        print(f"\n--- Block {i} ---")
        
        mse_values[f"block_{i}_self_attn"] = save_comparison(
            f"Block {i} Self-Attention",
            orig_activations[f"block_{i}_self_attn"],
            native_activations[f"block_{i}_self_attn"],
            f"[5.{4+idx}a]"
        )
        
        mse_values[f"block_{i}_cross_attn"] = save_comparison(
            f"Block {i} Cross-Attention",
            orig_activations[f"block_{i}_cross_attn"],
            native_activations[f"block_{i}_cross_attn"],
            f"[5.{4+idx}b]"
        )
        
        mse_values[f"block_{i}_ffn"] = save_comparison(
            f"Block {i} FFN",
            orig_activations[f"block_{i}_ffn"],
            native_activations[f"block_{i}_ffn"],
            f"[5.{4+idx}c]"
        )
        
        mse_values[f"block_{i}"] = save_comparison(
            f"Block {i} Output",
            orig_activations[f"block_{i}"],
            native_activations[f"block_{i}"],
            f"[5.{4+idx}d]"
        )
    
    # Compare final layer
    print("\n--- FINAL LAYER ---")
    mse_values["final_layer"] = save_comparison(
        "Final Layer Output",
        orig_activations["final_layer"],
        native_activations["final_layer"],
        "[5.6]"
    )
    
    # Compare final outputs
    print("\n--- FINAL MODEL OUTPUTS ---")
    mse_values["final_output"] = save_comparison(
        "Final Model Output",
        orig_output,
        native_output,
        "[5.7]"
    )
    
    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    print("\n[6] SUMMARY OF MSE VALUES")
    print("="*100)
    
    print("\nMSE values by layer (sorted by magnitude):")
    sorted_mse = sorted(mse_values.items(), key=lambda x: x[1], reverse=True)
    
    for name, mse in sorted_mse:
        if mse < 1e-6:
            status = "✅"
        elif mse < 1e-4:
            status = "✅"
        elif mse < 1e-2:
            status = "⚠️"
        else:
            status = "❌"
        print(f"  {status} {name:40s}: MSE = {mse:.6e}")
    
    # Find where divergence starts
    print("\n" + "="*100)
    print("DIVERGENCE ANALYSIS")
    print("="*100)
    
    divergence_threshold = 1e-4
    first_divergence = None
    
    check_order = ["patch_embed", "time_embed"]
    for i in sample_blocks:
        check_order.extend([
            f"block_{i}_self_attn",
            f"block_{i}_cross_attn",
            f"block_{i}_ffn",
            f"block_{i}",
        ])
    check_order.extend(["final_layer", "final_output"])
    
    for name in check_order:
        if name in mse_values:
            if mse_values[name] > divergence_threshold:
                if first_divergence is None:
                    first_divergence = name
                    print(f"\n❌ First significant divergence at: {name}")
                    print(f"   MSE: {mse_values[name]:.6e}")
                    
                    # Provide suggestions
                    if "attn" in name:
                        print("\n💡 Divergence in attention layer!")
                        print("   Possible issues:")
                        print("   1. LoRA weight loading/merging differs")
                        print("   2. LoRA scale application differs")
                        print("   3. Weight format conversion issue")
                        print("   4. Different attention implementation details")
                    break
    
    if first_divergence is None:
        print("\n✅ No significant divergence found! Models produce very similar outputs.")
    else:
        print("\n📊 Detailed analysis:")
        print(f"   Layers before divergence: ✅ Matching")
        print(f"   First divergence at: {first_divergence}")
        print(f"   Layers after: Check if error accumulates")
    
    # Save detailed comparison to file
    report_path = output_dir / "lora_pipeline_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("="*100 + "\n")
        f.write("LORA PIPELINE COMPARISON REPORT\n")
        f.write("="*100 + "\n\n")
        f.write("MSE VALUES (sorted by magnitude):\n\n")
        for name, mse in sorted_mse:
            f.write(f"{name:40s}: {mse:.6e}\n")
        f.write("\n" + "="*100 + "\n")
        if first_divergence:
            f.write(f"First divergence at: {first_divergence}\n")
            f.write(f"MSE: {mse_values[first_divergence]:.6e}\n")
        else:
            f.write("No significant divergence found\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Clean up hooks
    for hook in orig_hooks:
        hook.remove()
    for hook in native_hooks:
        hook.remove()
    
    print("\n" + "="*100)
    print("✅ LoRA pipeline comparison complete!")
    print("="*100)

if __name__ == "__main__":
    test_lora_pipeline_comparison()

