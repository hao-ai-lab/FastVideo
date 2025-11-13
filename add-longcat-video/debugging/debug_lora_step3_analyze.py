#!/usr/bin/env python3
"""
Step 3: Analyze and compare the saved activations from both models.
Can run with either conda environment.
"""

import torch
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("outputs/debug_lora_comparison")

def compute_stats(tensor):
    """Compute statistics for a tensor."""
    t_f = tensor.float()
    return {
        'mean': t_f.mean().item(),
        'std': t_f.std().item(),
        'min': t_f.min().item(),
        'max': t_f.max().item(),
    }

def compare_tensors(name, orig, native, step=""):
    """Compare two tensors and return MSE."""
    orig_f = orig.float()
    native_f = native.float()
    
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

def analyze_comparison():
    """Analyze and compare the saved activations."""
    
    print("="*100)
    print("STEP 3: ANALYZING COMPARISON (Original WITH LoRA vs Native WITHOUT LoRA)")
    print("="*100)
    print()
    print("⚠️  NOTE: This comparison is between:")
    print("   - Original LongCat WITH distilled LoRA applied")
    print("   - FastVideo native WITHOUT LoRA (base model only)")
    print()
    print("This will help us understand:")
    print("   1. Do the base weights match correctly?")
    print("   2. What effect does LoRA have on the original model?")
    print("   3. Where should we look to add LoRA support to native?")
    print()
    
    # ========================================================================
    # 1. LOAD ACTIVATIONS
    # ========================================================================
    print("\n[1] LOADING ACTIVATIONS")
    print("="*100)
    
    orig_path = output_dir / "original_activations.pt"
    native_path = output_dir / "native_activations.pt"
    
    if not orig_path.exists():
        raise FileNotFoundError(
            f"Original activations not found at {orig_path}. "
            "Please run debug_lora_step1_original.py first!"
        )
    
    if not native_path.exists():
        raise FileNotFoundError(
            f"Native activations not found at {native_path}. "
            "Please run debug_lora_step2_native.py first!"
        )
    
    orig_activations = torch.load(orig_path)
    native_activations = torch.load(native_path)
    
    print(f"  ✓ Loaded original activations: {len(orig_activations)} layers")
    print(f"  ✓ Loaded native activations: {len(native_activations)} layers")
    
    # ========================================================================
    # 2. COMPARE ACTIVATIONS
    # ========================================================================
    print("\n[2] COMPARING ACTIVATIONS")
    print("="*100)
    
    mse_values = {}
    
    # Compare embeddings
    print("\n--- EMBEDDINGS ---")
    
    if "patch_embed" in orig_activations and "patch_embed" in native_activations:
        mse_values["patch_embed"] = compare_tensors(
            "Patch Embedding", 
            orig_activations["patch_embed"], 
            native_activations["patch_embed"],
            "[2.1]"
        )
    
    if "time_embed" in orig_activations and "time_embed" in native_activations:
        mse_values["time_embed"] = compare_tensors(
            "Timestep Embedding",
            orig_activations["time_embed"],
            native_activations["time_embed"],
            "[2.2]"
        )
    
    # Caption embeddings (may differ in format due to compaction)
    print("\n[2.3] Caption Embedding")
    if "caption_embed" in orig_activations and "caption_embed" in native_activations:
        orig_cap = orig_activations["caption_embed"]
        native_cap = native_activations["caption_embed"]
        print(f"  Original: {orig_cap.shape}")
        print(f"  Native: {native_cap.shape}")
        print("  Note: Native uses compaction, shapes may differ")
    
    # Compare transformer blocks
    print("\n--- TRANSFORMER BLOCKS (with LoRA) ---")
    sample_blocks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    
    for idx, i in enumerate(sample_blocks):
        print(f"\n--- Block {i} ---")
        
        if f"block_{i}_self_attn" in orig_activations and f"block_{i}_self_attn" in native_activations:
            mse_values[f"block_{i}_self_attn"] = compare_tensors(
                f"Block {i} Self-Attention",
                orig_activations[f"block_{i}_self_attn"],
                native_activations[f"block_{i}_self_attn"],
                f"[2.{4+idx}a]"
            )
        
        if f"block_{i}_cross_attn" in orig_activations and f"block_{i}_cross_attn" in native_activations:
            mse_values[f"block_{i}_cross_attn"] = compare_tensors(
                f"Block {i} Cross-Attention",
                orig_activations[f"block_{i}_cross_attn"],
                native_activations[f"block_{i}_cross_attn"],
                f"[2.{4+idx}b]"
            )
        
        if f"block_{i}_ffn" in orig_activations and f"block_{i}_ffn" in native_activations:
            mse_values[f"block_{i}_ffn"] = compare_tensors(
                f"Block {i} FFN",
                orig_activations[f"block_{i}_ffn"],
                native_activations[f"block_{i}_ffn"],
                f"[2.{4+idx}c]"
            )
        
        if f"block_{i}" in orig_activations and f"block_{i}" in native_activations:
            mse_values[f"block_{i}"] = compare_tensors(
                f"Block {i} Output",
                orig_activations[f"block_{i}"],
                native_activations[f"block_{i}"],
                f"[2.{4+idx}d]"
            )
    
    # Compare final layer
    print("\n--- FINAL LAYER ---")
    if "final_layer" in orig_activations and "final_layer" in native_activations:
        mse_values["final_layer"] = compare_tensors(
            "Final Layer Output",
            orig_activations["final_layer"],
            native_activations["final_layer"],
            "[2.5]"
        )
    
    # Compare final outputs
    print("\n--- FINAL MODEL OUTPUTS ---")
    if "final_output" in orig_activations and "final_output" in native_activations:
        mse_values["final_output"] = compare_tensors(
            "Final Model Output",
            orig_activations["final_output"],
            native_activations["final_output"],
            "[2.6]"
        )
    
    # ========================================================================
    # 3. SUMMARY AND ANALYSIS
    # ========================================================================
    print("\n[3] SUMMARY OF MSE VALUES")
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
                    
                    # Provide detailed analysis
                    if "attn" in name:
                        print("\n💡 Divergence in attention layer!")
                        print("   Possible issues:")
                        print("   1. LoRA weight loading/merging differs between implementations")
                        print("   2. LoRA scale application differs")
                        print("   3. Weight format conversion issue in checkpoint conversion")
                        print("   4. Different attention implementation details (RoPE, masking, etc.)")
                    elif "ffn" in name:
                        print("\n💡 Divergence in FFN layer!")
                        print("   FFN layers typically don't have LoRA, so this suggests:")
                        print("   1. Error accumulation from previous layers")
                        print("   2. Different normalization behavior")
                    elif "embed" in name:
                        print("\n💡 Divergence in embedding layer!")
                        print("   Embeddings shouldn't have LoRA, so this suggests:")
                        print("   1. Different weight loading")
                        print("   2. Different preprocessing")
                    break
    
    if first_divergence is None:
        print("\n✅ No significant divergence found! Models produce very similar outputs.")
        print("   The LoRA implementations are equivalent.")
    else:
        print("\n📊 Detailed analysis:")
        print(f"   Layers before divergence: ✅ Matching")
        print(f"   First divergence at: {first_divergence}")
        print(f"   Layers after divergence: Check if error accumulates")
        
        # Check if divergence grows
        diverged_layers = [name for name in check_order if name in mse_values and mse_values[name] > divergence_threshold]
        if len(diverged_layers) > 1:
            print(f"\n   Total diverged layers: {len(diverged_layers)}")
            print(f"   MSE at first divergence: {mse_values[first_divergence]:.6e}")
            print(f"   MSE at final output: {mse_values.get('final_output', 0):.6e}")
            if 'final_output' in mse_values:
                if mse_values['final_output'] > mse_values[first_divergence] * 10:
                    print("   ⚠️  Error is ACCUMULATING through the network!")
                else:
                    print("   ℹ️  Error stays relatively stable")
    
    # Save detailed report
    report_path = output_dir / "lora_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("="*100 + "\n")
        f.write("LORA COMPARISON REPORT\n")
        f.write("="*100 + "\n\n")
        f.write("MSE VALUES (sorted by magnitude):\n\n")
        for name, mse in sorted_mse:
            status = "✅" if mse < 1e-4 else ("⚠️" if mse < 1e-2 else "❌")
            f.write(f"{status} {name:40s}: {mse:.6e}\n")
        f.write("\n" + "="*100 + "\n")
        if first_divergence:
            f.write(f"First divergence at: {first_divergence}\n")
            f.write(f"MSE: {mse_values[first_divergence]:.6e}\n")
        else:
            f.write("No significant divergence found\n")
        f.write("\n" + "="*100 + "\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    print("\n" + "="*100)
    print("✅ Analysis complete!")
    print("="*100)
    
    # Return summary
    return {
        'first_divergence': first_divergence,
        'mse_values': mse_values,
        'sorted_mse': sorted_mse,
    }

if __name__ == "__main__":
    analyze_comparison()

