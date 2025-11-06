"""
Detailed debugging script for LongCat native implementation with visualizations.

This script adds comprehensive visualizations and checks to identify issues:
1. CFG-zero implementation
2. Noise prediction negation
3. Parameter ordering
4. Attention patterns
5. Intermediate activations
"""

import torch
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

# Create output directory
output_dir = Path("outputs/debug_longcat_native")
output_dir.mkdir(parents=True, exist_ok=True)

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping visualizations")

def save_tensor_stats(tensor, name, step=""):
    """Save tensor statistics to file and print."""
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": tensor.float().mean().item(),
        "std": tensor.float().std().item(),
        "min": tensor.float().min().item(),
        "max": tensor.float().max().item(),
        "abs_mean": tensor.float().abs().mean().item(),
    }
    
    msg = f"\n{'='*80}\n{step} {name}\n{'='*80}"
    msg += f"\n  Shape: {stats['shape']}"
    msg += f"\n  Dtype: {stats['dtype']}"
    msg += f"\n  Mean: {stats['mean']:.6f}"
    msg += f"\n  Std:  {stats['std']:.6f}"
    msg += f"\n  Min:  {stats['min']:.6f}"
    msg += f"\n  Max:  {stats['max']:.6f}"
    msg += f"\n  |Mean|: {stats['abs_mean']:.6f}"
    
    # Distribution
    t_flat = tensor.float().flatten()
    msg += f"\n\n  Distribution:"
    msg += f"\n    |values| < 0.001: {(t_flat.abs() < 0.001).float().mean().item()*100:.1f}%"
    msg += f"\n    |values| < 0.01:  {(t_flat.abs() < 0.01).float().mean().item()*100:.1f}%"
    msg += f"\n    |values| < 0.1:   {(t_flat.abs() < 0.1).float().mean().item()*100:.1f}%"
    msg += f"\n    |values| < 1.0:   {(t_flat.abs() < 1.0).float().mean().item()*100:.1f}%"
    msg += f"\n    |values| > 10:    {(t_flat.abs() > 10).float().mean().item()*100:.1f}%"
    
    print(msg)
    return stats

def visualize_tensor(tensor, name, step="", save_path=None):
    """Visualize tensor distribution as histogram."""
    if not HAS_MATPLOTLIB:
        return
        
    if save_path is None:
        save_path = output_dir / f"{step}_{name}.png"
    
    # Flatten and convert to numpy
    data = tensor.float().cpu().numpy().flatten()
    
    # Create histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Full range histogram
    axes[0].hist(data, bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{name} - Full Range')
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed histogram (central 99%)
    p1, p99 = np.percentile(data, [0.5, 99.5])
    mask = (data >= p1) & (data <= p99)
    axes[1].hist(data[mask], bins=100, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{name} - Central 99%')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Visualization saved to: {save_path}")


def test_native_with_instrumentation():
    """Test native model with detailed instrumentation."""
    print("="*100)
    print("DETAILED LONGCAT NATIVE DEBUGGING")
    print("="*100)
    
    # Set device and distributed environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))  # Random port to avoid conflicts
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    device = "cuda"
    
    # ========================================================================
    # 1. LOAD NATIVE MODEL
    # ========================================================================
    print("\n" + "="*100)
    print("[1] LOADING NATIVE MODEL")
    print("="*100)
    
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.distributed.parallel_state import (
        initialize_model_parallel,
        init_distributed_environment
    )
    from safetensors.torch import load_file
    import json
    
    # Initialize distributed environment for single GPU
    init_distributed_environment()
    initialize_model_parallel()
    
    transformer_path = "weights/longcat-native/transformer"
    
    with open(f"{transformer_path}/config.json") as f:
        config_dict = json.load(f)
    
    # Use LongCatVideoConfig
    model_config = LongCatVideoConfig()
    model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    
    state_dict = load_file(f"{transformer_path}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"  ✓ Model loaded")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device).to(torch.bfloat16).eval()
    
    # ========================================================================
    # 2. CREATE TEST INPUTS
    # ========================================================================
    print("\n" + "="*100)
    print("[2] CREATING TEST INPUTS")
    print("="*100)
    
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    torch.manual_seed(42)
    
    # Latent input
    latents = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device=device
    ) * 0.18215
    
    save_tensor_stats(latents, "Input Latents", "[2]")
    visualize_tensor(latents, "input_latents", "step0")
    
    # Timesteps (mimicking denoising at t=500)
    timestep = torch.tensor([500.0], device=device)
    print(f"\n  Timestep: {timestep.item()}")
    
    # Text embeddings [B, N_text, C_text]
    text_emb = torch.randn(batch_size, 256, 4096, dtype=torch.bfloat16, device=device)
    save_tensor_stats(text_emb, "Text Embeddings", "[2]")
    
    # ========================================================================
    # 3. TEST SINGLE FORWARD PASS (NO CFG)
    # ========================================================================
    print("\n" + "="*100)
    print("[3] SINGLE FORWARD PASS (NO CFG)")
    print("="*100)
    
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
    # Create a dummy batch for forward context
    dummy_batch = ForwardBatch(data_type="t2v")
    
    with torch.no_grad():
        # Set forward context as required by DistributedAttention
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=dummy_batch,
        ):
            # Forward pass without CFG
            output_single = model(
                hidden_states=latents,
                encoder_hidden_states=text_emb,
                timestep=timestep,
            )
    
    save_tensor_stats(output_single, "Single Forward Output", "[3]")
    visualize_tensor(output_single, "output_single", "step1")
    
    # Check if output looks reasonable
    if output_single.float().abs().mean() < 1e-5:
        print("\n  ❌ WARNING: Output is nearly zero! (dead network)")
    elif output_single.float().abs().mean() > 1000:
        print("\n  ❌ WARNING: Output has very large values! (numerical instability)")
    else:
        print("\n  ✅ Output magnitude looks reasonable")
    
    # ========================================================================
    # 4. TEST CFG FORWARD PASS
    # ========================================================================
    print("\n" + "="*100)
    print("[4] CFG FORWARD PASS")
    print("="*100)
    
    # Create unconditional (negative) embeddings (zeros for null prompt)
    uncond_text_emb = torch.zeros_like(text_emb)
    save_tensor_stats(uncond_text_emb, "Unconditional Text Embeddings", "[4]")
    
    # Concatenate for batched CFG
    latents_cfg = torch.cat([latents, latents], dim=0)  # [2, C, T, H, W]
    text_emb_cfg = torch.cat([uncond_text_emb, text_emb], dim=0)  # [2, N, C]
    timestep_cfg = timestep.expand(2)  # [2]
    
    print(f"\n  CFG Latents shape: {latents_cfg.shape}")
    print(f"  CFG Text shape: {text_emb_cfg.shape}")
    print(f"  CFG Timestep shape: {timestep_cfg.shape}")
    
    with torch.no_grad():
        # Forward pass with CFG batching
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=dummy_batch,
        ):
            output_cfg_batched = model(
                hidden_states=latents_cfg,
                encoder_hidden_states=text_emb_cfg,
                timestep=timestep_cfg,
            )
    
    save_tensor_stats(output_cfg_batched, "CFG Batched Output (before split)", "[4]")
    visualize_tensor(output_cfg_batched, "output_cfg_batched", "step2")
    
    # Split into conditional and unconditional
    noise_pred_uncond, noise_pred_cond = output_cfg_batched.chunk(2)
    
    print("\n" + "-"*80)
    print("After splitting CFG batch:")
    print("-"*80)
    save_tensor_stats(noise_pred_uncond, "Unconditional Prediction", "[4]")
    save_tensor_stats(noise_pred_cond, "Conditional Prediction", "[4]")
    
    visualize_tensor(noise_pred_uncond, "noise_pred_uncond", "step3")
    visualize_tensor(noise_pred_cond, "noise_pred_cond", "step3")
    
    # ========================================================================
    # 5. APPLY CFG-ZERO
    # ========================================================================
    print("\n" + "="*100)
    print("[5] APPLYING CFG-ZERO OPTIMIZATION")
    print("="*100)
    
    guidance_scale = 3.5
    print(f"  Guidance scale: {guidance_scale}")
    
    # Flatten for CFG-zero calculation
    B = noise_pred_cond.shape[0]
    positive_flat = noise_pred_cond.reshape(B, -1)
    negative_flat = noise_pred_uncond.reshape(B, -1)
    
    # Calculate optimized scale (CFG-zero)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm
    
    print(f"\n  Dot product (cond · uncond): {dot_product.item():.6f}")
    print(f"  Squared norm (||uncond||²): {squared_norm.item():.6f}")
    print(f"  Optimized scale (st_star): {st_star.item():.6f}")
    
    # Reshape for broadcasting
    st_star_broadcast = st_star.view(B, 1, 1, 1, 1)
    
    # Apply CFG-zero formula
    noise_pred_cfg = (
        noise_pred_uncond * st_star_broadcast + 
        guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star_broadcast)
    )
    
    save_tensor_stats(noise_pred_cfg, "After CFG-zero (before negation)", "[5]")
    visualize_tensor(noise_pred_cfg, "after_cfg_zero", "step4")
    
    # Compare with standard CFG
    noise_pred_standard_cfg = (
        noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    )
    
    diff_cfg = torch.abs(noise_pred_cfg - noise_pred_standard_cfg).float()
    print(f"\n  Difference from standard CFG:")
    print(f"    Max diff: {diff_cfg.max().item():.6f}")
    print(f"    Mean diff: {diff_cfg.mean().item():.6f}")
    
    # ========================================================================
    # 6. NEGATE NOISE PREDICTION
    # ========================================================================
    print("\n" + "="*100)
    print("[6] NEGATING NOISE PREDICTION (Flow Matching)")
    print("="*100)
    
    noise_pred_before_neg = noise_pred_cfg.clone()
    noise_pred_negated = -noise_pred_cfg
    
    save_tensor_stats(noise_pred_before_neg, "Before Negation", "[6]")
    save_tensor_stats(noise_pred_negated, "After Negation", "[6]")
    
    visualize_tensor(noise_pred_negated, "after_negation", "step5")
    
    # Verify negation
    diff_neg = torch.abs(noise_pred_negated + noise_pred_before_neg).float()
    print(f"\n  Negation verification (should be ~0): {diff_neg.max().item():.6e}")
    
    # ========================================================================
    # 7. SIMULATE SCHEDULER STEP
    # ========================================================================
    print("\n" + "="*100)
    print("[7] SIMULATING SCHEDULER STEP")
    print("="*100)
    
    # Load scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video",
        subfolder="scheduler"
    )
    scheduler.set_timesteps(50)
    
    # Find timestep index
    timestep_value = timestep.item()
    timestep_indices = [i for i, t in enumerate(scheduler.timesteps) if abs(t.item() - timestep_value) < 0.1]
    
    if timestep_indices:
        timestep_idx = timestep_indices[0]
        print(f"  Using timestep index: {timestep_idx}/{len(scheduler.timesteps)}")
    else:
        timestep_idx = 25  # Middle of schedule
        print(f"  Using default timestep index: {timestep_idx}")
    
    t_scheduler = scheduler.timesteps[timestep_idx]
    
    # Test with negated prediction (correct)
    latents_after_step_negated = scheduler.step(
        noise_pred_negated,
        t_scheduler,
        latents,
        return_dict=False
    )[0]
    
    # Test WITHOUT negation (incorrect, for comparison)
    latents_after_step_no_neg = scheduler.step(
        noise_pred_before_neg,
        t_scheduler,
        latents,
        return_dict=False
    )[0]
    
    print("\n  With negation (correct):")
    save_tensor_stats(latents_after_step_negated, "Latents after step (negated)", "[7]")
    
    print("\n  WITHOUT negation (incorrect):")
    save_tensor_stats(latents_after_step_no_neg, "Latents after step (no neg)", "[7]")
    
    visualize_tensor(latents_after_step_negated, "latents_after_negated", "step6")
    visualize_tensor(latents_after_step_no_neg, "latents_after_no_neg", "step6")
    
    # Compare
    diff_scheduler = torch.abs(latents_after_step_negated - latents_after_step_no_neg).float()
    print(f"\n  Difference (negated vs no negation):")
    print(f"    Max diff: {diff_scheduler.max().item():.6f}")
    print(f"    Mean diff: {diff_scheduler.mean().item():.6f}")
    
    # ========================================================================
    # 8. CHECK PARAMETER ORDER
    # ========================================================================
    print("\n" + "="*100)
    print("[8] CHECKING PARAMETER ORDER")
    print("="*100)
    
    # Test with WRONG parameter order (LongCat original order)
    print("\n  Testing WRONG order (hidden_states, timestep, encoder_hidden_states)...")
    try:
        # This should give nonsensical results if order matters
        # We'll manually mess up the call by swapping parameters
        with torch.no_grad():
            # Simulate wrong order by passing timestep where text should be
            # This will fail since timestep is 1D but we need to make it look like text
            
            # Create fake "text" from timestep (broadcast to match expected shape)
            fake_text_from_timestep = torch.ones(
                batch_size, 256, 4096, dtype=torch.bfloat16, device=device
            ) * timestep.item()
            
            # Create fake "timestep" from text (take mean as scalar)
            fake_timestep_from_text = text_emb.float().mean() * torch.ones(
                batch_size, device=device
            )
            
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=dummy_batch,
            ):
                output_wrong_order = model(
                    hidden_states=latents,
                    encoder_hidden_states=fake_text_from_timestep,  # WRONG: timestep data
                    timestep=fake_timestep_from_text,  # WRONG: text data
                )
        
        save_tensor_stats(output_wrong_order, "Output with WRONG parameter order", "[8]")
        visualize_tensor(output_wrong_order, "output_wrong_order", "step7")
        
        # Compare with correct order
        diff_order = torch.abs(output_single - output_wrong_order).float()
        print(f"\n  Difference from correct order:")
        print(f"    Max diff: {diff_order.max().item():.6f}")
        print(f"    Mean diff: {diff_order.mean().item():.6f}")
        
        if diff_order.mean().item() < 0.01:
            print("  ⚠️  WARNING: Wrong parameter order gives similar results! Check implementation!")
        else:
            print("  ✅ Parameter order matters (good)")
            
    except Exception as e:
        print(f"  ✅ Wrong parameter order failed as expected: {e}")
    
    # ========================================================================
    # 9. SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("[9] DEBUGGING SUMMARY")
    print("="*100)
    
    checks = []
    
    # Check 1: Output magnitude
    output_magnitude = output_single.float().abs().mean().item()
    if 0.01 < output_magnitude < 100:
        checks.append(("✅", "Output magnitude is reasonable", output_magnitude))
    else:
        checks.append(("❌", "Output magnitude is unusual", output_magnitude))
    
    # Check 2: CFG produces different results
    cfg_diff = torch.abs(noise_pred_cond - noise_pred_uncond).float().mean().item()
    if cfg_diff > 0.01:
        checks.append(("✅", "CFG produces different conditional/unconditional", cfg_diff))
    else:
        checks.append(("❌", "CFG not producing different results", cfg_diff))
    
    # Check 3: CFG-zero vs standard CFG
    cfgzero_diff = diff_cfg.mean().item()
    if cfgzero_diff > 0.001:
        checks.append(("✅", "CFG-zero differs from standard CFG", cfgzero_diff))
    else:
        checks.append(("⚠️ ", "CFG-zero very similar to standard CFG", cfgzero_diff))
    
    # Check 4: Negation makes a difference
    neg_effect = diff_scheduler.mean().item()
    if neg_effect > 0.01:
        checks.append(("✅", "Negation affects scheduler step", neg_effect))
    else:
        checks.append(("❌", "Negation has no effect on scheduler", neg_effect))
    
    # Check 5: st_star value
    st_star_val = st_star.item()
    if -1 < st_star_val < 2:
        checks.append(("✅", "CFG-zero st_star is in reasonable range", st_star_val))
    else:
        checks.append(("⚠️ ", "CFG-zero st_star is unusual", st_star_val))
    
    print("\n  Diagnostic Checks:")
    for icon, msg, val in checks:
        print(f"    {icon} {msg}: {val:.6f}")
    
    # Final assessment
    failed_checks = sum(1 for icon, _, _ in checks if icon == "❌")
    warning_checks = sum(1 for icon, _, _ in checks if icon == "⚠️ ")
    
    print(f"\n  {'='*80}")
    if failed_checks == 0 and warning_checks == 0:
        print("  ✅ ALL CHECKS PASSED - Implementation looks correct!")
        print("  If generation still produces noise, the issue may be:")
        print("    - Incorrect weights loaded")
        print("    - Wrong scheduler configuration")
        print("    - Issue in VAE encoding/decoding")
        print("    - Wrong text encoder or embeddings")
    elif failed_checks > 0:
        print(f"  ❌ {failed_checks} CRITICAL ISSUES FOUND")
        print("  Review the failed checks above")
    else:
        print(f"  ⚠️  {warning_checks} WARNINGS")
        print("  Review warnings but may not be critical")
    print(f"  {'='*80}")
    
    print(f"\n  📁 All visualizations saved to: {output_dir}")
    
    return {
        "output_single": output_single,
        "noise_pred_uncond": noise_pred_uncond,
        "noise_pred_cond": noise_pred_cond,
        "noise_pred_cfg": noise_pred_cfg,
        "noise_pred_negated": noise_pred_negated,
        "latents_after_step": latents_after_step_negated,
    }


def compare_with_original():
    """Compare native implementation with original LongCat."""
    print("\n\n" + "="*100)
    print("COMPARING NATIVE VS ORIGINAL IMPLEMENTATION")
    print("="*100)
    
    # TODO: Add comparison with original if needed
    pass


if __name__ == "__main__":
    results = test_native_with_instrumentation()
    print("\n✅ Debugging complete!")

