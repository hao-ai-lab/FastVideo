"""
Test a single forward pass and check intermediate values.
"""

import torch
import sys
sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalModel

def test_original_model():
    print("="*80)
    print("Testing Original LongCat Model Forward Pass")
    print("="*80)
    
    print("\n[1] Loading model...")
    dit_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit"
    
    model = OriginalModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda").eval()
    print("  ✓ Model loaded")
    
    print("\n[2] Creating test input...")
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Input latent - using small values like real latents
    x = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.18215  # Scale like VAE latents
    
    # Timesteps
    t = torch.tensor([500.0], device="cuda")
    
    # Text embeddings
    text_emb = torch.randn(batch_size, 256, 4096, dtype=torch.bfloat16, device="cuda")
    
    print(f"  Input shape: {x.shape}")
    print(f"  Input stats: mean={x.float().mean().item():.4f}, std={x.float().std().item():.4f}")
    print(f"  Timestep: {t.item()}")
    print(f"  Text embedding shape: {text_emb.shape}")
    
    print("\n[3] Running forward pass...")
    with torch.no_grad():
        try:
            output = model(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=text_emb,
            )
            
            print(f"  ✓ Forward pass successful!")
            print(f"\n[4] Output analysis:")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            
            output_fp32 = output.float()
            print(f"\n  Statistics:")
            print(f"    Mean: {output_fp32.mean().item():.6f}")
            print(f"    Std:  {output_fp32.std().item():.6f}")
            print(f"    Min:  {output_fp32.min().item():.6f}")
            print(f"    Max:  {output_fp32.max().item():.6f}")
            
            # Check if output looks reasonable
            if output_fp32.abs().mean() < 1e-5:
                print(f"\n  ✗ WARNING: Output is nearly zero (possible dead network)")
            elif output_fp32.abs().mean() > 1000:
                print(f"\n  ✗ WARNING: Output has very large values (possible numerical instability)")
            elif torch.isnan(output_fp32).any():
                print(f"\n  ✗ WARNING: Output contains NaN values")
            elif torch.isinf(output_fp32).any():
                print(f"\n  ✗ WARNING: Output contains Inf values")
            else:
                print(f"\n  ✓ Output looks reasonable")
            
            # Check output distribution
            print(f"\n  Distribution check:")
            print(f"    |values| < 0.01: {(output_fp32.abs() < 0.01).float().mean().item()*100:.1f}%")
            print(f"    |values| < 0.1:  {(output_fp32.abs() < 0.1).float().mean().item()*100:.1f}%")
            print(f"    |values| < 1.0:  {(output_fp32.abs() < 1.0).float().mean().item()*100:.1f}%")
            print(f"    |values| > 10:   {(output_fp32.abs() > 10).float().mean().item()*100:.1f}%")
            
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

def test_native_model():
    print("\n\n" + "="*80)
    print("Testing FastVideo Native LongCat Model Forward Pass")
    print("="*80)
    
    print("\n[1] Loading native model directly...")
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoArchConfig
    from fastvideo.configs.models.base import ModelConfig
    from safetensors.torch import load_file
    import json
    
    try:
        transformer_path = "weights/longcat-native/transformer"
        
        # Load config
        with open(f"{transformer_path}/config.json") as f:
            config_dict = json.load(f)
        
        # Create model config
        model_config = ModelConfig(arch_config=LongCatVideoArchConfig())
        
        # Create model
        model = LongCatTransformer3DModel(
            config=model_config,
            hf_config=config_dict,
        )
        
        # Load weights
        state_dict = load_file(f"{transformer_path}/model.safetensors")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        model = model.to("cuda").to(torch.bfloat16).eval()
        print("  ✓ Native model loaded")
        
    except Exception as e:
        print(f"  ✗ Failed to load native model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[2] Creating test input (same as original)...")
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    # Set seed for reproducibility (same as original)
    torch.manual_seed(42)
    
    # Input latent - using small values like real latents
    x = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.18215
    
    # Timesteps
    t = torch.tensor([500.0], device="cuda")
    
    # Text embeddings
    text_emb = torch.randn(batch_size, 256, 4096, dtype=torch.bfloat16, device="cuda")
    
    print(f"  Input shape: {x.shape}")
    print(f"  Input stats: mean={x.float().mean().item():.4f}, std={x.float().std().item():.4f}")
    print(f"  Timestep: {t.item()}")
    print(f"  Text embedding shape: {text_emb.shape}")
    
    print("\n[3] Running forward pass...")
    with torch.no_grad():
        try:
            output = model(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=text_emb,
            )
            
            print(f"  ✓ Forward pass successful!")
            print(f"\n[4] Output analysis:")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            
            output_fp32 = output.float()
            print(f"\n  Statistics:")
            print(f"    Mean: {output_fp32.mean().item():.6f}")
            print(f"    Std:  {output_fp32.std().item():.6f}")
            print(f"    Min:  {output_fp32.min().item():.6f}")
            print(f"    Max:  {output_fp32.max().item():.6f}")
            
            # Check if output looks reasonable
            if output_fp32.abs().mean() < 1e-5:
                print(f"\n  ✗ WARNING: Output is nearly zero (possible dead network)")
            elif output_fp32.abs().mean() > 1000:
                print(f"\n  ✗ WARNING: Output has very large values (possible numerical instability)")
            elif torch.isnan(output_fp32).any():
                print(f"\n  ✗ WARNING: Output contains NaN values")
            elif torch.isinf(output_fp32).any():
                print(f"\n  ✗ WARNING: Output contains Inf values")
            else:
                print(f"\n  ✓ Output looks reasonable")
            
            # Check output distribution
            print(f"\n  Distribution check:")
            print(f"    |values| < 0.01: {(output_fp32.abs() < 0.01).float().mean().item()*100:.1f}%")
            print(f"    |values| < 0.1:  {(output_fp32.abs() < 0.1).float().mean().item()*100:.1f}%")
            print(f"    |values| < 1.0:  {(output_fp32.abs() < 1.0).float().mean().item()*100:.1f}%")
            print(f"    |values| > 10:   {(output_fp32.abs() > 10).float().mean().item()*100:.1f}%")
            
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

def compare_models():
    """Compare both models with the same inputs."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    
    print("="*80)
    print("Comparing Original vs Native LongCat Models")
    print("="*80)
    
    # Create shared test inputs
    print("\n[1] Creating test inputs...")
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    torch.manual_seed(42)
    
    x = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.18215
    
    t = torch.tensor([500.0], device="cuda", dtype=torch.float32)
    text_emb = torch.randn(batch_size, 1, 256, 4096, dtype=torch.bfloat16, device="cuda")  # [B, 1, N_tokens, C]
    
    print(f"  Input shape: {x.shape}")
    print(f"  Input stats: mean={x.float().mean().item():.4f}, std={x.float().std().item():.4f}")
    print(f"  Text emb shape: {text_emb.shape}")
    
    # Test original model
    print("\n[2] Testing ORIGINAL model...")
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalModel
    
    dit_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit"
    orig_model = OriginalModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
        cp_split_hw=[1, 1],  # Single GPU inference
    )
    orig_model = orig_model.to("cuda").eval()
    print("  ✓ Loaded")
    
    with torch.no_grad():
        orig_output = orig_model(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=text_emb,  # [B, 1, N, C] for original
        )
    
    print(f"  Output shape: {orig_output.shape}")
    print(f"  Output stats: mean={orig_output.float().mean().item():.6f}, std={orig_output.float().std().item():.6f}")
    
    # Clear memory
    del orig_model
    torch.cuda.empty_cache()
    
    # Test native model
    print("\n[3] Testing NATIVE model...")
    from fastvideo.models.dits.longcat import LongCatTransformer3DModel
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.distributed.parallel_state import initialize_model_parallel
    from safetensors.torch import load_file
    import json
    
    # Initialize distributed environment
    initialize_model_parallel()
    
    transformer_path = "weights/longcat-native/transformer"
    
    with open(f"{transformer_path}/config.json") as f:
        config_dict = json.load(f)
    
    # Use LongCatVideoConfig which includes prefix
    model_config = LongCatVideoConfig()
    native_model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)
    
    state_dict = load_file(f"{transformer_path}/model.safetensors")
    native_model.load_state_dict(state_dict, strict=False)
    native_model = native_model.to("cuda").to(torch.bfloat16).eval()
    print("  ✓ Loaded")
    
    with torch.no_grad():
        # Native model expects [B, N, C] not [B, 1, N, C]
        text_emb_native = text_emb.squeeze(1)  # [B, N, C]
        native_output = native_model(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=text_emb_native,
        )
    
    print(f"  Output shape: {native_output.shape}")
    print(f"  Output stats: mean={native_output.float().mean().item():.6f}, std={native_output.float().std().item():.6f}")
    
    # Compare outputs
    print("\n[4] Comparing outputs...")
    if orig_output.shape == native_output.shape:
        diff = torch.abs(orig_output.float() - native_output.float())
        print(f"  Max difference: {diff.max().item():.6e}")
        print(f"  Mean difference: {diff.mean().item():.6e}")
        print(f"  Median difference: {diff.median().item():.6e}")
        
        rel_diff = diff / (orig_output.float().abs() + 1e-8)
        print(f"  Mean relative error: {rel_diff.mean().item():.6e}")
        
        if diff.max().item() < 1e-4:
            print("\n  ✓ Outputs are VERY SIMILAR!")
        elif diff.max().item() < 1e-2:
            print("\n  ✓ Outputs are similar (small numerical differences)")
        elif diff.max().item() < 0.1:
            print("\n  ⚠ Outputs have moderate differences")
        else:
            print("\n  ✗ Outputs are SIGNIFICANTLY DIFFERENT!")
            
        # Distribution of differences
        print(f"\n  Difference distribution:")
        print(f"    < 1e-5: {(diff < 1e-5).float().mean().item()*100:.1f}%")
        print(f"    < 1e-4: {(diff < 1e-4).float().mean().item()*100:.1f}%")
        print(f"    < 1e-3: {(diff < 1e-3).float().mean().item()*100:.1f}%")
        print(f"    < 1e-2: {(diff < 1e-2).float().mean().item()*100:.1f}%")
        print(f"    < 0.1:  {(diff < 0.1).float().mean().item()*100:.1f}%")
        print(f"    > 0.1:  {(diff > 0.1).float().mean().item()*100:.1f}%")
    else:
        print(f"  ✗ Shape mismatch: {orig_output.shape} vs {native_output.shape}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")
    
    compare_models()

