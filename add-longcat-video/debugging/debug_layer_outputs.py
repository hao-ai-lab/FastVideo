"""
Compare layer-by-layer outputs between original LongCat and FastVideo native implementation.
"""

import torch
import sys
sys.path.insert(0, "/mnt/fast-disks/hao_lab/shao/LongCat-Video")

from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel as OriginalLongCatModel
from fastvideo.models.dits.longcat import LongCatTransformer3DModel
from fastvideo.configs.models.dits.longcat import LongCatVideoArchConfig

def load_original_model():
    """Load the original third-party LongCat model."""
    print("\n[1] Loading original LongCat model...")
    
    dit_path = "/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/dit"
    
    model = OriginalLongCatModel.from_pretrained(
        dit_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda").eval()
    
    print(f"  ✓ Loaded original model")
    return model

def load_native_model():
    """Load the FastVideo native LongCat model."""
    print("\n[2] Loading FastVideo native model...")
    
    # Use the same VideoGenerator to load the model
    from fastvideo import VideoGenerator
    
    generator = VideoGenerator.from_pretrained(
        "weights/longcat-native",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
    )
    
    print(f"  ✓ Loaded native model via VideoGenerator")
    return generator

def compare_outputs():
    """Compare outputs of key layers."""
    print("="*80)
    print("Layer-by-Layer Output Comparison")
    print("="*80)
    
    # Load models
    original_model = load_original_model()
    native_model = load_native_model()
    
    print("\n[3] Creating test input...")
    # Create a small test input
    batch_size = 1
    num_frames = 9
    height = 30
    width = 52
    latent_channels = 16
    
    # Input latent
    x = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        dtype=torch.bfloat16, device="cuda"
    )
    
    # Timesteps
    t = torch.tensor([500.0], device="cuda")
    
    # Text embeddings (using dummy for now)
    text_emb = torch.randn(batch_size, 256, 4096, dtype=torch.bfloat16, device="cuda")
    
    print(f"  Input shape: {x.shape}")
    print(f"  Timestep: {t}")
    print(f"  Text embedding shape: {text_emb.shape}")
    
    print("\n[4] Running forward pass...")
    
    with torch.no_grad():
        # Original model
        print("\n  Running original model...")
        try:
            orig_output = original_model(
                x=x,
                timestep=t,
                context=text_emb,
            )
            print(f"    Original output shape: {orig_output.shape}")
            print(f"    Original output stats:")
            print(f"      Mean: {orig_output.float().mean().item():.6f}")
            print(f"      Std:  {orig_output.float().std().item():.6f}")
            print(f"      Min:  {orig_output.float().min().item():.6f}")
            print(f"      Max:  {orig_output.float().max().item():.6f}")
        except Exception as e:
            print(f"    ✗ Error in original model: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Native model  
        print("\n  Running native model...")
        try:
            # Reshape input for native model (expects B, C, T, H, W)
            native_output = native_model(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=text_emb,
            )
            print(f"    Native output shape: {native_output.shape}")
            print(f"    Native output stats:")
            print(f"      Mean: {native_output.float().mean().item():.6f}")
            print(f"      Std:  {native_output.float().std().item():.6f}")
            print(f"      Min:  {native_output.float().min().item():.6f}")
            print(f"      Max:  {native_output.float().max().item():.6f}")
        except Exception as e:
            print(f"    ✗ Error in native model: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n[5] Comparing outputs...")
    if orig_output.shape == native_output.shape:
        diff = torch.abs(orig_output - native_output)
        print(f"  Max difference: {diff.max().item():.6e}")
        print(f"  Mean difference: {diff.mean().item():.6e}")
        print(f"  Relative error: {(diff.mean() / orig_output.abs().mean()).item():.6e}")
        
        # Check if outputs are similar
        if diff.max().item() < 0.01:
            print("\n  ✓ Outputs are very similar!")
        elif diff.max().item() < 0.1:
            print("\n  ⚠ Outputs have small differences")
        else:
            print("\n  ✗ Outputs are significantly different!")
    else:
        print(f"  ✗ Shape mismatch: {orig_output.shape} vs {native_output.shape}")

if __name__ == "__main__":
    compare_outputs()

