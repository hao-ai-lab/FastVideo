#!/usr/bin/env python3
"""
Test LongCat video generation through FastVideo.
"""

import os
import torch
from fastvideo import VideoGenerator
from fastvideo.logger import init_logger

logger = init_logger(__name__)

MODEL_PATH = "weights/longcat-for-fastvideo"
OUTPUT_DIR = "outputs/longcat_test"

def test_inference():
    print("\n" + "=" * 80)
    print("                    LONGCAT INFERENCE TEST")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"\n1. Loading LongCat model from: {MODEL_PATH}")
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,  # Offload to manage memory
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
    )
    print("   ✓ Model loaded successfully!")
    
    # Test parameters
    prompt = "A cat playing piano, high quality, cinematic"
    
    print(f"\n2. Generating video...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Output: {OUTPUT_DIR}")
    
    try:
        # Generate video
        video = generator.generate_video(
            prompt=prompt,
            output_path=OUTPUT_DIR,
            save_video=True,
            num_inference_steps=20,  # Reduced for faster testing
            guidance_scale=4.0,      # LongCat's default guidance scale
            height=480,              # Match LongCatT2V480PConfig
            width=832,               # Match LongCatT2V480PConfig
            num_frames=65,           # Reduced for faster testing (3s @ 21fps)
            seed=42,                 # For reproducibility
        )
        
        print("\n" + "=" * 80)
        print("                         ✓ INFERENCE SUCCESSFUL!")
        print("=" * 80)
        print(f"\nVideo saved to: {OUTPUT_DIR}")
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("                         ✗ INFERENCE FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_inference()

