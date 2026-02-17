#!/usr/bin/env python3
"""
FINAL TEST: Generate with EXACT SGLang parameters (with corrected dimensions!)

CRITICAL FIX: SGLang generated 1280×720 (landscape), but we were testing 720×1280 (portrait)!
This test uses the CORRECT dimensions that match SGLang's output.
"""
import sys
sys.path.insert(0, '/FastVideo')

import os
import numpy as np
from PIL import Image

def main():
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.environ["_FLASH_ATTN_DISABLED"] = "1"

    print("="*80)
    print("FINAL TEST: FastVideo with CORRECTED SGLang-matching parameters")
    print("="*80)

    # SGLang's actual output dimensions
    PROMPT = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    WIDTH = 1280   # SGLang image is 1280 wide
    HEIGHT = 720   # SGLang image is 720 tall
    SEED = 42      # Need to determine SGLang's actual seed
    STEPS = 50     # Need to determine SGLang's actual steps

    print(f"\nParameters:")
    print(f"  prompt: {PROMPT[:50]}...")
    print(f"  width:  {WIDTH} (CORRECTED from 720)")
    print(f"  height: {HEIGHT} (CORRECTED from 1280)")
    print(f"  seed:   {SEED} (assumed)")
    print(f"  steps:  {STEPS} (assumed)")

    print("\n" + "="*80)
    print("Loading FastVideo and generating...")
    print("="*80)

    from fastvideo import VideoGenerator

    gen = VideoGenerator.from_pretrained(
        model_path="black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        text_encoder_cpu_offload=True,
    )

    result = gen.generate_video(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=1,
        seed=SEED,
        num_inference_steps=STEPS,
        guidance_scale=1.0,  # Match SGLang's guidance_scale
        save_video=False,
        return_frames=True,
    )

    fv_frame = np.array(result[0])

    print("\n" + "="*80)
    print("FastVideo Output")
    print("="*80)
    print(f"Shape: {fv_frame.shape}")
    print(f"  Width×Height: {fv_frame.shape[1]}×{fv_frame.shape[0]}")
    print(f"  Expected: {WIDTH}×{HEIGHT}")
    print(f"  Match: {fv_frame.shape[1] == WIDTH and fv_frame.shape[0] == HEIGHT}")
    print(f"\nPixel stats:")
    print(f"  min={fv_frame.min()}, max={fv_frame.max()}, mean={fv_frame.mean():.2f}")
    print(f"  Top-left corner: {fv_frame[:2, :2, :]}")

    # Save for comparison
    output_path = "fastvideo_corrected_dimensions.png"
    Image.fromarray(fv_frame).save(output_path)
    print(f"\nSaved to: {output_path}")

    # Load and compare with SGLang
    sg_path = "A_cinematic_portrait_of_a_fox_35mm_film_soft_light_gentle_grain._20260215-132609_878dfa3f.png"
    print("\n" + "="*80)
    print("SGLang Output (pre-generated)")
    print("="*80)
    sg_image = Image.open(sg_path)
    sg_frame = np.array(sg_image)
    print(f"Shape: {sg_frame.shape}")
    print(f"  Width×Height: {sg_frame.shape[1]}×{sg_frame.shape[0]}")
    print(f"\nPixel stats:")
    print(f"  min={sg_frame.min()}, max={sg_frame.max()}, mean={sg_frame.mean():.2f}")
    print(f"  Top-left corner: {sg_frame[:2, :2, :]}")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    if fv_frame.shape == sg_frame.shape:
        diff = np.abs(fv_frame.astype(float) - sg_frame.astype(float))
        print(f"✅ Shapes match: {fv_frame.shape}")
        print(f"\nPixel difference:")
        print(f"  max abs diff:  {diff.max():.2f}")
        print(f"  mean abs diff: {diff.mean():.2f}")
        print(f"  std abs diff:  {diff.std():.2f}")
        
        # Pixel-wise comparison
        close_pixels = (diff.mean(axis=2) < 5).sum()
        total_pixels = diff.shape[0] * diff.shape[1]
        print(f"\n  Close pixels (diff<5): {close_pixels}/{total_pixels} ({100*close_pixels/total_pixels:.1f}%)")
        
        if diff.mean() < 1.0:
            print("\n🎉 EXCELLENT MATCH (mean diff < 1 pixel)")
        elif diff.mean() < 5.0:
            print("\n✅ GOOD MATCH (mean diff < 5 pixels)")
        elif diff.mean() < 20.0:
            print("\n⚠️  MODERATE DIFFERENCE (mean diff < 20 pixels)")
        else:
            print("\n❌ SIGNIFICANT DIFFERENCE (mean diff >= 20 pixels)")
            print("\nPossible causes:")
            print("  - Different random seed (SGLang may use different default)")
            print("  - Different num_inference_steps (SGLang may use different default)")
            print("  - Different guidance parameters")
            print("  - Different model precision or attention backend")
    else:
        print(f"❌ Shape mismatch!")
        print(f"  FastVideo: {fv_frame.shape}")
        print(f"  SGLang:    {sg_frame.shape}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
We corrected the critical dimension error:
  - Original test: 720×1280 (portrait) 
  - Corrected test: 1280×720 (landscape) ✅

This test will show whether dimensions were the only issue, or if there
are other parameter differences (seed, steps, etc.) between FastVideo
and SGLang's defaults that we still need to match.
""")

if __name__ == '__main__':
    main()
