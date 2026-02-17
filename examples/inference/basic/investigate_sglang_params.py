#!/usr/bin/env python3
"""
Try to reverse-engineer what SGLang parameters were used by testing with different seeds/steps
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/FastVideo')

from fastvideo.pipelines import FluxPipeline

def load_image(path):
    """Load image and return as numpy array"""
    img = Image.open(path)
    return np.array(img)

def compare_images(img1, img2):
    """Compare two images and return metrics"""
    if img1.shape != img2.shape:
        return None
    
    diff = np.abs(img1.astype(float) - img2.astype(float))
    return {
        'max': diff.max(),
        'mean': diff.mean(),
        'std': diff.std(),
        'close_5': (diff < 5).sum() / diff.size * 100,
        'close_10': (diff < 10).sum() / diff.size * 100,
    }

def main():
    print("=" * 80)
    print("INVESTIGATING SGLang PARAMETERS")
    print("=" * 80)
    
    # Load reference SGLang image
    sglang_path = Path("/FastVideo/examples/inference/basic/A_cinematic_portrait_of_a_fox_35mm_film_soft_light_gentle_grain._20260215-132609_878dfa3f.png")
    if not sglang_path.exists():
        print(f"ERROR: SGLang image not found at {sglang_path}")
        return
    
    sglang_img = load_image(sglang_path)
    print(f"\nSGLang reference image:")
    print(f"  Shape: {sglang_img.shape}")
    print(f"  Stats: min={sglang_img.min()}, max={sglang_img.max()}, mean={sglang_img.mean():.2f}")
    print(f"  Top-left 2x4:\n{sglang_img[:2, :4]}")
    
    # Initialize FastVideo generator
    print("\n" + "=" * 80)
    print("Testing FastVideo with different parameters...")
    print("=" * 80)
    
    gen = FluxPipeline.from_pretrained_basic(
        model_name="black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    
    prompt = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    
    # Test different parameter combinations
    test_configs = [
        {"seed": 42, "steps": 50, "guidance": 1.0, "label": "seed=42, steps=50 (our guess)"},
        {"seed": 0, "steps": 50, "guidance": 1.0, "label": "seed=0, steps=50"},
        {"seed": 42, "steps": 28, "guidance": 1.0, "label": "seed=42, steps=28"},
        {"seed": 0, "steps": 28, "guidance": 1.0, "label": "seed=0, steps=28"},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n[{i+1}/{len(test_configs)}] Testing: {config['label']}")
        
        try:
            output = gen(
                prompt=prompt,
                seed=config['seed'],
                num_inference_steps=config['steps'],
                guidance_scale=config['guidance'],
                width=1280,
                height=720,
            )
            
            if output is None or 'images' not in output:
                print("  ERROR: No output generated")
                continue
            
            # Extract first image
            images = output['images']
            if not images:
                print("  ERROR: No images in output")
                continue
            
            pil_image = images[0]
            frame = np.array(pil_image)
            
            # Ensure correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            
            # Compare
            comparison = compare_images(sglang_img, frame)
            
            if comparison:
                result = {
                    **config,
                    **comparison
                }
                results.append(result)
                
                print(f"  ✓ Generated successfully")
                print(f"    Shape: {frame.shape}")
                print(f"    Stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
                print(f"    Comparison to SGLang:")
                print(f"      max diff: {comparison['max']:.2f}")
                print(f"      mean diff: {comparison['mean']:.2f}")
                print(f"      close pixels (<5): {comparison['close_5']:.2f}%")
                print(f"      close pixels (<10): {comparison['close_10']:.2f}%")
            else:
                print(f"  ERROR: Shape mismatch")
        
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No successful tests completed")
        return
    
    # Sort by mean difference
    results.sort(key=lambda r: r['mean'])
    
    print("\nBest matches (sorted by mean pixel difference):\n")
    for i, r in enumerate(results[:3]):
        print(f"{i+1}. {r['label']}")
        print(f"   Mean diff: {r['mean']:.2f} pixels")
        print(f"   Max diff: {r['max']:.2f} pixels")
        print(f"   Close pixels (<5): {r['close_5']:.2f}%")
        print()

if __name__ == "__main__":
    main()
