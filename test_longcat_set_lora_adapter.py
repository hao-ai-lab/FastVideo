#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test LongCat distilled LoRA using set_lora_adapter method.

This tests the alternative way of loading LoRA adapters by:
1. First creating a VideoGenerator without LoRA
2. Then calling set_lora_adapter() to load the LoRA

Usage:
    python test_longcat_set_lora_adapter.py
"""

import time
from pathlib import Path

import torch
from fastvideo import VideoGenerator


def test_distilled_with_set_lora_adapter(
    model_path: str = "weights/longcat-native",
    num_gpus: int = 1
):
    """Test distilled LoRA generation using set_lora_adapter method."""
    
    prompt = "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    
    seed = 42
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Testing set_lora_adapter with LongCat Distilled LoRA")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Method: set_lora_adapter (NOT from_pretrained lora_path)")
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    start_time = time.time()
    
    # Step 1: Create generator WITHOUT LoRA
    print("Step 1: Creating VideoGenerator without LoRA...")
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        use_fsdp_inference=(num_gpus > 1),
        dit_cpu_offload=False,
    )
    init_time = time.time() - start_time
    print(f"✓ Generator initialized in {init_time:.2f}s")
    print()
    
    # Step 2: Load LoRA using set_lora_adapter
    print("Step 2: Loading LoRA using set_lora_adapter()...")
    lora_load_start = time.time()
    lora_path = f"{model_path}/lora/distilled"
    
    try:
        generator.set_lora_adapter(
            lora_nickname="distilled",
            lora_path=lora_path
        )
        lora_load_time = time.time() - lora_load_start
        print(f"✓ LoRA loaded in {lora_load_time:.2f}s")
        print(f"  LoRA path: {lora_path}")
        print()
    except Exception as e:
        print(f"✗ FAILED to load LoRA: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Generate video
    print("Step 3: Generating video...")
    gen_start = time.time()
    output_path = output_dir / "output_t2v_set_lora_adapter.mp4"
    
    try:
        video = generator.generate_video(
            prompt=prompt,
            negative_prompt=None,  # Not used in distilled
            height=480,
            width=832,
            num_frames=93,
            num_inference_steps=16,
            guidance_scale=1.0,
            seed=seed,
            output_path=str(output_path),
            save_video=True,
            return_frames=True,
        )
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        print(f"✓ Generated in {gen_time:.2f}s → {output_path}")
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Initialization: {init_time:.2f}s")
        print(f"  LoRA loading:   {lora_load_time:.2f}s")
        print(f"  Generation:     {gen_time:.2f}s")
        print(f"  Total:          {total_time:.2f}s")
        print()
        print("✓ TEST PASSED: set_lora_adapter method works!")
        return True
        
    except Exception as e:
        print(f"✗ FAILED to generate video: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LongCat distilled LoRA using set_lora_adapter"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/longcat-native",
        help="Path to converted native weights",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    
    args = parser.parse_args()
    
    success = test_distilled_with_set_lora_adapter(
        model_path=args.model_path,
        num_gpus=args.num_gpus
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

