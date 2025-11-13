#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test LongCat native inference with and without LoRA adapters.

Modes:
- standard: 50 steps, 480p, no LoRA
- distilled: 16 steps, 480p, with distilled LoRA
- refinement: 50 steps, 720p, with refinement LoRA (requires BSA - not implemented)

Usage:
    # Standard generation (50 steps, 480p)
    python test_longcat_lora_inference.py --mode standard

    # Distilled generation (16 steps, 480p with LoRA)
    python test_longcat_lora_inference.py --mode distilled

    # Refinement (720p with LoRA) - not yet implemented
    python test_longcat_lora_inference.py --mode refinement
"""

import argparse
import time
from pathlib import Path

import torch
from fastvideo import VideoGenerator


def test_standard(model_path: str, num_gpus: int = 1, num_steps: int = 50):
    """Test standard generation (50 steps, 480p)."""
    
    prompt = "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    seed = 42
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print(f"Standard Generation ({num_steps} steps, 480p)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    start_time = time.time()
    
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        use_fsdp_inference=(num_gpus > 1),
        dit_cpu_offload=False,
    )
    
    output_name = f"output_t2v_{num_steps}steps.mp4" if num_steps != 50 else "output_t2v.mp4"
    output_path = output_dir / output_name
    video = generator.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=num_steps,
        guidance_scale=4.0,
        seed=seed,
        output_path=str(output_path),
        save_video=True,
        return_frames=True,
    )
    
    total_time = time.time() - start_time
    print(f"✓ Generated in {total_time:.2f}s → {output_path}")
    print()


def test_distilled(model_path: str, num_gpus: int = 1):
    """Test distilled LoRA generation (16 steps, 480p)."""
    
    prompt = "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    
    seed = 42
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Distilled Generation (16 steps, 480p with LoRA)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"LoRA: {model_path}/lora/distilled")
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    start_time = time.time()
    
    lora_path = f"{model_path}/lora/distilled"
    generator = VideoGenerator.from_pretrained(
        model_path,
        lora_path=lora_path,
        lora_nickname="distilled",
        num_gpus=num_gpus,
        use_fsdp_inference=(num_gpus > 1),
        dit_cpu_offload=False,
    )
    
    output_path = output_dir / "output_t2v_distill.mp4"
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
    
    total_time = time.time() - start_time
    print(f"✓ Generated in {total_time:.2f}s → {output_path}")
    print(f"  Speedup: ~3x faster than standard (16 vs 50 steps)")
    print()


def test_refinement(model_path: str, num_gpus: int = 1):
    """Test refinement LoRA (720p) - placeholder for future implementation."""
    
    print("=" * 80)
    print("Refinement Generation (720p with LoRA)")
    print("=" * 80)
    print()
    print("⚠️  Not implemented: Requires Block Sparse Attention (BSA)")
    print()
    print("To implement:")
    print("  1. Block Sparse Attention in fastvideo/models/dits/longcat.py")
    print("  2. enable_bsa() method in LongCatTransformer3DModel")
    print("  3. generate_refine() in pipeline for i2v upscaling")
    print()
    print("Reference: /mnt/fast-disks/hao_lab/shao/LongCat-Video/")
    print("  - longcat_video/block_sparse_attention/bsa_interface.py")
    print("  - longcat_video/modules/longcat_video_dit.py (enable_bsa)")
    print("  - longcat_video/pipeline_longcat_video.py (generate_refine)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test LongCat inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/longcat-native",
        help="Path to converted native weights",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "standard-16", "distilled", "refinement"],
        default="distilled",
        help="Test mode: standard (50 steps), standard-16 (16 steps for comparison), distilled (16 steps with LoRA), refinement (720p with LoRA)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    
    args = parser.parse_args()
    
    if args.mode == "standard":
        test_standard(args.model_path, args.num_gpus, num_steps=50)
    elif args.mode == "standard-16":
        test_standard(args.model_path, args.num_gpus, num_steps=16)
    elif args.mode == "distilled":
        test_distilled(args.model_path, args.num_gpus)
    elif args.mode == "refinement":
        test_refinement(args.model_path, args.num_gpus)


if __name__ == "__main__":
    main()
