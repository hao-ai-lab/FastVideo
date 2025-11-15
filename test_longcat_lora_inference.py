#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test LongCat native inference with and without LoRA adapters.

Modes:
- standard: 50 steps, 480p, no LoRA
- distilled: 16 steps, 480p, with distilled LoRA
- refinement: 50 steps, 720p, with refinement LoRA (requires BSA)
- distilled_refine: distilled (16 steps, 480p) + refinement (50 steps, 720p)

Usage:
    # Standard generation (50 steps, 480p)
    python test_longcat_lora_inference.py --mode standard

    # Distilled generation (16 steps, 480p with LoRA)
    python test_longcat_lora_inference.py --mode distilled

    # Distilled + Refinement pipeline (480p -> 720p)
    python test_longcat_lora_inference.py --mode distilled_refine

    # Refinement only (720p with LoRA)
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


def test_distilled_refine(model_path: str, num_gpus: int = 1):
    """Test distilled + refinement pipeline (480p -> 720p)."""
    
    prompt = "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    
    seed = 42
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Distilled + Refinement Pipeline (480p -> 720p)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"LoRA: {model_path}/lora/distilled -> {model_path}/lora/refinement")
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    total_start_time = time.time()
    
    # Stage 1: Distilled generation (480p, 16 steps)
    print("-" * 80)
    print("Stage 1: Distilled Generation (16 steps, 480p)")
    print("-" * 80)
    
    distill_start = time.time()
    
    distill_lora_path = f"{model_path}/lora/distilled"
    generator = VideoGenerator.from_pretrained(
        model_path,
        lora_path=distill_lora_path,
        lora_nickname="distilled",
        num_gpus=num_gpus,
        use_fsdp_inference=(num_gpus > 1),
        dit_cpu_offload=False,
    )
    
    output_distill_path = output_dir / "output_t2v_distill_stage1.mp4"
    video_distill = generator.generate_video(
        prompt=prompt,
        negative_prompt=None,
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=16,
        guidance_scale=1.0,
        seed=seed,
        output_path=str(output_distill_path),
        save_video=True,
        return_frames=True,
    )
    
    distill_time = time.time() - distill_start
    print(f"✓ Stage 1 completed in {distill_time:.2f}s → {output_distill_path}")
    print()
    
    # Clean up Stage 1 generator and free GPU memory
    print("Shutting down Stage 1 generator to free GPU memory...")
    generator.shutdown()
    del generator
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ Stage 1 resources released")
    print()
    
    # Stage 2: Refinement (720p, 50 steps)
    print("-" * 80)
    print("Stage 2: Refinement Generation (50 steps, 720p)")
    print("-" * 80)
    
    refine_start = time.time()
    
    # Reload generator with refinement LoRA and BSA enabled
    # Use aggressive offloading to save GPU memory for 720p generation
    refinement_lora_path = f"{model_path}/lora/refinement"
    generator = VideoGenerator.from_pretrained(
        model_path,
        lora_path=refinement_lora_path,
        lora_nickname="refinement",
        num_gpus=num_gpus,
        use_fsdp_inference=(num_gpus > 1),
        dit_cpu_offload=True,  # Enable CPU offload to save memory
        vae_cpu_offload=False,  # Keep VAE on GPU for speed
        text_encoder_cpu_offload=True,  # Offload text encoder
        enable_bsa=True,  # Enable BSA for refinement
        bsa_sparsity=0.875,
        bsa_chunk_q=[4, 4, 8],
        bsa_chunk_k=[4, 4, 8],
    )
    
    # Generate refined video at 720p using refine-from
    output_refine_path = output_dir / "output_t2v_refine.mp4"
    video_refine = generator.generate_video(
        prompt=prompt,
        stage1_video=video_distill,
        t_thresh=0.5,
        spatial_refine_only=False,
        num_cond_frames=0,
        height=720,
        width=1280,
        num_inference_steps=50,
        guidance_scale=1.0,
        seed=seed,
        output_path=str(output_refine_path),
        save_video=True,
        return_frames=True,
        fps=30,
    )
    
    refine_time = time.time() - refine_start
    print(f"✓ Stage 2 completed in {refine_time:.2f}s → {output_refine_path}")
    print()
    
    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"✓ Total pipeline completed in {total_time:.2f}s")
    print(f"  Stage 1 (distilled): {distill_time:.2f}s")
    print(f"  Stage 2 (refinement): {refine_time:.2f}s")
    print("=" * 80)
    print()


def test_refinement(model_path: str, num_gpus: int = 1, input_video_path: str = None):
    """Test refinement LoRA (720p) from existing video."""
    
    print("=" * 80)
    print("Refinement Generation (720p with LoRA)")
    print("=" * 80)
    print()
    
    if input_video_path is None:
        print("⚠️  No input video provided. Use --input-video to specify a 480p video for refinement.")
        print("   Or use --mode distilled_refine to run the full pipeline.")
        print()
        return
    
    print(f"Model: {model_path}")
    print(f"LoRA: {model_path}/lora/refinement")
    print(f"Input: {input_video_path}")
    print()
    
    # TODO: Implement refinement from existing video file
    print("⚠️  Not yet implemented: Refinement from existing video file")
    print("   Use --mode distilled_refine to run the full pipeline instead.")
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
        choices=["standard", "standard-16", "distilled", "distilled_refine", "refinement"],
        default="distilled_refine",
        help="Test mode: standard (50 steps), standard-16 (16 steps for comparison), distilled (16 steps with LoRA), distilled_refine (480p->720p pipeline), refinement (720p with LoRA)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--input-video",
        type=str,
        default=None,
        help="Input video path for refinement mode",
    )
    
    args = parser.parse_args()
    
    if args.mode == "standard":
        test_standard(args.model_path, args.num_gpus, num_steps=50)
    elif args.mode == "standard-16":
        test_standard(args.model_path, args.num_gpus, num_steps=16)
    elif args.mode == "distilled":
        test_distilled(args.model_path, args.num_gpus)
    elif args.mode == "distilled_refine":
        test_distilled_refine(args.model_path, args.num_gpus)
    elif args.mode == "refinement":
        test_refinement(args.model_path, args.num_gpus, args.input_video)


if __name__ == "__main__":
    main()
