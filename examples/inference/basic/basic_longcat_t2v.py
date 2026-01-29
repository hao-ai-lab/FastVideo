"""
LongCat Text-to-Video (T2V) Example Script

This script demonstrates LongCat T2V inference using the FastVideo Python API.
It runs both basic generation (50 steps) and distill+refine generation
(16 steps distill + 50 steps refinement to 720p).

Usage:
    python examples/inference/basic/basic_longcat_t2v.py
"""

import glob
import os

from fastvideo import VideoGenerator

# Common prompts and settings matching the shell script examples
PROMPT = (
    "In a realistic photography style, a white boy around seven or eight years old "
    "sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. "
    "He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a "
    "medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, "
    "who eagerly licks it with its tongue. The sun is shining brightly, and the background "
    "features a green lawn and several tall trees, creating a warm and loving scene."
)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

SEED = 42


def basic_generation():
    """
    Run basic LongCat T2V generation (50 steps at 480p).
    
    This uses the full 50-step denoising process for highest quality.
    """
    print("=" * 60)
    print("LongCat T2V: Basic Generation (50 steps, 480p)")
    print("=" * 60)
    
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LongCat-Video-T2V-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        enable_bsa=False,
    )
    
    output_path = "outputs_video/longcat_t2v_basic"
    
    generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        output_path=output_path,
        save_video=True,
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=50,
        fps=15,
        guidance_scale=4.0,
        seed=SEED,
    )
    
    print(f"\nBasic generation complete! Video saved to: {output_path}")
    generator.shutdown()


def distill_refine_generation():
    """
    Run LongCat T2V with distill+refine pipeline (16 steps + refinement to 720p).
    
    This uses the distilled LoRA for fast 480p generation (16 steps),
    then refines to 720p using the refinement LoRA with BSA enabled.
    """
    print("\n" + "=" * 60)
    print("LongCat T2V: Distill + Refine Pipeline")
    print("=" * 60)
    
    # Stage 1: Distilled generation (16 steps at 480p)
    print("\n[Stage 1] Distilled generation (16 steps, 480p)")
    print("-" * 40)
    
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LongCat-Video-T2V-Diffusers",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        enable_bsa=False,
        lora_path="FastVideo/LongCat-Video-T2V-Distilled-LoRA",
        lora_nickname="distilled",
    )
    
    distill_output_path = "outputs_video/longcat_t2v_distill"
    
    generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        output_path=distill_output_path,
        save_video=True,
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=16,
        fps=15,
        guidance_scale=1.0,
        seed=SEED,
    )
    
    print(f"Distilled generation complete! Video saved to: {distill_output_path}")
    generator.shutdown()
    
    # Stage 2: Refinement (480p -> 720p)
    print("\n[Stage 2] Refinement (480p -> 720p with BSA)")
    print("-" * 40)
    
    # Find the actual saved video file from stage 1
    video_files = glob.glob(os.path.join(distill_output_path, "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {distill_output_path}")
    # Use the most recently created video file
    distill_video_path = max(video_files, key=os.path.getmtime)
    print(f"Using stage 1 video: {distill_video_path}")
    
    # Create a new generator with refinement LoRA and BSA enabled
    refine_generator = VideoGenerator.from_pretrained(
        "FastVideo/LongCat-Video-T2V-Diffusers",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        enable_bsa=True,
        bsa_sparsity=0.875,
        bsa_chunk_q=[4, 4, 8],
        bsa_chunk_k=[4, 4, 8],
        lora_path="FastVideo/LongCat-Video-T2V-Refinement-LoRA",
        lora_nickname="refinement",
    )
    
    refine_output_path = "outputs_video/longcat_t2v_refine_720p"
    
    refine_generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        output_path=refine_output_path,
        save_video=True,
        refine_from=distill_video_path,
        t_thresh=0.5,
        spatial_refine_only=False,
        num_cond_frames=0,
        height=720,
        width=1280,
        num_inference_steps=50,
        fps=30,
        guidance_scale=1.0,
        seed=SEED,
    )
    
    print(f"Refinement complete! Video saved to: {refine_output_path}")
    refine_generator.shutdown()


def main():
    """Run both basic and distill+refine generation pipelines."""
    print("\n" + "=" * 60)
    print("LongCat Text-to-Video Example")
    print("=" * 60 + "\n")
    
    # Run basic generation
    basic_generation()
    
    # Run distill+refine pipeline
    distill_refine_generation()
    
    print("\n" + "=" * 60)
    print("All generations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


