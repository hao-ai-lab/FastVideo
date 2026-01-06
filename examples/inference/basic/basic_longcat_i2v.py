"""
LongCat Image-to-Video (I2V) Example Script

This script demonstrates LongCat I2V inference using the FastVideo Python API.
LongCat I2V takes an input image and generates a video from it.

It runs both basic generation (50 steps) and distill+refine generation
(16 steps distill + 50 steps refinement to 720p with BSA).

Usage:
    python examples/inference/basic/basic_longcat_i2v.py

Note:
    Refinement uses 768x768 dimensions where latent (48x48) is divisible by 8,
    compatible with BSA chunks [4, 4, 8].
"""

import glob
import os

from fastvideo import VideoGenerator

# Common prompts and settings matching the shell script examples
PROMPT = (
    "A woman sits at a wooden table by the window in a cozy café. She reaches out "
    "with her right hand, picks up the white coffee cup from the saucer, and gently "
    "brings it to her lips to take a sip. After drinking, she places the cup back on "
    "the table and looks out the window, enjoying the peaceful atmosphere."
)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# Input image path
IMAGE_PATH = "assets/girl.png"

SEED = 42


def basic_generation():
    """
    Run basic LongCat I2V generation (50 steps at 480p).
    
    This uses the full 50-step denoising process for highest quality.
    """
    print("=" * 60)
    print("LongCat I2V: Basic Generation (50 steps, 480p)")
    print("=" * 60)
    
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LongCat-Video-I2V-Diffusers",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        enable_bsa=False,
    )
    
    output_path = "outputs_video/longcat_i2v_basic"
    
    generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image_path=IMAGE_PATH,
        output_path=output_path,
        save_video=True,
        height=480,
        width=480,  # Square
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
    Run LongCat I2V with distill+refine pipeline (16 steps + refinement to 768p).
    
    This uses the distilled LoRA for fast 480p generation (16 steps),
    then refines to 768p using the refinement LoRA with BSA enabled.
    """
    print("\n" + "=" * 60)
    print("LongCat I2V: Distill + Refine Pipeline")
    print("=" * 60)
    
    # Stage 1: Distilled generation (16 steps at 480p)
    print("\n[Stage 1] Distilled generation (16 steps, 480p)")
    print("-" * 40)
    
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LongCat-Video-I2V-Diffusers",
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
    
    distill_output_path = "outputs_video/longcat_i2v_distill"
    
    generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image_path=IMAGE_PATH,
        output_path=distill_output_path,
        save_video=True,
        height=480,
        width=480,  # Square
        num_frames=93,
        num_inference_steps=16,
        fps=15,
        guidance_scale=1.0,
        seed=SEED,
    )
    
    print(f"Distilled generation complete! Video saved to: {distill_output_path}")
    generator.shutdown()
    
    # Stage 2: Refinement (480p -> 768p)
    print("\n[Stage 2] Refinement (480p -> 768p with BSA)")
    print("-" * 40)
    
    # Find the actual saved video file from stage 1
    video_files = glob.glob(os.path.join(distill_output_path, "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {distill_output_path}")
    # Use the most recently created video file
    distill_video_path = max(video_files, key=os.path.getmtime)
    print(f"Using stage 1 video: {distill_video_path}")
    
    # Create a new generator with refinement LoRA and BSA enabled
    # Note: Refinement uses the T2V model (not I2V) since it's upscaling the generated video
    # For BSA [4, 4, 8]: latent must be divisible by 8
    # 768x768: latent 48x48, 48%8=0 ✓
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
        bsa_chunk_q=[4, 4, 4],
        bsa_chunk_k=[4, 4, 4],
        lora_path="FastVideo/LongCat-Video-T2V-Refinement-LoRA",
        lora_nickname="refinement",
    )
    
    refine_output_path = "outputs_video/longcat_i2v_refine_720p"
    
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
        width=720,
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
    print("LongCat Image-to-Video Example")
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


