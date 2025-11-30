"""
LongCat I2V Demo Script

Supports three modes:
1. Basic I2V (480p): 50 steps, guidance_scale=4.0
2. Distilled I2V (480p): 16 steps with cfg_step_lora, guidance_scale=1.0
3. Full pipeline: Distilled 480p + Refined 720p upscaling

Usage:
    # Basic I2V (50 steps)
    python run_demo_longcat_i2v.py --model_path weights/longcat-for-i2v --mode basic
    
    # Distilled I2V (16 steps, faster)
    python run_demo_longcat_i2v.py --model_path weights/longcat-for-i2v --mode distilled
    
    # Full pipeline: distilled 480p -> refined 720p
    python run_demo_longcat_i2v.py --model_path weights/longcat-for-i2v --mode full
"""

import argparse
import gc
import os
import time

import numpy as np
import torch
from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.longcat import get_bucket_config

# Test data
IMAGE_PATH = "assets/girl_original.png"  # Original 2048x2048 square image
PROMPT = "A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip."
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def get_closest_bucket_dimensions(image_height,
                                  image_width,
                                  resolution="480p",
                                  scale_factor_spatial=32):
    """Find the closest bucket dimensions for a given input image aspect ratio."""
    bucket_config = get_bucket_config(resolution, scale_factor_spatial)
    ratio = image_height / image_width
    closest_bucket = sorted(bucket_config.keys(),
                            key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = bucket_config[closest_bucket][0]
    return target_h, target_w


def generate_basic_i2v(generator,
                       image_path,
                       target_height,
                       target_width,
                       seed=42):
    """Generate basic I2V video (50 steps, guidance_scale=4.0)."""
    print("\n" + "=" * 60)
    print("Basic I2V Generation (480p, 50 steps)")
    print("=" * 60)

    start_time = time.time()

    video = generator.generate_video(
        prompt=PROMPT,
        image_path=image_path,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=93,
        height=target_height,
        width=target_width,
        num_inference_steps=50,
        guidance_scale=4.0,
        seed=seed,
        output_path="output_longcat_i2v",
        save_video=True,
        return_frames=True,
    )

    gen_time = time.time() - start_time
    print(
        f"✓ Basic I2V complete in {gen_time:.1f}s! Saved to output_longcat_i2v/"
    )
    return video


def generate_distilled_i2v(generator,
                           image_path,
                           target_height,
                           target_width,
                           model_path,
                           seed=42):
    """Generate distilled I2V video (16 steps with LoRA, guidance_scale=1.0)."""
    print("\n" + "=" * 60)
    print("Distilled I2V Generation (480p, 16 steps)")
    print("=" * 60)

    # Load distilled LoRA
    lora_path = os.path.join(model_path, "lora/distilled")
    print(f"Loading distilled LoRA from: {lora_path}")
    generator.set_lora_adapter(lora_nickname="distilled", lora_path=lora_path)

    start_time = time.time()

    video = generator.generate_video(
        prompt=PROMPT,
        image_path=image_path,
        negative_prompt=None,  # Not used in distilled mode
        num_frames=93,
        height=target_height,
        width=target_width,
        num_inference_steps=16,
        guidance_scale=1.0,
        seed=seed,
        output_path="output_longcat_i2v_distilled",
        save_video=True,
        return_frames=True,
    )

    gen_time = time.time() - start_time
    print(
        f"✓ Distilled I2V complete in {gen_time:.1f}s! Saved to output_longcat_i2v_distilled/"
    )
    return video


def generate_refined_i2v(stage1_frames, image_path, model_path, seed=42):
    """
    Refine 480p video to 720p using refinement LoRA + BSA.
    
    The I2V pipeline supports refinement via LongCatRefineInitStage.
    For refinement, we pass stage1_video and the pipeline:
    1. Skips image encoding (we already have the video)
    2. Encodes stage1_video, mixes with noise
    3. Denoises to produce refined output
    """
    print("\n" + "=" * 60)
    print("Refinement I2V Generation (720p upscaling)")
    print("=" * 60)

    # Convert frames to PIL images if needed
    if isinstance(stage1_frames, list) and isinstance(stage1_frames[0],
                                                      np.ndarray):
        stage1_video = [Image.fromarray(frame) for frame in stage1_frames]
    else:
        stage1_video = stage1_frames

    print(f"Stage 1 video: {len(stage1_video)} frames")

    # Get 720p dimensions based on input aspect ratio
    orig_image = Image.open(image_path)
    # For refinement, use scale_factor_spatial=64 (VAE*patch*BSA granularity = 8*2*4)
    target_height, target_width = get_closest_bucket_dimensions(
        orig_image.height,
        orig_image.width,
        resolution="720p",
        scale_factor_spatial=64)
    print(f"Target 720p resolution: {target_width}x{target_height}")

    # Initialize generator with BSA enabled for refinement
    print("\nInitializing refinement generator with BSA enabled...")

    refine_generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config={
            "enable_bsa": True,
            "vae_tiling": True
        },  # Enable BSA + VAE tiling for refinement
    )

    # Load refinement LoRA
    lora_path = os.path.join(model_path, "lora/refinement")
    print(f"Loading refinement LoRA from: {lora_path}")
    refine_generator.set_lora_adapter(lora_nickname="refinement",
                                      lora_path=lora_path)

    start_time = time.time()

    # For I2V refinement: pass stage1_video to trigger refinement mode
    # The stage1_video already has the image-conditioned first frame from I2V distill,
    # so we don't need explicit image conditioning (num_cond_frames=0)
    video = refine_generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        stage1_video=
        stage1_video,  # 480p video to refine (already has I2V first frame)
        num_cond_frames=0,  # 0 = no explicit image re-conditioning needed
        t_thresh=0.5,  # Refinement noise threshold
        spatial_refine_only=True,  # Spatial only to fit in 1 GPU
        num_frames=93,  # Keep same frame count
        height=target_height,
        width=target_width,
        num_inference_steps=50,
        guidance_scale=4.0,
        seed=seed,
        output_path="output_longcat_i2v_refined",
        save_video=True,
        return_frames=True,
    )

    gen_time = time.time() - start_time
    print(
        f"✓ Refinement complete in {gen_time:.1f}s! Saved to output_longcat_i2v_refined/"
    )

    # Cleanup refinement generator
    refine_generator.shutdown()
    del refine_generator
    torch.cuda.empty_cache()
    gc.collect()

    return video


def main():
    parser = argparse.ArgumentParser(description="LongCat I2V Demo")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to LongCat weights")
    parser.add_argument("--image_path",
                        type=str,
                        default=IMAGE_PATH,
                        help="Path to input image")
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "distilled", "full"],
        help=
        "Generation mode: basic (50 steps), distilled (16 steps), full (distilled + refine)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check if image exists, try alternatives
    if not os.path.exists(args.image_path):
        alternatives = ["assets/girl_square_480p.png", "assets/girl_480p.png"]
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"Image {args.image_path} not found, using {alt}")
                args.image_path = alt
                break
        else:
            print(
                f"ERROR: No valid image found. Tried: {args.image_path}, {alternatives}"
            )
            return

    # Load image and get proper dimensions based on aspect ratio
    image = Image.open(args.image_path)
    img_width, img_height = image.size
    target_height, target_width = get_closest_bucket_dimensions(
        img_height, img_width, resolution="480p")

    print("=" * 60)
    print(f"LongCat I2V Demo - Mode: {args.mode.upper()}")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"  Input size: {img_width}x{img_height} (W x H)")
    print(f"  Aspect ratio (H/W): {img_height/img_width:.2f}")
    print(f"  Target 480p size: {target_width}x{target_height} (W x H)")

    # Initialize generator (for basic and distilled modes)
    if args.mode in ["basic", "distilled", "full"]:
        print("\nInitializing VideoGenerator...")

        # Import the I2V pipeline to ensure it's registered

        generator = VideoGenerator.from_pretrained(
            args.model_path,
            num_gpus=1,
            dit_cpu_offload=False,
            vae_cpu_offload=True,
            text_encoder_cpu_offload=True,
            use_fsdp_inference=False,
        )

    # Run based on mode
    if args.mode == "basic":
        generate_basic_i2v(generator, args.image_path, target_height,
                           target_width, args.seed)

    elif args.mode == "distilled":
        generate_distilled_i2v(generator, args.image_path, target_height,
                               target_width, args.model_path, args.seed)

    elif args.mode == "full":
        # Stage 1: Distilled 480p
        stage1_frames = generate_distilled_i2v(generator, args.image_path,
                                               target_height, target_width,
                                               args.model_path, args.seed)

        # Shutdown first generator to free memory
        print("\nShutting down stage 1 generator...")
        generator.shutdown()
        del generator
        torch.cuda.empty_cache()
        gc.collect()

        # Stage 2: Refine to 720p (creates its own generator with BSA)
        generate_refined_i2v(stage1_frames, args.image_path, args.model_path,
                             args.seed)

    print("\n" + "=" * 60)
    print("✓ All generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
