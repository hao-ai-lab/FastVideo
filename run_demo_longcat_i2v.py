"""
LongCat I2V Test Script

Minimal test for Tier 3 I2V: timestep masking + num_cond_latents + RoPE skipping
"""

import argparse
from PIL import Image
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.longcat import get_bucket_config

# Test data
IMAGE_PATH = "assets/girl_480p.png"
PROMPT = "A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere."
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to LongCat weights")
    parser.add_argument("--image_path",
                        type=str,
                        default=IMAGE_PATH,
                        help="Path to input image")
    parser.add_argument("--resolution",
                        type=str,
                        default="480p",
                        choices=["480p", "720p"],
                        help="Target resolution")
    args = parser.parse_args()

    # Load image and get proper dimensions based on aspect ratio
    image = Image.open(args.image_path)
    img_width, img_height = image.size
    target_height, target_width = get_closest_bucket_dimensions(
        img_height, img_width, resolution=args.resolution)

    print("=" * 60)
    print("LongCat I2V Test (Tier 3)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"  Input size: {img_width}x{img_height} (W x H)")
    print(f"  Aspect ratio: {img_height/img_width:.2f}")
    print(f"  Target size: {target_width}x{target_height} (W x H)")
    print()

    # Initialize generator
    print("Initializing VideoGenerator...")

    # Import the I2V pipeline to ensure it's registered

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=1,  # Single GPU, no distributed
        dit_cpu_offload=False,  # Keep on GPU for speed
        vae_cpu_offload=True,  # Offload VAE to save memory
        text_encoder_cpu_offload=True,
        use_fsdp_inference=False,  # Disable FSDP for simple test
    )

    # Generate video
    print("\nGenerating I2V video...")
    print(f"Prompt: {PROMPT[:80]}...")

    _ = generator.generate_video(
        prompt=PROMPT,
        image_path=args.image_path,  # I2V mode triggered by image_path
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=93,
        height=target_height,
        width=target_width,
        num_inference_steps=50,
        guidance_scale=4.0,
        seed=42,
        fps=15,  # Match LongCat-Video fps
        output_path="output_longcat_i2v",
        save_video=True,
    )

    print("\n" + "=" * 60)
    print("✓ I2V generation complete!")
    print(f"Output dimensions: {target_width}x{target_height}")
    print("Saved to: output_longcat_i2v/")
    print("=" * 60)


if __name__ == "__main__":
    main()
