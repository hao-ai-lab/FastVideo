from fastvideo import VideoGenerator
from fastvideo.models.dits.hyworld.resolution_utils import get_resolution_from_image

# Default prompt from HY-WorldPlay run.sh
DEFAULT_PROMPT = 'A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
DEFAULT_IMAGE = 'https://raw.githubusercontent.com/Tencent-Hunyuan/HY-WorldPlay/main/assets/img/test.png'

OUTPUT_PATH = "video_samples_hyworld"
def main():
    import argparse

    # pose: (a, w, s, d) - (15, 31)
    # num_frames: (61, 125)
    parser = argparse.ArgumentParser(description="HYWorld video generation with FastVideo")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Path or URL to input image")
    parser.add_argument("--pose", type=str, default='d-31', help="Pose string (e.g., 'a-31', 'w-31', 's-31', 'd-31')")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Output video path")
    parser.add_argument("--num-frames", type=int, default=125, help="Number of frames")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--resolution", type=str, default="480p", help="Only support 480p for now")
    parser.add_argument("--mode", type=str, default="ar", choices=["bi", "ar"], help="Model mode: 'bi' (bidirectional, chunk=16) or 'ar' (autoregressive, chunk=4)")
    args = parser.parse_args()

    # Set chunk_latent_frames based on mode
    chunk_latent_frames = 16 if args.mode == "bi" else 4

    # Automatically determine resolution from input image
    HEIGHT, WIDTH = get_resolution_from_image(args.image, args.resolution)
    print(f"Image: {args.image}")
    print(f"Pose: {args.pose}")
    print(f"Mode: {args.mode} (chunk_latent_frames={chunk_latent_frames})")
    print(f"Resolution: {HEIGHT}x{WIDTH} (from {args.resolution} buckets)")
    print(f"Num frames: {args.num_frames}")
    print(f"Output path: {args.output_path}")

    if args.mode == "bi":
        model_path = "FastVideo/HY-WorldPlay-Bidirectional-Diffusers"
    elif args.mode == "ar":
        model_path = "FastVideo/HY-WorldPlay-AR-Diffusers"
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Initialize generator
    print("\nInitializing VideoGenerator for HYWorld...")
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
        image_encoder_cpu_offload=False,
    )

    # Generate video
    # The pose string is automatically converted to camera matrices by the pipeline
    print("\nGenerating video...")
    generator.generate_video(
        prompt=args.prompt,
        image_path=args.image,
        pose=args.pose,  # Camera trajectory control
        output_path=args.output_path,
        save_video=True,
        negative_prompt="",
        num_frames=args.num_frames,
        fps=24,
        height=HEIGHT,
        width=WIDTH,
        seed=args.seed,
        chunk_latent_frames=chunk_latent_frames,
    )

    print(f"\nVideo saved to: {args.output_path}")


if __name__ == "__main__":
    main()
