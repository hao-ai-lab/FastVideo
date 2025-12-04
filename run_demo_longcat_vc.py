"""
LongCat Video Continuation (VC) Demo Script

Supports three modes:
1. Basic VC (480p): 50 steps, guidance_scale=4.0, with KV cache
2. Distilled VC (480p): 16 steps with cfg_step_lora, guidance_scale=1.0
3. Full pipeline: Distilled 480p + Refined 720p upscaling

Usage:
    # Basic VC with KV cache (50 steps)
    python run_demo_longcat_vc.py --model_path weights/longcat-for-i2v --mode basic
    
    # Basic VC without KV cache (for comparison/debugging)
    python run_demo_longcat_vc.py --model_path weights/longcat-for-i2v --mode basic --no-kv-cache
    
    # Distilled VC (16 steps, faster)
    python run_demo_longcat_vc.py --model_path weights/longcat-for-i2v --mode distilled
"""

import argparse
import json
import os
import shutil
import time


from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.longcat import get_bucket_config

# Test data
VIDEO_PATH = "assets/motorcycle.mp4"  # Example video from LongCat


def create_vc_model_config(source_model_path: str) -> str:
    """
    Create a VC model config from an existing I2V/T2V model path.
    
    This creates a symlinked directory with a modified model_index.json
    that specifies LongCatVideoContinuationPipeline.
    
    Returns the path to the VC model config.
    """
    # Use absolute paths for symlinks
    source_model_path = os.path.abspath(source_model_path)
    vc_model_path = source_model_path.rstrip('/') + "-vc"

    # Check if already exists and is valid
    vc_index_path = os.path.join(vc_model_path, "model_index.json")
    if os.path.exists(vc_index_path):
        with open(vc_index_path) as f:
            config = json.load(f)
        if config.get("_class_name") == "LongCatVideoContinuationPipeline":
            print(f"Using existing VC config at {vc_model_path}")
            return vc_model_path

    print(f"Creating VC model config at {vc_model_path}")

    # Create directory if needed
    os.makedirs(vc_model_path, exist_ok=True)

    # Symlink all items from source (directories and files)
    for item in os.listdir(source_model_path):
        source_item = os.path.join(source_model_path, item)
        target_item = os.path.join(vc_model_path, item)

        if item == "model_index.json":
            continue  # We'll create our own

        # Remove existing target if present
        if os.path.exists(target_item) or os.path.islink(target_item):
            if os.path.islink(target_item):
                os.unlink(target_item)
            elif os.path.isdir(target_item):
                shutil.rmtree(target_item)
            else:
                os.remove(target_item)

        # Create symlink (works for both files and directories)
        os.symlink(source_item, target_item)

    # Create modified model_index.json
    source_index = os.path.join(source_model_path, "model_index.json")
    with open(source_index) as f:
        config = json.load(f)

    # Change pipeline class to VC
    config["_class_name"] = "LongCatVideoContinuationPipeline"

    with open(vc_index_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created VC model config with pipeline: {config['_class_name']}")
    return vc_model_path


PROMPT = "A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides. The video captures the journey from the rider's perspective, emphasizing the sense of motion and adventure."
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def get_video_fps(video_path):
    """Get FPS of input video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_closest_bucket_dimensions(height,
                                  width,
                                  resolution="480p",
                                  scale_factor_spatial=32):
    """Find the closest bucket dimensions for a given aspect ratio."""
    bucket_config = get_bucket_config(resolution, scale_factor_spatial)
    ratio = height / width
    closest_bucket = sorted(bucket_config.keys(),
                            key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = bucket_config[closest_bucket][0]
    return target_h, target_w


def load_video_frames(video_path, target_fps=15, max_frames=None):
    """Load video frames and resample to target FPS."""
    from diffusers.utils import load_video

    # Load all frames
    frames = load_video(video_path)
    print(f"Loaded {len(frames)} frames from {video_path}")

    # Get original FPS and calculate stride
    original_fps = get_video_fps(video_path)
    stride = max(1, round(original_fps / target_fps))

    # Subsample frames
    frames = frames[::stride]
    print(
        f"Subsampled to {len(frames)} frames (stride={stride}, original_fps={original_fps:.1f}, target_fps={target_fps})"
    )

    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]
        print(f"Truncated to {max_frames} frames")

    return frames


def generate_basic_vc(generator,
                      video_path,
                      target_height,
                      target_width,
                      num_cond_frames=13,
                      use_kv_cache=True,
                      seed=42):
    """Generate basic VC video (50 steps, guidance_scale=4.0)."""
    print("\n" + "=" * 60)
    cache_str = "with KV cache" if use_kv_cache else "without KV cache"
    print(f"Basic VC Generation (480p, 50 steps, {cache_str})")
    print("=" * 60)

    # Load video frames just for info display
    video_frames = load_video_frames(video_path, target_fps=15)
    first_frame = video_frames[0]
    print(f"First frame size: {first_frame.size}")
    print(f"Target resolution: {target_width}x{target_height}")

    start_time = time.time()

    # Use video_path instead of video_frames - the stage will load it
    video = generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        video_path=video_path,  # Pass video path - stage will load frames
        num_cond_frames=num_cond_frames,
        num_frames=93,
        height=target_height,
        width=target_width,
        num_inference_steps=50,
        guidance_scale=4.0,
        seed=seed,
        output_path="output_longcat_vc",
        save_video=True,
        return_frames=True,
    )

    gen_time = time.time() - start_time
    print(
        f"✓ Basic VC complete in {gen_time:.1f}s! Saved to output_longcat_vc/")
    return video


def generate_distilled_vc(generator,
                          video_path,
                          target_height,
                          target_width,
                          model_path,
                          num_cond_frames=13,
                          use_kv_cache=True,
                          seed=42):
    """Generate distilled VC video (16 steps with LoRA, guidance_scale=1.0)."""
    print("\n" + "=" * 60)
    print("Distilled VC Generation (480p, 16 steps)")
    print("=" * 60)

    # Load distilled LoRA
    lora_path = os.path.join(model_path, "lora/distilled")
    if not os.path.exists(lora_path):
        # Try alternative path
        lora_path = os.path.join(model_path, "lora/cfg_step_lora.safetensors")
    print(f"Loading distilled LoRA from: {lora_path}")
    generator.set_lora_adapter(lora_nickname="distilled", lora_path=lora_path)

    start_time = time.time()

    video = generator.generate_video(
        prompt=PROMPT,
        negative_prompt=None,  # Not used in distilled mode
        video_path=video_path,  # Pass video path - stage will load frames
        num_cond_frames=num_cond_frames,
        num_frames=93,
        height=target_height,
        width=target_width,
        num_inference_steps=16,
        guidance_scale=1.0,
        seed=seed,
        output_path="output_longcat_vc_distilled",
        save_video=True,
        return_frames=True,
    )

    gen_time = time.time() - start_time
    print(
        f"✓ Distilled VC complete in {gen_time:.1f}s! Saved to output_longcat_vc_distilled/"
    )
    return video


def main():
    parser = argparse.ArgumentParser(description="LongCat VC Demo")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to LongCat weights")
    parser.add_argument("--video_path",
                        type=str,
                        default=VIDEO_PATH,
                        help="Path to input video")
    parser.add_argument("--num_cond_frames",
                        type=int,
                        default=13,
                        help="Number of conditioning frames")
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "distilled"],
        help="Generation mode: basic (50 steps), distilled (16 steps)")
    parser.add_argument("--no-kv-cache",
                        action="store_true",
                        help="Disable KV cache (for debugging/comparison)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    use_kv_cache = not args.no_kv_cache

    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video not found at {args.video_path}")
        print("Please provide a valid video path with --video_path")
        return

    # Load first frame to get dimensions
    video_frames = load_video_frames(args.video_path,
                                     target_fps=15,
                                     max_frames=1)
    first_frame = video_frames[0]
    img_width, img_height = first_frame.size
    target_height, target_width = get_closest_bucket_dimensions(
        img_height, img_width, resolution="480p")

    print("=" * 60)
    print(f"LongCat VC Demo - Mode: {args.mode.upper()}")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Video: {args.video_path}")
    print(f"  First frame size: {img_width}x{img_height} (W x H)")
    print(f"  Aspect ratio (H/W): {img_height/img_width:.2f}")
    print(f"  Target 480p size: {target_width}x{target_height} (W x H)")
    print(f"  Num cond frames: {args.num_cond_frames}")
    print(f"  KV cache: {use_kv_cache}")

    # Create VC model config (symlinks weights, changes model_index.json)
    vc_model_path = create_vc_model_config(args.model_path)

    # Initialize generator
    print("\nInitializing VideoGenerator...")

    generator = VideoGenerator.from_pretrained(
        vc_model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config={
            "use_kv_cache": use_kv_cache,
            "offload_kv_cache": False,
        },
    )

    # Run based on mode
    if args.mode == "basic":
        generate_basic_vc(generator, args.video_path, target_height,
                          target_width, args.num_cond_frames, use_kv_cache,
                          args.seed)

    elif args.mode == "distilled":
        generate_distilled_vc(generator, args.video_path, target_height,
                              target_width, args.model_path,
                              args.num_cond_frames, use_kv_cache, args.seed)

    print("\n" + "=" * 60)
    print("✓ All generation complete!")
    print("=" * 60)

    # Cleanup
    generator.shutdown()


if __name__ == "__main__":
    main()
