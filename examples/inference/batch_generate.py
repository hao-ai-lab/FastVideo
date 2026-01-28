#!/usr/bin/env python3
"""
Batch video generation script for FastVideo.

Takes a text file with one prompt per line and generates videos for each prompt.

Usage:
    python batch_generate.py --prompts prompts.txt --output_dir outputs/
    python batch_generate.py --prompts prompts.txt --model Wan-AI/Wan2.1-T2V-14B-Diffusers --num_gpus 2
    python batch_generate.py --prompts prompts.txt --attention_backend SAGE_ATTN_THREE
"""

import argparse
import os
import re
import time
from pathlib import Path

# Note: VideoGenerator import is deferred to main() so we can set
# FASTVIDEO_ATTENTION_BACKEND environment variable first


def sanitize_filename(prompt: str, max_length: int = 50) -> str:
    """
    Create a safe filename from a prompt.
    
    Args:
        prompt: The prompt text
        max_length: Maximum length of the filename (excluding extension)
    
    Returns:
        A sanitized filename string
    """
    # Take first part of prompt and clean it
    sanitized = re.sub(r'[^\w\s-]', '', prompt.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    sanitized = sanitized.strip('_')
    
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit('_', 1)[0]
    
    return sanitized or "video"


def load_prompts(prompt_file: str) -> list[str]:
    """
    Load prompts from a text file.
    
    Args:
        prompt_file: Path to the text file with prompts (one per line)
    
    Returns:
        List of prompts
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Batch video generation with FastVideo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model (Wan2.1-T2V-1.3B)
  python batch_generate.py --prompts prompts.txt

  # Use 14B model with 2 GPUs
  python batch_generate.py --prompts prompts.txt --model Wan-AI/Wan2.1-T2V-14B-Diffusers --num_gpus 2

  # Custom resolution and frame count
  python batch_generate.py --prompts prompts.txt --height 720 --width 1280 --num_frames 125

  # Use CPU offloading for low VRAM
  python batch_generate.py --prompts prompts.txt --dit_cpu_offload --vae_cpu_offload

  # Use custom fine-tuned weights
  python batch_generate.py --prompts prompts.txt --init_weights_from_safetensors checkpoints/my_model/transformer

  # Use SageAttention3 backend (RTX 5090 only)
  python batch_generate.py --prompts prompts.txt --attention_backend SAGE_ATTN_THREE

  # Use Flash Attention backend
  python batch_generate.py --prompts prompts.txt --attention_backend FLASH_ATTN
        """
    )
    
    # Attention backend argument (must be parsed early)
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        choices=[
            "TORCH_SDPA",
            "FLASH_ATTN",
            "SLIDING_TILE_ATTN",
            "VIDEO_SPARSE_ATTN",
            "SAGE_ATTN",
            "SAGE_ATTN_THREE",
        ],
        help="Attention backend to use (default: auto-detect). SAGE_ATTN_THREE requires RTX 5090."
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--prompts", "-p",
        type=str,
        required=True,
        help="Path to text file with prompts (one per line)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="batch_outputs",
        help="Directory to save generated videos (default: batch_outputs)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Model path or HuggingFace model ID (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    
    # Video parameters
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video height (default: model default)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video width (default: model default)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to generate (default: model default)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second (default: model default)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of inference steps (default: model default)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale (default: model default)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None, uses different seed per video)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt (default: model default)"
    )
    
    # Memory optimization arguments
    parser.add_argument(
        "--use_fsdp_inference",
        action="store_true",
        help="Use FSDP for inference (set if GPU OOM)"
    )
    parser.add_argument(
        "--dit_cpu_offload",
        action="store_true",
        help="Offload DiT to CPU when not in use"
    )
    parser.add_argument(
        "--vae_cpu_offload",
        action="store_true",
        help="Offload VAE to CPU when not in use"
    )
    parser.add_argument(
        "--text_encoder_cpu_offload",
        action="store_true",
        default=True,
        help="Offload text encoder to CPU when not in use (default: True)"
    )
    parser.add_argument(
        "--no_text_encoder_cpu_offload",
        action="store_true",
        help="Disable text encoder CPU offloading"
    )
    parser.add_argument(
        "--pin_cpu_memory",
        action="store_true",
        default=True,
        help="Pin CPU memory for faster transfers (default: True)"
    )
    parser.add_argument(
        "--no_pin_cpu_memory",
        action="store_true",
        help="Disable CPU memory pinning"
    )
    parser.add_argument(
        "--init_weights_from_safetensors",
        type=str,
        default=None,
        help="Path to custom safetensors weights to load (e.g., checkpoint/transformer directory)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start processing from this prompt index (0-based)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Stop processing at this prompt index (exclusive)"
    )
    parser.add_argument(
        "--filename_prefix",
        type=str,
        default="",
        help="Prefix for output filenames"
    )
    parser.add_argument(
        "--use_index_only",
        action="store_true",
        help="Use only index numbers for filenames instead of prompt text"
    )
    
    args = parser.parse_args()
    
    # Set attention backend BEFORE importing VideoGenerator
    if args.attention_backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
        print(f"Using attention backend: {args.attention_backend}")
    
    # Now import VideoGenerator (after setting env var)
    from fastvideo import VideoGenerator
    
    # Validate inputs
    if not os.path.exists(args.prompts):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts}")
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    
    # Apply start/end indices
    end_idx = args.end_idx if args.end_idx is not None else len(prompts)
    prompts_to_process = prompts[args.start_idx:end_idx]
    print(f"Processing prompts {args.start_idx} to {end_idx - 1} ({len(prompts_to_process)} videos)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle boolean flag inversions
    text_encoder_cpu_offload = not args.no_text_encoder_cpu_offload
    pin_cpu_memory = not args.no_pin_cpu_memory
    
    # Initialize generator
    print(f"\nInitializing VideoGenerator with model: {args.model}")
    if args.init_weights_from_safetensors:
        print(f"Loading custom weights from: {args.init_weights_from_safetensors}")
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        use_fsdp_inference=args.use_fsdp_inference,
        dit_cpu_offload=args.dit_cpu_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        text_encoder_cpu_offload=text_encoder_cpu_offload,
        pin_cpu_memory=pin_cpu_memory,
        init_weights_from_safetensors=args.init_weights_from_safetensors,
    )
    print("Generator initialized successfully!\n")
    
    # Build optional kwargs for generate_video
    generate_kwargs = {}
    if args.height is not None:
        generate_kwargs["height"] = args.height
    if args.width is not None:
        generate_kwargs["width"] = args.width
    if args.num_frames is not None:
        generate_kwargs["num_frames"] = args.num_frames
    if args.fps is not None:
        generate_kwargs["fps"] = args.fps
    if args.num_inference_steps is not None:
        generate_kwargs["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        generate_kwargs["guidance_scale"] = args.guidance_scale
    if args.negative_prompt is not None:
        generate_kwargs["negative_prompt"] = args.negative_prompt
    
    # Process each prompt
    total_start_time = time.time()
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts_to_process):
        global_idx = args.start_idx + i
        
        # Generate filename
        if args.use_index_only:
            filename = f"{args.filename_prefix}{global_idx:04d}"
        else:
            prompt_part = sanitize_filename(prompt)
            filename = f"{args.filename_prefix}{global_idx:04d}_{prompt_part}"
        
        output_path = os.path.join(args.output_dir, f"{filename}.mp4")
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prompts_to_process)}] Generating video {global_idx}")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
        
        try:
            video_start_time = time.time()
            
            # Set seed for this video if specified
            kwargs = generate_kwargs.copy()
            if args.seed is not None:
                kwargs["seed"] = args.seed + i  # Different seed per video but reproducible
            
            # Generate video
            result = generator.generate_video(
                prompt=prompt,
                output_path=output_path,
                save_video=True,
                guidance_scale=3.0,
                **kwargs
            )
            
            video_time = time.time() - video_start_time
            print(f"✓ Video generated in {video_time:.2f}s")
            successful += 1
            
        except Exception as e:
            print(f"✗ Failed to generate video: {e}")
            failed += 1
            continue
    
    # Summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(prompts_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    if successful > 0:
        print(f"Average time per video: {total_time/successful:.2f}s")
    print(f"Output directory: {args.output_dir}")
    
    # Shutdown generator
    generator.shutdown()


if __name__ == "__main__":
    main()

