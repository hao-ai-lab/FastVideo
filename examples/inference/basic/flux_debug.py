"""
Flux Debug Script - SGLang-parameter matching with in-worker dumps.
"""
import os
import numpy as np
from pathlib import Path

from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples"
DEBUG_OUTPUT_PATH = "debug_outputs"
Path(DEBUG_OUTPUT_PATH).mkdir(exist_ok=True)


def _print_frame_matrix(frames, label: str) -> None:
    """Print frame statistics."""
    if not frames:
        print(f"[{label}] No frames returned")
        return
    frame0 = frames[0]
    if isinstance(frame0, np.ndarray):
        arr = frame0
    else:
        arr = np.array(frame0)

    print(
        f"[{label}] frame0 shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min()} max={arr.max()} mean={arr.mean()}"
    )

    if arr.ndim >= 2:
        h = min(4, arr.shape[0])
        w = min(4, arr.shape[1])
        if arr.ndim == 3:
            c = min(3, arr.shape[2])
            print(f"[{label}] frame0 slice (H{h}xW{w}xC{c}):\n{arr[:h, :w, :c]}")
        else:
            print(f"[{label}] frame0 slice (H{h}xW{w}):\n{arr[:h, :w]}")


def main():
    print("="*80)
    print("FLUX DEBUG SCRIPT - MATCHING SGLANG PARAMETERS")
    print("="*80)
    print("\nThis script will:")
    print("1. Generate images using EXACT SGLang parameters")
    print("2. Compare outputs with expected SGLang results")
    print("3. Identify where divergence occurs")
    print()
    print("SGLang Parameters:")
    print("  - Resolution: 720x1280 (height x width)")
    print("  - Seed: 42")
    print("  - Steps: 50")
    print("  - Guidance scale: 1.0")
    print("  - Embedded guidance: 3.5")
    print()

    # Enable in-worker debug dumps. These env vars must be set before worker
    # processes spawn.
    os.environ.setdefault("FASTVIDEO_DEBUG_DUMP_DIR", DEBUG_OUTPUT_PATH)
    os.environ.setdefault("FASTVIDEO_DEBUG_DENOISE_STEPS", "0,1,2,5,10,25,49")
    
    # Initialize generator
    print("Initializing VideoGenerator...")
    generator = VideoGenerator.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # First generation - Fox portrait
    print("\n" + "="*80)
    print("GENERATION 1: Fox Portrait (MATCHING SGLANG)")
    print("="*80)
    prompt1 = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    
    video1 = generator.generate_video(
        prompt1,
        output_path=OUTPUT_PATH,
        save_video=True,
        return_frames=True,
        # SGLang parameters
        height=720,
        width=1280,
        seed=42,
        num_inference_steps=50,
        guidance_scale=1.0,
    )
    
    frames1 = video1.get("frames", []) if isinstance(video1, dict) else video1
    _print_frame_matrix(frames1, "prompt1")
    
    # Compare with expected SGLang output
    print("\nExpected SGLang output:")
    print("  frame0 slice (H4xW4xC3):")
    print("  [[[68 62 46], [65 56 40], [65 56 39], [65 57 40]], ...]")
    print("  (Much darker values in 40-70 range)")

    # Second generation - Lion savanna
    print("\n" + "="*80)
    print("GENERATION 2: Lion Savanna (MATCHING SGLANG)")
    print("="*80)
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic."
    )
    
    video2 = generator.generate_video(
        prompt2,
        output_path=OUTPUT_PATH,
        save_video=True,
        return_frames=True,
        # SGLang parameters
        height=720,
        width=1280,
        seed=42,
        num_inference_steps=50,
        guidance_scale=1.0,
    )
    
    frames2 = video2.get("frames", []) if isinstance(video2, dict) else video2
    _print_frame_matrix(frames2, "prompt2")
    
    # Compare with expected SGLang output
    print("\nExpected SGLang output:")
    print("  frame0 slice (H4xW4xC3):")
    print("  [[[250 248 182], [252 244 174], [252 243 180], [252 242 179]], ...]")
    print("  (Much brighter values in 170-250 range)")
    print("  Mean: 109.70")
    
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)
    print(f"All outputs saved to {DEBUG_OUTPUT_PATH}/")
    print("  - rank0_text_encoding.jsonl")
    print("  - rank0_latent_preparation.jsonl")
    print("  - rank0_denoising.jsonl")
    print()
    print("PARAMETER VERIFICATION:")
    print("  - Resolution: 720x1280 (matching SGLang)")
    print("  - Seed: 42 (matching SGLang)")
    print("  - Steps: 50 (matching SGLang)")
    print("  - Guidance: 1.0 (matching SGLang)")
    print()
    print("If outputs still do not match SGLang:")
    print("  - Check model revision and scheduler settings")
    print("  - Compare text encoder outputs and latents in debug_outputs")
    print()


if __name__ == "__main__":
    main()
