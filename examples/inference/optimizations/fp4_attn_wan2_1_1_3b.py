"""FP4 Flash Attention 4 inference example on Blackwell GPUs.

Quantizes Q and K to NVFP4 E2M1 with per-block E4M3 scale factors,
achieving up to 1.39x attention kernel speedup over BF16 FA4.

Requirements:
    - Blackwell GPU (B200/B300, sm100a/sm103a)
    - flash-attention-fp4, cutlass-dsl, flashinfer
    - See docs/source/getting_started/fp4_attention.md for installation

Usage:
    # FP4 attention
    FASTVIDEO_NVFP4_FA4=1 CUTE_DSL_ENABLE_TVM_FFI=1 python fp4_attention.py

    # BF16 baseline for comparison
    python fp4_attention.py
"""

import argparse
import os
import time

from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples"


def main():
    parser = argparse.ArgumentParser(description="FP4 FA4 video generation benchmark")
    parser.add_argument("--nvfp4_fa4", action="store_true",
                        help="Enable NVFP4 FP4 quantized QK flash attention")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for DIT")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--infer_steps", type=int, default=50)
    args = parser.parse_args()

    if args.nvfp4_fa4:
        os.environ["FASTVIDEO_NVFP4_FA4"] = "1"
        os.environ["CUTE_DSL_ENABLE_TVM_FFI"] = "1"

    mode = "nvfp4" if args.nvfp4_fa4 else "bf16"
    if args.compile:
        mode += "_compile"
    print(f"Mode: {mode.upper()}")

    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        # FSDP shards tensors across GPUs — incompatible with FP4 make_ptr path
        use_fsdp_inference=not args.nvfp4_fa4,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        enable_torch_compile=args.compile,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    # Warmup (2 runs for compile to JIT)
    n_warmup = 2 if args.compile else 1
    for i in range(n_warmup):
        generator.generate_video(prompt, save_video=False, infer_steps=2)

    # Timed run
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    start = time.time()
    generator.generate_video(
        prompt,
        output_path=os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4"),
        save_video=True,
        infer_steps=args.infer_steps,
    )
    elapsed = time.time() - start
    print(f"[{mode.upper()}] {args.infer_steps} steps in {elapsed:.2f}s "
          f"({args.infer_steps / elapsed:.2f} it/s)")

    generator.shutdown()


if __name__ == "__main__":
    main()
