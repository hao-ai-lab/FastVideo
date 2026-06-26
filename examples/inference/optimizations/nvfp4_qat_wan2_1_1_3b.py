"""NVFP4 + Attn-QAT (modified SageAttention3) inference on Blackwell.

Runs Wan2.1-T2V-1.3B fully in 4-bit: NVFP4 linear layers (activations
quantized on the fly) together with the modified SageAttention3 FP4 attention
backend (``ATTN_QAT_INFER``). This is the inference half of the
Quantization-Aware Distillation (QAD) recipe.

Requirements:
    - RTX 5090 / consumer Blackwell (sm_120a). The attn_qat_infer kernel hard
      gates on sm_120; on other GPUs it falls back to Flash Attention.
    - The attn_qat_infer kernel built into fastvideo-kernel (see #1455) and
      flashinfer for the NVFP4 linear matmuls.

Usage:
    python nvfp4_qat_wan2_1_1_3b.py                 # NVFP4 linear + Attn-QAT attn
    python nvfp4_qat_wan2_1_1_3b.py --bf16          # BF16 baseline
"""

import argparse
import os
import time

OUTPUT_PATH = "video_samples"


def main():
    parser = argparse.ArgumentParser(description="NVFP4 + Attn-QAT video generation")
    parser.add_argument("--bf16", action="store_true",
                        help="BF16 baseline (no NVFP4 linear, default attention)")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--quant-method", default="nvfp4_qat", choices=["nvfp4_qat", "NVFP4"],
                        help="Linear quantization config. Wan-2.1 uses nvfp4_qat (matches its "
                             "to_q/k/v/out + ffn layers); NVFP4 is LTX2-specific and will NOT "
                             "quantize Wan.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the DiT")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--infer_steps", type=int, default=50)
    args = parser.parse_args()

    # The attention backend is selected via env var before the engine starts.
    # ATTN_QAT_INFER -> AttnQatInferBackend (modified SageAttention3 FP4).
    if not args.bf16:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "ATTN_QAT_INFER"

    # Import after the env var so the platform picks up the selection.
    from fastvideo import VideoGenerator
    from fastvideo.layers.quantization import get_quantization_config

    mode = "bf16" if args.bf16 else args.quant_method
    if args.compile:
        mode += "_compile"
    print(f"Mode: {mode.upper()}")

    # transformer_quant needs a QuantizationConfig *instance* — the bare string
    # is not resolved on the from_pretrained kwarg path.
    extra = {} if args.bf16 else {"transformer_quant": get_quantization_config(args.quant_method)()}
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        use_fsdp_inference=args.bf16,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        enable_torch_compile=args.compile,
        **extra,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    n_warmup = 2 if args.compile else 1
    for _ in range(n_warmup):
        generator.generate(request={"prompt": prompt, "sampling": {"num_inference_steps": 2},
                                    "output": {"save_video": False}})

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    start = time.time()
    generator.generate(request={
        "prompt": prompt,
        "sampling": {"num_inference_steps": args.infer_steps},
        "output": {"save_video": True, "output_path": os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")},
    })
    elapsed = time.time() - start
    print(f"[{mode.upper()}] {args.infer_steps} steps in {elapsed:.2f}s "
          f"({args.infer_steps / elapsed:.2f} it/s)")

    generator.shutdown()


if __name__ == "__main__":
    main()
