"""FP8 weight quantization inference example.

Runs Wan2.1-T2V-1.3B with FP8 e4m3 quantized DiT linear layers (attention
projections and FFN). Weights are quantized in-place after loading; activations
are quantized dynamically at runtime. Reduces GPU memory relative to BF16 and
can improve throughput on sm89+ GPUs.

Requirements:
    - GPU: sm89+ (H100, L40S, RTX 4090, Ada Lovelace, or newer)
      Falls back to a bf16 dequant path on older GPUs.
    - TAEHV (optional): Follow install instructions at https://github.com/madebyollin/taehv

Usage:
    python fp8_wan2_1_1_3b.py              # FP8 per-tensor (default)
    python fp8_wan2_1_1_3b.py --bf16       # BF16 baseline
    python fp8_wan2_1_1_3b.py --granularity channel  # per-channel (higher accuracy but slower)
    python fp8_wan2_1_1_3b.py --taehv-checkpoint /path/to/taew2_1.pth
"""

import argparse
import os
import sys
import time

import torch

OUTPUT_PATH = "video_samples"


def load_taehv(checkpoint_path, device="cuda", dtype=torch.float16):
    repo_dir = os.path.dirname(checkpoint_path)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from taehv import TAEHV
    print(f"Loading TAEHV from {checkpoint_path}...")
    model = TAEHV(checkpoint_path=checkpoint_path).to(device, dtype)
    print("TAEHV loaded.")
    return model


@torch.no_grad()  # type: ignore[misc]
def decode_with_taehv(taehv_model, latents):
    latents = latents.permute(0, 2, 1, 3, 4)
    latents = latents.to(device=next(taehv_model.parameters()).device,
                         dtype=next(taehv_model.parameters()).dtype)
    decoded = taehv_model.decode_video(latents, parallel=False, show_progress_bar=False)
    frames = []
    for frame in decoded[0]:
        frame_np = (frame.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
        frames.append(frame_np)
    return frames


def main():
    parser = argparse.ArgumentParser(description="FP8 video generation benchmark")
    parser.add_argument("--bf16", action="store_true",
                        help="BF16 baseline (no FP8 quantization)")
    parser.add_argument("--granularity", choices=["tensor", "channel"], default="tensor",
                        help="FP8 weight scale granularity: tensor (faster) or channel (more accurate)")
    parser.add_argument("--taehv-checkpoint", default=None, metavar="PATH",
                        help="Path to taew2_1.pth; enables TAEHV tiny autoencoder decoding")
    parser.add_argument("--model", default="FastVideo/FastWan-QAD-FP8-1.3B",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for the DiT")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--infer_steps", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")

    from fastvideo import VideoGenerator
    from fastvideo.layers.quantization import get_quantization_config

    mode = "bf16" if args.bf16 else f"fp8_{args.granularity}"
    if not args.no_compile:
        mode += "_compile"
    use_taehv = args.taehv_checkpoint is not None
    print(f"Mode: {mode.upper()}" + ("  decoder=TAEHV" if use_taehv else "  decoder=VAE"))

    taehv_model = load_taehv(args.taehv_checkpoint) if use_taehv else None

    # transformer_quant needs a QuantizationConfig *instance* — the bare string
    # is not resolved on the from_pretrained kwarg path.
    extra = {} if args.bf16 else {
        "transformer_quant": get_quantization_config("FP8")(granularity=args.granularity)
    }
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=args.num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=use_taehv,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=not args.no_compile,
        enable_torch_compile_vae=not args.no_compile and not use_taehv,
        output_type="latent" if use_taehv else "pil",
        **extra,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    n_warmup = 1 if not args.no_compile else 0
    for _ in range(n_warmup):
        generator.generate(request={"prompt": prompt, "sampling": {"num_inference_steps": 3, "guidance_scale": 1.0},
                                    "output": {"save_video": False}})

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    start = time.time()
    if use_taehv:
        result = generator.generate(request={
            "prompt": prompt,
            "sampling": {"num_inference_steps": args.infer_steps, "guidance_scale": 1.0},
            "output": {"save_video": False},
        })
        import imageio
        frames = decode_with_taehv(taehv_model, result.samples)
        video_path = os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")
        imageio.mimsave(video_path, frames, fps=16, format="mp4")
        print(f"Saved TAEHV-decoded video to: {video_path}")
    else:
        generator.generate(request={
            "prompt": prompt,
            "sampling": {"num_inference_steps": args.infer_steps, "guidance_scale": 1.0},
            "output": {"save_video": True, "output_path": os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")},
        })
    elapsed = time.time() - start
    print(f"[{mode.upper()}] {args.infer_steps} steps in {elapsed:.2f}s "
          f"({args.infer_steps / elapsed:.2f} it/s)")

    generator.shutdown()


if __name__ == "__main__":
    main()
