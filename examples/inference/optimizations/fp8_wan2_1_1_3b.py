"""FP8 weight quantization inference example.

Runs FastWan-QAD-FP8-1.3B (a distilled Wan2.1-T2V-1.3B-Diffusers checkpoint) with
FP8 e4m3 quantized DiT linear layers. Uses the SAGE_ATTN attention backend.

Requirements:
    - GPU: sm89+ (H100, L40S, RTX 4090, Ada Lovelace, or newer)
      Falls back to a bf16 dequant path on older GPUs.
    - TAEHV (optional): Follow install instructions at https://github.com/madebyollin/taehv

Usage:
    python fp8_wan2_1_1_3b.py --taehv-checkpoint /path/to/taehv/taew2_1.pth                           # FP8 per-tensor
    python fp8_wan2_1_1_3b.py --taehv-checkpoint /path/to/taehv/taew2_1.pth --bf16                    # BF16 baseline
    python fp8_wan2_1_1_3b.py --taehv-checkpoint /path/to/taehv/taew2_1.pth --granularity channel     # per-channel
"""

import argparse
import os
import time

import torch

OUTPUT_PATH = "video_samples"


class TaehvDecoder:
    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float16) -> None:
        from taehv import TAEHV
        self.device = device
        self.dtype = dtype
        print(f"Loading TAEHV from {checkpoint_path} ...")
        self.model = TAEHV(checkpoint_path=checkpoint_path).to(device, dtype).eval()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        latents = latents.permute(0, 2, 1, 3, 4).to(self.device, self.dtype)
        decoded = self.model.decode_video(latents, parallel=False, show_progress_bar=False)
        frames = (decoded[0].clamp(0, 1) * 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1).cpu().numpy()


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
    from fastvideo.configs.pipelines.base import PipelineConfig

    mode = "bf16" if args.bf16 else f"fp8_{args.granularity}"
    if not args.no_compile:
        mode += "_compile"
    use_taehv = args.taehv_checkpoint is not None
    print(f"Mode: {mode.upper()}" + ("  decoder=TAEHV" if use_taehv else "  decoder=VAE"))

    taehv = TaehvDecoder(args.taehv_checkpoint) if use_taehv else None

    pipeline_config = PipelineConfig.from_pretrained(args.model)
    pipeline_config.text_encoder_precisions = ("bf16",)
    if not args.bf16:
        from fastvideo.layers.quantization import get_quantization_config
        pipeline_config.dit_config.quant_config = get_quantization_config("FP8")(granularity=args.granularity)

    generator = VideoGenerator.from_pretrained(
        args.model,
        pipeline_config=pipeline_config,
        num_gpus=args.num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=use_taehv,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=not args.no_compile,
        enable_torch_compile_text_encoder=not args.no_compile,
        enable_torch_compile_vae=not args.no_compile and not use_taehv,
        output_type="latent" if use_taehv else "pil",
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    n_warmup = 2 if not args.no_compile else 0
    for _ in range(n_warmup):
        warmup_result = generator.generate(request={"prompt": prompt, "sampling": {"num_inference_steps": 3, "guidance_scale": 1.0},
                                                    "output": {"save_video": False}})
        if use_taehv:
            taehv.decode(warmup_result.samples)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    video_path = os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")
    if use_taehv:
        import imageio
        result = generator.generate(request={
            "prompt": prompt,
            "sampling": {"num_inference_steps": args.infer_steps, "guidance_scale": 1.0},
            "output": {"save_video": False},
        })
        denoise_elapsed = result.generation_time
        torch.cuda.synchronize()
        t_decode = time.perf_counter()
        frames = taehv.decode(result.samples)
        torch.cuda.synchronize()
        decode_elapsed = time.perf_counter() - t_decode
        total = denoise_elapsed + decode_elapsed
        imageio.mimsave(video_path, frames, fps=16, format="mp4")
        print(f"Saved TAEHV-decoded video to: {video_path}")
        print(f"[{mode.upper()}] {args.infer_steps} steps in {total:.3f}s "
              f"(denoise {denoise_elapsed:.3f}s + decode {decode_elapsed:.3f}s)")
    else:
        result = generator.generate(request={
            "prompt": prompt,
            "sampling": {"num_inference_steps": args.infer_steps, "guidance_scale": 1.0},
            "output": {"save_video": True, "output_path": video_path},
        })
        elapsed = result.generation_time
        print(f"[{mode.upper()}] {args.infer_steps} steps in {elapsed:.3f}s "
              f"({args.infer_steps / elapsed:.2f} it/s)")

    generator.shutdown()


if __name__ == "__main__":
    main()
