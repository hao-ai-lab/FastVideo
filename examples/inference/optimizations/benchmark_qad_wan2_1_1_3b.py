"""Standalone benchmark harness for the FastWan-QAD Wan2.1-1.3B recipes.

Measures steady-state end-to-end generation latency (denoise + optional TAEHV
decode) over N timed runs after a warmup phase, and prints a min/max/mean/std
stats table. This is the multi-run timing machinery that used to live in
``FastWan_QAD_TAEHV.py``, kept as a separate harness so the example scripts
stay lean single-run examples.

Pipeline setup mirrors the corresponding example script in this directory:
    fp8        -> fp8_wan2_1_1_3b.py        (FastVideo/FastWan-QAD-FP8-1.3B, SAGE_ATTN)
    nvfp4_qat  -> nvfp4_qat_wan2_1_1_3b.py  (FastVideo/FastWan-QAD-1.3B, ATTN_QAT_INFER)
    nvfp4_sa2  -> nvfp4_sa2_wan2_1_1_3b.py  (FastVideo/FastWan-QAD-1.3B-SA2, SAGE_ATTN)

Hardware requirements per mode (default quantized paths; --bf16 relaxes them):
    fp8        sm89+ (H100, L40S, RTX 4090, Ada Lovelace, or newer)
    nvfp4_qat  RTX 5090-class Blackwell (sm_120a): flashinfer FP4 gemm + attn_qat_infer
    nvfp4_sa2  RTX 5090-class Blackwell (sm_120a): flashinfer FP4 gemm + SageAttention2

Usage:
    python benchmark_qad_wan2_1_1_3b.py --mode nvfp4_qat --taehv-checkpoint /path/to/taehv/taew2_1.pth
    python benchmark_qad_wan2_1_1_3b.py --mode fp8 --warmups 5 --runs 20
    python benchmark_qad_wan2_1_1_3b.py --mode nvfp4_sa2 --bf16    # BF16 baseline
"""

import argparse
import contextlib
import logging
import os
import statistics
import time

import torch

from _qad_common import flashinfer_arch_list, require_fp4_capable_gpu

OUTPUT_PATH = "video_samples"

# mode -> (default model, attention backend, needs the flashinfer FP4 gemm path)
MODES = {
    "fp8": ("FastVideo/FastWan-QAD-FP8-1.3B", "SAGE_ATTN", False),
    "nvfp4_qat": ("FastVideo/FastWan-QAD-1.3B", "ATTN_QAT_INFER", True),
    "nvfp4_sa2": ("FastVideo/FastWan-QAD-1.3B-SA2", "SAGE_ATTN", True),
}

PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
)


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
        decoded = self.model.decode_video(latents, parallel=True, show_progress_bar=False)
        frames = (decoded[0].clamp(0, 1) * 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1).cpu().numpy()


@contextlib.contextmanager
def silence_request_log():
    """Quiet ``VideoGenerator.generate``'s per-request multi-line INFO printout
    so the per-run timing lines stay readable."""
    vg_logger = logging.getLogger("fastvideo.entrypoints.video_generator")
    prev_level = vg_logger.level
    vg_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        vg_logger.setLevel(prev_level)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-run QAD Wan2.1-1.3B benchmark")
    parser.add_argument("--mode", required=True, choices=sorted(MODES),
                        help="Which QAD recipe to benchmark (mirrors the example script of the same name)")
    parser.add_argument("--warmups", type=int, default=5,
                        help="Warmup runs before timing (default: 5)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Timed runs to collect min/max/mean/std over (default: 20)")
    parser.add_argument("--bf16", action="store_true", help="BF16 baseline (no quantization)")
    parser.add_argument("--granularity", choices=["tensor", "channel"], default="tensor",
                        help="FP8 weight scale granularity (fp8 mode only)")
    parser.add_argument("--taehv-checkpoint", default=None, metavar="PATH",
                        help="Path to taew2_1.pth; enables TAEHV tiny autoencoder decoding")
    parser.add_argument("--model", default=None,
                        help="Model path or HuggingFace ID (default: per --mode)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for the DiT")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--infer_steps", type=int, default=3)
    args = parser.parse_args()

    model, backend, needs_fp4 = MODES[args.mode]
    if args.model:
        model = args.model

    if needs_fp4 and not args.bf16:
        require_fp4_capable_gpu()

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", backend)
    if needs_fp4:
        os.environ["FASTVIDEO_DISABLE_ATTENTION_COMPILE"] = "0"
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", flashinfer_arch_list())

    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig

    mode = "bf16" if args.bf16 else args.mode
    if not args.no_compile:
        mode += "_compile"
    use_taehv = args.taehv_checkpoint is not None
    print(f"Mode: {mode.upper()}  model={model}  " + ("decoder=TAEHV" if use_taehv else "decoder=VAE"))

    taehv = TaehvDecoder(args.taehv_checkpoint) if use_taehv else None

    pipeline_config = PipelineConfig.from_pretrained(model)
    pipeline_config.text_encoder_precisions = ("bf16",)
    if not args.bf16:
        if args.mode == "fp8":
            from fastvideo.layers.quantization import get_quantization_config
            pipeline_config.dit_config.quant_config = get_quantization_config("FP8")(
                granularity=args.granularity)
        else:
            from fastvideo.layers.quantization.nvfp4_qat_config import NVFP4QATConfig
            pipeline_config.dit_config.quant_config = NVFP4QATConfig()

    generator = VideoGenerator.from_pretrained(
        model,
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

    request = {
        "prompt": PROMPT,
        "sampling": {"num_inference_steps": args.infer_steps, "guidance_scale": 1.0},
        "output": {"save_video": False},
    }

    # Warmup: pay the torch.compile cost and warm TAEHV's cuDNN algo selection
    # so the timed runs below measure steady-state latency only.
    with silence_request_log():
        for _ in range(args.warmups):
            warm = generator.generate(request=request)
            if use_taehv:
                taehv.decode(warm.samples)

    # Benchmark: time each run end-to-end. ``denoise`` is the generator's own
    # generation_time; for TAEHV we add the in-script decode. Nothing is written
    # to disk inside the loop so I/O never pollutes the timings.
    denoise_times: list[float] = []
    decode_times: list[float] = []
    totals: list[float] = []
    frames = None
    with silence_request_log():
        for i in range(args.runs):
            result = generator.generate(request=request)
            denoise_elapsed = result.generation_time
            denoise_times.append(denoise_elapsed)

            if use_taehv:
                torch.cuda.synchronize()
                decode_start = time.perf_counter()
                frames = taehv.decode(result.samples)
                torch.cuda.synchronize()
                decode_elapsed = time.perf_counter() - decode_start
                decode_times.append(decode_elapsed)
                total = denoise_elapsed + decode_elapsed
            else:
                total = denoise_elapsed
            totals.append(total)

            line = f"  run {i + 1:02d}/{args.runs}: {total:.3f}s"
            if use_taehv:
                line += f" (denoise {denoise_elapsed:.3f}s + decode {decode_elapsed:.3f}s)"
            print(line)

    if frames is not None:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        import imageio
        output_path = os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")
        imageio.mimsave(output_path, frames, fps=16, format="mp4")
        print(f"Saved video to {output_path}")

    # Report min / max / mean / std over the timed runs.
    def _stat_row(name: str, xs: list[float]) -> str:
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return (f"  {name:<11}{min(xs):>8.3f}{max(xs):>9.3f}"
                f"{statistics.mean(xs):>9.3f}{std:>9.3f}")

    if totals:
        print(f"\n[{mode.upper()}] {args.runs} runs, {args.warmups} warmup, "
              f"{args.infer_steps} steps:")
        print(f"  {'metric':<11}{'min':>8}{'max':>9}{'mean':>9}{'std':>9}   (s)")
        print(_stat_row("total", totals))
        if use_taehv:
            print(_stat_row("denoise", denoise_times))
            print(_stat_row("decode", decode_times))

    generator.shutdown()


if __name__ == "__main__":
    main()
