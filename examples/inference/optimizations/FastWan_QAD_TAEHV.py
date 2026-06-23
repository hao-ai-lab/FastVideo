"""Fast NVFP4 linear inference for Wan2.1-T2V-1.3B with TAEHV decoding.

This is the FP4-linear fast path from ``fp4_linear_wan2_1_1_3b.py`` with the
heavy Wan VAE swapped out for TAEHV -- a tiny autoencoder that decodes Wan2.1
latents directly (no denormalization) and is dramatically faster / lighter.

How it works: the generator runs with ``output_type="latent"`` so the pipeline
returns raw denoised latents instead of pixels (the Wan VAE is offloaded and
never used). We then decode those latents with TAEHV in this script and save
the frames ourselves. This mirrors the FastVideo-Quantization
``quantization_example_taehv.py`` proof-of-concept, but kept clean: TAEHV is a
pip package (no ``sys.path`` hacks), the latent->uint8 conversion is vectorized,
and there is no dead profiler / sanitization code.

Requirements:
    - Blackwell GPU (B200/B300, sm100a/sm103a) for the FP4 linear path
    - flashinfer (``pip install flashinfer-python``)
    - TAEHV weights ``taew2_1.pth`` (https://github.com/madebyollin/taehv)

Usage:
    python fp4_linear_taehv_wan2_1_1_3b.py                    # FP4 + TAEHV + compile
    python fp4_linear_taehv_wan2_1_1_3b.py --no-taehv         # FP4 + full Wan VAE
    python fp4_linear_taehv_wan2_1_1_3b.py --no-compile       # eager
    python fp4_linear_taehv_wan2_1_1_3b.py --baseline         # dense bf16 reference
    python fp4_linear_taehv_wan2_1_1_3b.py --distilled_model ''  # base Wan2.1 weights
    python fp4_linear_taehv_wan2_1_1_3b.py --warmups 5 --benchmark-runs 20  # timing stats (default)
"""

import argparse
import contextlib
import logging
import os
import statistics
import time

import imageio
import torch

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_qat_config import NVFP4QATConfig

OUTPUT_PATH = "video_samples"

# Distilled, quantization-aware (QAD) transformer for Wan2.1-1.3B (3 steps,
# guidance 1.0). Loaded on top of the base Wan2.1 pipeline; pass
# ``--distilled_model ''`` to run the base weights instead.
DEFAULT_DISTILLED_MODEL = "FastVideo/FastWan-QAD-1.3B"
DISTILLED_WEIGHTS_FILE = (
    "generator_inference_transformer/diffusion_pytorch_model.safetensors"
)

# TAEHV checkpoint for Wan2.1. Clone https://github.com/madebyollin/taehv to get
# ``taew2_1.pth`` (Wan 2.1 / Wan 2.2-14B / Qwen-Image all use this VAE).
DEFAULT_TAEHV_CHECKPOINT = "/workspace/taehv/taew2_1.pth"

PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
)


class TaehvDecoder:
    """Thin wrapper around the TAEHV tiny autoencoder for Wan2.1 latents.

    TAEHV consumes the *normalized* latents the diffusion model produces (the
    same representation FastVideo carries internally), so no denormalization is
    needed -- unlike the full Wan VAE path.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float16) -> None:
        from taehv import TAEHV  # pip-installed; no sys.path manipulation
        self.device = device
        self.dtype = dtype
        print(f"Loading TAEHV from {checkpoint_path} ...")
        self.model = TAEHV(checkpoint_path=checkpoint_path).to(device, dtype).eval()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        """Decode FastVideo latents into uint8 RGB frames.

        Args:
            latents: ``[B, C, T, H, W]`` (NCTHW) normalized latent tensor.

        Returns:
            A ``(T, H, W, 3)`` uint8 numpy array ready for ``imageio.mimsave``.
        """
        # NCTHW -> NTCHW (TAEHV's expected layout), on the TAEHV device/dtype.
        latents = latents.permute(0, 2, 1, 3, 4).to(self.device, self.dtype)
        decoded = self.model.decode_video(
            latents, parallel=True, show_progress_bar=False)
        # decoded: [B, T, 3, H, W] in [0, 1]. Take batch 0, vectorize to uint8.
        frames = (decoded[0].clamp(0, 1) * 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1).cpu().numpy()


def resolve_distilled_weights(hf_id: str) -> str:
    """Return a local path to the distilled transformer safetensors."""
    if os.path.exists(hf_id):
        return hf_id
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=hf_id, filename=DISTILLED_WEIGHTS_FILE)


@contextlib.contextmanager
def silence_request_log():
    """Quiet ``VideoGenerator.generate``'s per-request config printout.

    Each ``generate(...)`` call logs a multi-line debug block (height/width/
    prompt/steps/...) at INFO via ``logger.info`` in
    ``fastvideo.entrypoints.video_generator``. There is no built-in switch,
    so this context manager raises that logger's level to WARNING while the
    warmup calls run, then restores it for the timed run.
    """
    vg_logger = logging.getLogger("fastvideo.entrypoints.video_generator")
    prev_level = vg_logger.level
    vg_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        vg_logger.setLevel(prev_level)


def resolve_taehv_checkpoint(path: str) -> str:
    """Validate the TAEHV checkpoint path, with a helpful error if missing."""
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"TAEHV checkpoint not found at {path!r}. Clone the weights with:\n"
        "    git clone https://github.com/madebyollin/taehv\n"
        "and pass --taehv_checkpoint <repo>/taew2_1.pth")


def build_generator(args: argparse.Namespace) -> VideoGenerator:
    model_id = args.model

    # Half precision everywhere; DiT linears are additionally NVFP4-quantized
    # via dit_config.quant_config below.
    pipeline_config = PipelineConfig.from_pretrained(model_id)
    pipeline_config.dit_precision = "bf16"
    pipeline_config.vae_precision = "bf16"
    pipeline_config.text_encoder_precisions = ("bf16",)

    if not args.baseline:
        pipeline_config.dit_config.quant_config = NVFP4QATConfig()

    compile_enabled = not args.no_compile

    extra_kwargs = {}
    if args.distilled_model:
        weights_path = resolve_distilled_weights(args.distilled_model)
        print(f"Using distilled weights: {args.distilled_model} -> {weights_path}")
        extra_kwargs["init_weights_from_safetensors"] = weights_path

    if args.taehv:
        # Skip the in-pipeline VAE decode entirely: the pipeline returns raw
        # latents, the Wan VAE is offloaded to CPU (and not compiled) since we
        # decode with TAEHV in this script instead.
        extra_kwargs["output_type"] = "latent"

    generator = VideoGenerator.from_pretrained(
        model_id,
        pipeline_config=pipeline_config,
        num_gpus=args.num_gpus,
        # Keep everything resident on the GPU -- no offloading, except the
        # unused Wan VAE when TAEHV handles decoding.
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=args.taehv,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=compile_enabled,
        enable_torch_compile_text_encoder=compile_enabled,
        enable_torch_compile_vae=compile_enabled and not args.taehv,
        **extra_kwargs,
    )
    return generator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FP4 linear Wan2.1-1.3B with TAEHV decoding benchmark")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--baseline", action="store_true",
                        help="Run dense bf16 instead of FP4 linear")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (eager)")
    parser.add_argument("--taehv", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Decode with TAEHV instead of the full Wan VAE "
                             "(use --no-taehv for the Wan VAE path)")
    parser.add_argument("--taehv_checkpoint", default=DEFAULT_TAEHV_CHECKPOINT,
                        help="Path to the TAEHV taew2_1.pth checkpoint")
    parser.add_argument("--distilled_model", default=DEFAULT_DISTILLED_MODEL,
                        help="HuggingFace ID (or local path) of a distilled "
                             "transformer checkpoint to load on top of --model. "
                             "Pass '' to use the base --model weights instead.")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--infer_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--warmups", type=int, default=5,
                        help="Warmup runs before timing (default: 5).")
    parser.add_argument("--benchmark-runs", type=int, default=20,
                        help="Timed runs to collect min/max/mean/std over (default: 20).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for FP4 inference.")

    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (capability {cap[0]}.{cap[1]})")
    if not args.baseline and cap[0] < 10:
        print("Warning: NVFP4 requires Blackwell (capability 10.0+); "
              "FP4 kernels may be unavailable on this GPU.")

    mode = "bf16" if args.baseline else "fp4_linear"
    mode += "_taehv" if args.taehv else "_wanvae"
    if not args.no_compile:
        mode += "_compile"
    print(f"Mode: {mode.upper()}")

    # Load TAEHV before the (slow) generator build so a bad checkpoint path
    # fails fast.
    taehv = TaehvDecoder(resolve_taehv_checkpoint(args.taehv_checkpoint)) \
        if args.taehv else None

    generator = build_generator(args)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Warmup: pay the DiT torch.compile cost and warm TAEHV's cuDNN algo
    # selection + allocator growth (~0.2s on the first decode) so the timed
    # runs below measure steady-state latency only.
    with silence_request_log():
        for _ in range(args.warmups):
            warm = generator.generate(request={
                "prompt": PROMPT,
                "sampling": {"num_inference_steps": 2, "guidance_scale": args.guidance_scale},
                "output": {"save_video": False, "return_frames": args.taehv},
            })
            if args.taehv:
                taehv.decode(warm.samples)

    output_path = os.path.join(OUTPUT_PATH, f"raccoon_{mode}.mp4")

    # Benchmark: time each run end-to-end. ``denoise`` is the generator's own
    # generation_time; for TAEHV we add the in-script decode. The mp4 is not
    # written inside the loop (only the final run's frames are saved below) so
    # disk I/O never pollutes the timings.
    denoise_times: list[float] = []
    decode_times: list[float] = []
    totals: list[float] = []
    frames = None
    with silence_request_log():
        for i in range(args.benchmark_runs):
            result = generator.generate(request={
                "prompt": PROMPT,
                "sampling": {
                    "num_inference_steps": args.infer_steps,
                    "guidance_scale": args.guidance_scale,
                },
                "output": {
                    "save_video": False,
                    "return_frames": args.taehv,
                    "output_path": output_path,
                },
            })
            denoise_elapsed = result.generation_time
            denoise_times.append(denoise_elapsed)

            if args.taehv:
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

            line = f"  run {i + 1:02d}/{args.benchmark_runs}: {total:.3f}s"
            if args.taehv:
                line += (f" (denoise {denoise_elapsed:.3f}s + "
                         f"decode {decode_elapsed:.3f}s)")
            print(line)

    if args.taehv and frames is not None:
        imageio.mimsave(output_path, frames, fps=16, format="mp4")
        print(f"Saved video to {output_path}")

    # Report min / max / mean / std over the timed runs.
    def _stat_row(name: str, xs: list[float]) -> str:
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return (f"  {name:<11}{min(xs):>8.3f}{max(xs):>9.3f}"
                f"{statistics.mean(xs):>9.3f}{std:>9.3f}")

    if totals:
        print(f"\n[{mode.upper()}] {args.benchmark_runs} runs, {args.warmups} warmup, "
              f"{args.infer_steps} steps:")
        print(f"  {'metric':<11}{'min':>8}{'max':>9}{'mean':>9}{'std':>9}   (s)")
        print(_stat_row("total", totals))
        if args.taehv:
            print(_stat_row("denoise", denoise_times))
            print(_stat_row("decode", decode_times))

    generator.shutdown()


if __name__ == "__main__":
    main()
