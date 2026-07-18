"""DGX Spark (GB10) reproduction benchmark for the performance guide.

Reproduces the two headline claims in
``docs/getting_started/installation/spark_performance.md`` on your own GB10:

  1. A distilled few-step model is usable (~30 s/video) and **decode-bound** on
     the Spark's unified LPDDR5X memory.
  2. bf16 VAE decode is essentially lossless (MS-SSIM ~1.0 vs fp32) and modestly
     faster — which is why FastVideo already defaults Wan's *decode* to bf16.

Two parts, both in-process so they control for the Spark's run-to-run variance:

  A. **Generation timing** — loads a distilled model once, generates ``--runs``
     videos back-to-back (after ``--warmup``), reports the *median* generation
     time. Set ``FASTVIDEO_STAGE_LOGGING=1`` to also see the per-stage
     (denoise / VAE decode / text-encode) split that shows the decode bottleneck.

  B. **Decode precision A/B** — decodes ONE fixed latent fp32-vs-bf16 in the same
     process and reports MS-SSIM + speedup. This isolates the decode delta with
     no denoise non-determinism and no video-codec noise.

**Why median, not a single run:** on the GB10 a 3-step generation is dominated by
one-time per-process startup (Triton autotune, allocator warmup) that never
amortizes over so few steps, so single-run totals wobble ~±30%. Always compare
few-step levers back-to-back / as medians, never as two separate single runs.

Run it safely on a shared box (see the best-practices note in the perf guide):

    FASTVIDEO_STAGE_LOGGING=1 nice -n 19 nohup \
        python examples/inference/optimizations/spark_benchmark.py > spark_bench.log 2>&1 &
    tail -f spark_bench.log

Knobs (flags or env): --model, --runs (3), --warmup (1), --steps (3),
--frames (81), --height (448), --width (832), --seed (42), --out, --skip-gen,
--skip-decode.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch


def _p(msg: str) -> None:
    print(f"[spark-bench] {msg}", flush=True)


def bench_generation(args) -> None:
    """Part A: median few-step generation time on a distilled model."""
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    # VSA auto-routes to the Triton kernel on sm_121; do NOT force TORCH_SDPA on a
    # VSA checkpoint (the SDPA path builds a model without the gate weights the
    # checkpoint carries and fails to load).
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    _p(f"loading {args.model} ...")
    load_t0 = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        # Leave offload at these defaults: "CPU" offload is the same unified RAM
        # on the GB10, so the win is tiling + sane resolution, not offloading.
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=0.8,
    )
    _p(f"model loaded in {time.perf_counter() - load_t0:.1f}s")

    sampling_param = SamplingParam.from_pretrained(args.model)
    sampling_param.num_frames = args.frames
    sampling_param.height = args.height
    sampling_param.width = args.width
    sampling_param.num_inference_steps = args.steps
    sampling_param.seed = args.seed

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, "
        "its eyes wide with interest. Soft natural light, warm cheerful tones, "
        "mid-shot, cinematic.")

    def _gen():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = generator.generate_video(prompt, output_path=args.out,
                                          save_video=True,
                                          sampling_param=sampling_param)
        torch.cuda.synchronize()
        return getattr(video, "generation_time", time.perf_counter() - t0)

    for _ in range(args.warmup):
        _gen()

    # Peak GPU memory: report torch's own allocator high-water mark, not
    # nvidia-smi. On the GB10's unified pool nvidia-smi reads [N/A] and the
    # system "used" figure conflates CPU+GPU+cache; torch.cuda.max_memory_reserved
    # is the model's actual GPU-side footprint.
    torch.cuda.reset_peak_memory_stats()

    times = []
    for i in range(args.runs):
        dt = _gen()
        times.append(dt)
        _p(f"gen run {i + 1}/{args.runs}: {dt:.2f}s")

    med = statistics.median(times)
    peak_gb = torch.cuda.max_memory_reserved() / 1e9
    free_b, total_b = torch.cuda.mem_get_info()
    _p(f"median generation time over {args.runs} runs "
       f"({args.warmup} warmup, {args.steps} steps): {med:.2f}s")
    _p(f"peak GPU memory (torch reserved): {peak_gb:.1f} GB   "
       f"| unified pool free/total: {free_b / 1e9:.1f}/{total_b / 1e9:.1f} GB")
    _p("set FASTVIDEO_STAGE_LOGGING=1 to see the denoise / decode / text split "
       "(few-step generation is VAE-decode-bound on the GB10).")
    generator.shutdown()


def bench_decode_precision(args) -> None:
    """Part B: fp32-vs-bf16 VAE decode of one fixed latent (SSIM + speedup)."""
    try:
        from diffusers import AutoencoderKLWan
        from torchmetrics.functional import (
            multiscale_structural_similarity_index_measure as msssim)
    except ImportError as e:  # torchmetrics is not a hard FastVideo dep
        _p(f"skipping decode A/B (missing dependency: {e}); "
           "`uv pip install torchmetrics` to enable it.")
        return

    dev = "cuda"
    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()

    # Wan latent geometry for height x width x frames, patch (4,8,8):
    #   T_lat = (frames - 1) // 4 + 1,  H_lat = height // 8,  W_lat = width // 8
    t_lat = (args.frames - 1) // 4 + 1
    z = torch.randn(1, 16, t_lat, args.height // 8, args.width // 8,
                    device=dev, dtype=torch.float32)

    def _decode(autocast: bool):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=autocast):
            out = vae.decode(z, return_dict=False)[0]
        torch.cuda.synchronize()
        return out.float(), time.perf_counter() - t0

    def _frames(o):  # (1,3,T,H,W) [-1,1] -> (T,3,H,W) [0,1]
        return ((o.clamp(-1, 1) + 1) / 2)[0].permute(1, 0, 2, 3).contiguous()

    _decode(False)  # warm both paths (excluded from timing)
    _decode(True)

    o32, t32 = _decode(autocast=False)          # fp32
    o16, t16 = _decode(autocast=True)           # bf16 (== vae_decode_precision="bf16")
    ssim = msssim(_frames(o16), _frames(o32), data_range=1.0).item()

    _p(f"fp32 decode : {t32 * 1000:8.1f} ms")
    _p(f"bf16 decode : {t16 * 1000:8.1f} ms   ({t32 / t16:.2f}x faster)")
    _p(f"MS-SSIM(bf16, fp32) on identical latent: {ssim:.4f}  "
       "(>= ~0.99 -> lossless; this is why Wan decode defaults to bf16)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="spark_bench_samples")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip Part A (generation timing)")
    ap.add_argument("--skip-decode", action="store_true",
                    help="skip Part B (decode precision A/B)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (run this on the GB10).")
    cap = torch.cuda.get_device_capability()
    _p(f"{torch.cuda.get_device_name()} (cc {cap[0]}.{cap[1]}), "
       f"torch {torch.__version__}")

    if not args.skip_gen:
        bench_generation(args)
    if not args.skip_decode:
        bench_decode_precision(args)


if __name__ == "__main__":
    main()
