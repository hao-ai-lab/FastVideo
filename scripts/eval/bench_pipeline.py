"""Throughput / memory bench harness for ``Evaluator.evaluate(samples=...)``.

Same fixture, same metric list, repeatable wall-time + decode/compute
split + peak GPU memory. Used to land the pool architecture and to
detect future perf regressions on the eval path.

Bench fixtures live under ``outputs_video/_bench/sample_{0..N-1}.mp4``
— symlinks to a single LTX2-generated mp4 so decode work is uniform
and timing variance stays low.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from fastvideo.eval import Evaluator
from fastvideo.eval.worker import pop_timings

# Subset chosen to fit one H200 and exercise the decode-then-compute
# pattern. Skipping motion_smoothness (memory-hungry at 1080p).
METRICS = [
    "vbench.aesthetic_quality",
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.imaging_quality",
    "vbench.temporal_flickering",
]

PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions.")

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "outputs_video" / "_bench"


def _peak_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _build_samples(n: int, *, with_reference: bool = False) -> list[dict]:
    """Build N samples pointing at the same mp4. The fixture mp4 has both
    a video stream and an audio track, so the same path serves both
    ``video=`` and ``audio=`` keyed metrics.

    With ``with_reference=True``, also adds ``reference`` keyed at the
    same mp4 — useful for metrics that need a paired reference (lpips,
    ssim, psnr). Self-pair gives ground-truth-perfect scores, which
    makes correctness regressions easy to spot.
    """
    out = []
    for i in range(n):
        path = str(BENCH_DIR / f"sample_{i}.mp4")
        sample: dict = {"video": path, "audio": path, "text_prompt": PROMPT}
        if with_reference:
            sample["reference"] = path
        out.append(sample)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4, help="number of samples to score")
    p.add_argument("--out", default="bench/result.json", help="JSON output path")
    p.add_argument("--metrics", default=",".join(METRICS), help="comma-separated metric names")
    p.add_argument("--reference", action="store_true",
                   help="also pass `reference` keyed at the same mp4 (for lpips/ssim/psnr)")
    args = p.parse_args()

    metrics = args.metrics.split(",")
    samples = _build_samples(args.n, with_reference=args.reference)

    print(f"[bench] building Evaluator(metrics={metrics})")
    ev = Evaluator(metrics=metrics, num_gpus=1)

    # Warmup so weight loads / cache primes aren't billed to the timed run.
    print("[bench] warmup (single sample)…")
    _ = ev.evaluate(**samples[0])
    pop_timings()  # discard warmup timings
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"[bench] timed run, n={args.n}…")
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    results = ev.evaluate(samples=samples)
    t1 = time.perf_counter()
    rss1 = _rss_mb()

    timings = pop_timings()
    elapsed = t1 - t0

    out = {
        "n": args.n,
        "metrics": metrics,
        "elapsed_s": elapsed,
        "per_sample_avg_s": elapsed / args.n,
        "decode_ms_total": timings["decode_ms"],
        "compute_ms_total": timings["compute_ms"],
        "decode_ms_per_sample": timings["decode_ms"] / max(timings["n"], 1),
        "compute_ms_per_sample": timings["compute_ms"] / max(timings["n"], 1),
        "samples_seen_by_workers": timings["n"],
        "peak_gpu_mb": _peak_mem_mb(),
        "host_rss_delta_mb": rss1 - rss0,
        "scores_first_sample": _summarize_first_result(results),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {out_path}")
    print(json.dumps(out, indent=2))


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _summarize_first_result(results) -> dict:
    first = results[0] if isinstance(results, list) else results
    out = {}
    for name, r in first.items():
        out[name] = r.score if r.score is not None else f"SKIPPED({r.details.get('skipped', '?')})"
    return out


if __name__ == "__main__":
    main()
