# SPDX-License-Identifier: Apache-2.0
"""Compare H3 decoders on saved normalized video rows without rerunning denoise.

The input NPZ must contain a ``video`` array of packed diffusion rows. Geometry
and the DiT manifest must match the generation that produced those rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--backends", nargs="+", choices=("h3-vae", "taeh3"), default=["h3-vae", "taeh3"])
    parser.add_argument("--taeh3-checkpoint", type=Path)
    parser.add_argument("--taeh3-chunk-size", type=int, default=5)
    parser.add_argument("--vae-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory to preserve previous results")

    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline, _cleanup_mlx
    from fastvideo.mlx_runtime.minimax_h3_taeh3 import ensure_taeh3_checkpoint

    with np.load(args.latents) as archive:
        rows = archive["video"]
    if not np.isfinite(rows).all():
        raise ValueError("The saved video rows contain non-finite values")
    args.output_dir.mkdir(parents=True)
    checksum = hashlib.sha256()
    with args.latents.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(chunk)
    checkpoint = ensure_taeh3_checkpoint(args.taeh3_checkpoint) if "taeh3" in args.backends else None
    report = {
        "mlx": mx.__version__,
        "platform": platform.platform(),
        "device": mx.device_info(),
        "latents": str(args.latents.resolve()),
        "latents_sha256": checksum.hexdigest(),
        "geometry": [args.height, args.width, args.num_frames],
        "dtype": args.vae_dtype,
        "chunk_size": args.taeh3_chunk_size,
        "checkpoint_download_excluded": True,
        "decoder_loading_included": True,
        "trials": [],
    }
    for repeat in range(args.repeats):
        # Reverse each paired trial's order to expose warmup/order effects.
        order = args.backends if repeat % 2 == 0 else args.backends[::-1]
        for backend in order:
            pipeline = MiniMaxH3MLXPipeline(model_root=args.model_root,
                                           mlx_dit_checkpoint=args.mlx_checkpoint,
                                           video_decode_backend=backend,
                                           taeh3_checkpoint=checkpoint if backend == "taeh3" else None,
                                           taeh3_chunk_size=args.taeh3_chunk_size,
                                           vae_dtype=args.vae_dtype)
            _cleanup_mlx()
            mx.reset_peak_memory()
            before = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
            started = time.perf_counter()
            frames = pipeline.decode_video(rows, height=args.height, width=args.width, num_frames=args.num_frames)
            elapsed = time.perf_counter() - started
            trial = {
                "repeat": repeat,
                "backend": backend,
                "decode_s": elapsed,
                "shape": list(frames.shape),
                "peak_active_gib": mx.get_peak_memory() / 2**30,
                "process_lifetime_rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
                "swap_before": before,
                "swap_after": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip(),
            }
            if repeat == 0:
                np.save(args.output_dir / f"{backend}_frames.npy", frames)
            report["trials"].append(trial)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(trial), flush=True)
            del frames, pipeline
            _cleanup_mlx()


if __name__ == "__main__":
    main()
