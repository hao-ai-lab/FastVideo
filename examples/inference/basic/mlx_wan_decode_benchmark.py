"""Compare Wan VAE and TAEHV decode on saved FastWan latents."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.inference.basic.mlx_wan_prompt_to_video import DEFAULT_MODEL_ROOT, decode_latents_to_video


def _torch_mps_memory() -> dict[str, int | None]:
    try:
        import torch
    except ImportError:
        return {
            "current_allocated_bytes": None,
            "driver_allocated_bytes": None,
            "recommended_max_bytes": None,
        }
    if not torch.backends.mps.is_available():
        return {
            "current_allocated_bytes": None,
            "driver_allocated_bytes": None,
            "recommended_max_bytes": None,
        }
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def _parse_backends(raw: str) -> list[str]:
    backends = [backend.strip() for backend in raw.split(",") if backend.strip()]
    allowed = {"wan-vae", "taehv"}
    unknown = sorted(set(backends) - allowed)
    if unknown:
        raise ValueError(f"Unsupported decode backends: {unknown}")
    return backends


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark decode backends on saved Wan/FastWan latents.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--latents-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("video_samples/mlx_decode_benchmark"))
    parser.add_argument("--backends", default="wan-vae,taehv")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true")
    args = parser.parse_args()

    latents = np.load(args.latents_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for backend in _parse_backends(args.backends):
        print(f"=== Decode backend: {backend} ===")
        before = _torch_mps_memory()
        start = time.perf_counter()
        output_path = args.output_dir / f"{args.latents_path.stem}_{backend}.mp4"
        decode_latents_to_video(
            model_root=args.model_root,
            latents_np=latents,
            output_path=output_path,
            fps=args.fps,
            device_arg=args.torch_device,
            dtype_arg=args.torch_dtype,
            backend=backend,
            taehv_source_path=args.taehv_source_path,
            taehv_checkpoint_path=args.taehv_checkpoint_path,
            taehv_parallel=args.taehv_parallel,
        )
        elapsed = time.perf_counter() - start
        after = _torch_mps_memory()
        metrics = {
            "backend": backend,
            "latents_path": str(args.latents_path),
            "latents_shape": list(latents.shape),
            "decode_export_s": elapsed,
            "torch_mps_current_before_bytes": before["current_allocated_bytes"],
            "torch_mps_current_after_bytes": after["current_allocated_bytes"],
            "torch_mps_driver_before_bytes": before["driver_allocated_bytes"],
            "torch_mps_driver_after_bytes": after["driver_allocated_bytes"],
            "torch_mps_recommended_max_bytes": after["recommended_max_bytes"],
            "output_path": str(output_path),
        }
        rows.append(metrics)
        print(json.dumps(metrics, indent=2))

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote decode metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
