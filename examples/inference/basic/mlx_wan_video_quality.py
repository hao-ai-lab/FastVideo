"""Compare generated MP4s against a reference MP4 with simple pixel metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_video(path: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int | list[int]]:
    if candidate.shape != reference.shape:
        raise ValueError(f"Shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    candidate_f = candidate.astype(np.float32)
    reference_f = reference.astype(np.float32)
    diff = candidate_f - reference_f
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else float("inf")
    return {
        "frames": int(candidate.shape[0]),
        "height": int(candidate.shape[1]),
        "width": int(candidate.shape[2]),
        "channels": int(candidate.shape[3]),
        "mse_vs_reference": mse,
        "mae_vs_reference": mae,
        "max_abs_vs_reference": max_abs,
        "psnr_db_vs_reference": psnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MP4s against a reference MP4.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    args = parser.parse_args()

    reference = _read_video(args.reference)
    rows = []
    for candidate_path in args.candidates:
        candidate = _read_video(candidate_path)
        row = {
            "reference_path": str(args.reference),
            "candidate_path": str(candidate_path),
            **_metrics(candidate, reference),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote video quality metrics to: {args.metrics_json}")


if __name__ == "__main__":
    main()
