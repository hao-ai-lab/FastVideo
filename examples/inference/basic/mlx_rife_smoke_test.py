# SPDX-License-Identifier: Apache-2.0
"""Tiny MLX RIFE frame-interpolation smoke test."""

from __future__ import annotations

import argparse
import time

import numpy as np

from fastvideo.mlx_runtime.rife_interp import interpolate, load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX RIFE 4.25 frame interpolation smoke test."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny two-frame interpolation test.",
    )
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("Nothing to do; pass --self-test")

    frame0 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255
    start = time.perf_counter()
    model = load_model()
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    frames = interpolate([frame0, frame1], factor=2, model=model)
    interp_s = time.perf_counter() - start
    assert len(frames) == 3
    assert frames[1].shape == frame0.shape
    assert frames[1].dtype == np.uint8
    print(
        "MLX RIFE self-test passed: "
        f"load_s={load_s:.3f} interp_s={interp_s:.3f} shape={frames[1].shape}"
    )


if __name__ == "__main__":
    main()
