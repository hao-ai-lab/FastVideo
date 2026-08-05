"""Probe a first Apple-native MLX runtime path for FastWan-like shapes.

This is not a full FastWan MLX port yet. It is the smallest useful first step:

1. verify MLX sees the Apple GPU,
2. derive the latent/token shape for a FastWan video request,
3. benchmark the MLX primitives the DiT runtime will need first:
   dense scaled-dot-product attention and a large hidden-size linear layer.

The default shape matches the current Mac POC target:
256x448, 33 frames, Wan patch size (1, 2, 2), hidden size 5120,
40 heads, 128 head dim.
"""

from __future__ import annotations

import argparse
from fastvideo.mlx_runtime.fastwan import (
    benchmark_mlx_attention,
    benchmark_mlx_linear,
    benchmark_torch_mps_attention,
    fastwan_shape,
    fastwan_shape_from_config,
    replace_tokens,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX primitive probe for FastWan-like Apple Silicon runtime work")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional transformer config.json. When set, heads/head_dim/patch size are read from the real model.",
    )
    parser.add_argument("--tokens", type=int, default=None, help="Override derived token count for smaller/larger probes.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--compare-torch-mps", action="store_true")
    args = parser.parse_args()

    import importlib.metadata

    import mlx.core as mx

    if args.config_path:
        shape = fastwan_shape_from_config(args.config_path, height=args.height, width=args.width, num_frames=args.num_frames)
    else:
        shape = fastwan_shape(height=args.height, width=args.width, num_frames=args.num_frames)
    if args.tokens is not None:
        shape = replace_tokens(shape, args.tokens)

    print(f"MLX version: {importlib.metadata.version('mlx')}")
    print(f"MLX default device: {mx.default_device()}")
    print(
        "FastWan-like shape: "
        f"{shape.height}x{shape.width}x{shape.num_frames} -> "
        f"latent {shape.latent_frames}x{shape.latent_height}x{shape.latent_width}, "
        f"tokens={shape.tokens}, hidden={shape.hidden_size}, heads={shape.num_heads}, head_dim={shape.head_dim}"
    )

    attention_ms = benchmark_mlx_attention(shape, args.warmup, args.iters)
    linear_ms = benchmark_mlx_linear(shape, args.warmup, args.iters)
    print(f"MLX SDPA median: {attention_ms:.2f} ms")
    print(f"MLX hidden linear median: {linear_ms:.2f} ms")

    if args.compare_torch_mps:
        torch_attention_ms = benchmark_torch_mps_attention(shape, args.warmup, args.iters)
        if torch_attention_ms is None:
            print("Torch MPS SDPA median: unavailable")
        else:
            print(f"Torch MPS SDPA median: {torch_attention_ms:.2f} ms")


if __name__ == "__main__":
    main()
