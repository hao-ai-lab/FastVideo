#!/usr/bin/env python3
"""
Benchmark the SM100 fused VSA64 tile layout against the legacy scatter/transpose route.

Times Wan VSA inference preprocessing through the backend
(``VideoSparseAttentionImpl.preprocess_qkv``), toggling
``FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT``:
  - legacy: tile scatter into BSHD + four BSHD->BHSD transposes
  - fused:  one Triton kernel writing tiled BHSD directly

With ``--include_attn`` the timed region also covers ``forward`` (the
``video_sparse_attn`` call). Outputs of both routes are checked for bitwise
equality before timing. The fused route only activates on SM100.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (for triton.testing.do_bench).") from e

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import (VideoSparseAttentionImpl,
                                                            VideoSparseAttentionMetadataBuilder)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the SM100 fused VSA64 tile layout")
    p.add_argument("--dit_shape",
                   type=int,
                   nargs=3,
                   default=[21, 30, 52],
                   metavar=("T", "H", "W"),
                   help="Post-patchify token grid (default: Wan 480p, 81 frames)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=40, help="Heads per rank")
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--include_attn", action="store_true", help="Also time the video_sparse_attn forward")
    p.add_argument("--warmup", type=int, default=200, help="Warmup time in ms (triton do_bench)")
    p.add_argument("--rep", type=int, default=2000, help="Measurement time in ms (triton do_bench)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_step(impl: VideoSparseAttentionImpl, qkvg: torch.Tensor, builder: VideoSparseAttentionMetadataBuilder,
              build_kwargs: dict, fused: bool, include_attn: bool) -> Callable[[], torch.Tensor]:
    metadata = builder.build(**build_kwargs)

    def step() -> torch.Tensor:
        # The routing gate reads the switch on every call.
        with envs.FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT.override(not fused):
            tiled = impl.preprocess_qkv(qkvg, metadata)
        q, k, v, gate = tiled.chunk(4)
        if include_attn:
            return impl.forward(q, k, v, gate, metadata)
        if metadata.fused_layout_active:
            return tiled
        # Legacy route: the transposes forward() applies before video_sparse_attn.
        return torch.cat([t.transpose(1, 2).contiguous() for t in (q, k, v, gate)])

    with torch.no_grad():
        step()
    if fused and not metadata.fused_layout_active:
        raise RuntimeError("Fused layout did not activate (requires SM100, bf16, head_dim 128, "
                           "and fastvideo_kernel.triton_kernels.vsa_tile_layout).")
    return step


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    impl = object.__new__(VideoSparseAttentionImpl)
    builder = VideoSparseAttentionMetadataBuilder()
    build_kwargs = dict(current_timestep=0,
                        raw_latent_shape=tuple(args.dit_shape),
                        patch_size=(1, 1, 1),
                        VSA_sparsity=args.sparsity,
                        device=torch.device("cuda"),
                        cache_tile_buf=True)
    sequence = int(np.prod(args.dit_shape))
    qkvg = torch.randn((4 * args.batch_size, sequence, args.num_heads, 128), device="cuda", dtype=torch.bfloat16)

    legacy = make_step(impl, qkvg, builder, build_kwargs, fused=False, include_attn=args.include_attn)
    fused = make_step(impl, qkvg, builder, build_kwargs, fused=True, include_attn=args.include_attn)
    with torch.no_grad():
        if not torch.equal(legacy(), fused()):
            raise RuntimeError("Fused and legacy outputs differ.")
        legacy_ms = do_bench(legacy, warmup=args.warmup, rep=args.rep, return_mode="median")
        fused_ms = do_bench(fused, warmup=args.warmup, rep=args.rep, return_mode="median")

    region = "layout+attn" if args.include_attn else "layout"
    print(f"GPU: {torch.cuda.get_device_name()} | dit_shape={tuple(args.dit_shape)} seq={sequence} "
          f"B={args.batch_size} H={args.num_heads} D=128 sparsity={args.sparsity}")
    print(f"{region} (median): legacy {legacy_ms:.3f} ms | fused {fused_ms:.3f} ms | "
          f"speedup {legacy_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    main()
