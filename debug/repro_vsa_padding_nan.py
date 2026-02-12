#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal repro for VSA Triton backward NaNs on padded (partial) blocks.

This targets the Triton path of the FastVideo VSA wrapper, even on H100:
  FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1

Run:
  python debug/repro_vsa_padding_nan.py

Notes:
  - Sequence length must be divisible by 64 (VSA block has 64 tokens).
  - We emulate padding by setting `variable_block_sizes[-1] < 64` and zeroing
    out the invalid tokens in the last block.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--num-blocks", type=int, default=8,
                   help="Number of 64-token blocks (seq_len=64*num_blocks)")
    p.add_argument("--partial-block-size", type=int, default=48,
                   help="Valid tokens in last block (<64 triggers padding)")
    p.add_argument(
        "--raw-latent-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("T", "H", "W"),
        help=(
            "If set, build padded blocks from a raw latent shape (T,H,W) using "
            "patch_size and tile_size. Example close to issue: "
            "--raw-latent-shape 16 44 64 (704x1024 -> 44x64 latents)."
        ),
    )
    p.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(1, 2, 2),
        metavar=("PT", "PH", "PW"),
        help="Patch size used by DiT (default: 1 2 2).",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        nargs=3,
        default=(4, 4, 4),
        metavar=("TT", "TH", "TW"),
        help="VSA tile size (default: 4 4 4).",
    )
    p.add_argument("--dtype",
                   type=str,
                   default="bf16",
                   choices=["bf16", "fp16"],
                   help="Computation dtype for q/k/v")
    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 30.0, 100.0, 300.0],
        help="Try multiple input scales to provoke NaNs",
    )
    p.add_argument(
        "--attend-last-only",
        action="store_true",
        help="If set, each Q block attends only the last KV block.",
    )
    p.add_argument(
        "--use-video-sparse-attn",
        action="store_true",
        help="Use fastvideo_kernel.video_sparse_attn wrapper instead of block_sparse_attn.",
    )
    p.add_argument("--topk", type=int, default=1,
                   help="Top-k blocks for video_sparse_attn (ignored otherwise)")
    return p.parse_args()


def _to_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    raise ValueError(dtype_str)


def _zero_invalid_tokens(x: torch.Tensor, valid_tokens_last_block: int) -> None:
    # x: [B, H, T, D], last block spans [T-64, T)
    if valid_tokens_last_block >= 64:
        return
    start = x.shape[2] - 64 + valid_tokens_last_block
    x[:, :, start:, :].zero_()


def _make_block_map(
    batch: int,
    heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    attend_last_only: bool,
    device: torch.device,
) -> torch.Tensor:
    block_map = torch.zeros(
        (batch, heads, num_q_blocks, num_kv_blocks),
        dtype=torch.bool,
        device=device,
    )
    if attend_last_only:
        block_map[:, :, :, -1] = True
    else:
        # attend to all blocks (maximal stress on padding paths)
        block_map[:] = True
    return block_map


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    Mirrors `fastvideo.attention.backends.video_sparse_attn.construct_variable_block_sizes`
    without relying on lru_cache/device hashing.

    Returns:
      - variable_block_sizes: [num_tiles_total] int32
      - num_tiles: (n_t, n_h, n_w)
    """
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t = _ceil_div(t, ts_t)
    n_h = _ceil_div(h, ts_h)
    n_w = _ceil_div(w, ts_w)

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.int32, device=device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    block_sizes = (t_sizes[:, None, None] * h_sizes[None, :, None] *
                   w_sizes[None, None, :]).reshape(-1)
    return block_sizes, (n_t, n_h, n_w)


def _build_qkv_from_block_sizes(
    batch: int,
    heads: int,
    num_blocks: int,
    block_elems: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    block_sizes: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build q/k/v as [B,H,seq_len,D] where each block has `block_sizes[b]`
    valid tokens (stored in the first positions), and the remainder is zero.
    """
    # mask: [num_blocks, block_elems] bool
    ar = torch.arange(block_elems, device=device, dtype=torch.int32)[None, :]
    mask = ar < block_sizes[:, None]
    mask = mask.view(1, 1, num_blocks, block_elems, 1)

    q = torch.randn((batch, heads, num_blocks, block_elems, dim),
                    device=device,
                    dtype=dtype)
    k = torch.randn((batch, heads, num_blocks, block_elems, dim),
                    device=device,
                    dtype=dtype)
    v = torch.randn((batch, heads, num_blocks, block_elems, dim),
                    device=device,
                    dtype=dtype)
    with torch.no_grad():
        q.mul_(scale)
        k.mul_(scale)
        v.mul_(scale)
        q.mul_(mask)
        k.mul_(mask)
        v.mul_(mask)

    seq_len = num_blocks * block_elems
    q = q.view(batch, heads, seq_len, dim).requires_grad_(True)
    k = k.view(batch, heads, seq_len, dim).requires_grad_(True)
    v = v.view(batch, heads, seq_len, dim).requires_grad_(True)
    return q, k, v


def _summarize_tensor(x: torch.Tensor, name: str) -> str:
    x_f = x.float()
    return (
        f"{name}: dtype={x.dtype}, shape={tuple(x.shape)}, "
        f"min={x_f.min().item():.3e}, max={x_f.max().item():.3e}, "
        f"nan={torch.isnan(x_f).any().item()}, inf={torch.isinf(x_f).any().item()}"
    )


def _any_nan(xs: Iterable[torch.Tensor]) -> bool:
    return any(torch.isnan(x).any().item() for x in xs)


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this repro.")

    # Force wrapper to use Triton even on H100/SM90.
    os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

    device = torch.device("cuda")
    dtype = _to_dtype(args.dtype)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"FASTVIDEO_KERNEL_VSA_FORCE_TRITON={os.environ.get('FASTVIDEO_KERNEL_VSA_FORCE_TRITON')}")
    print(
        f"batch={args.batch}, heads={args.heads}, head_dim={args.head_dim}, "
        f"num_blocks={args.num_blocks}, partial_block_size={args.partial_block_size}, dtype={args.dtype}"
    )

    tile_size = tuple(int(x) for x in args.tile_size)
    patch_size = tuple(int(x) for x in args.patch_size)
    block_elems = tile_size[0] * tile_size[1] * tile_size[2]

    if args.raw_latent_shape is not None:
        raw_latent_shape = tuple(int(x) for x in args.raw_latent_shape)
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        variable_block_sizes, num_tiles = _construct_variable_block_sizes(
            dit_seq_shape=dit_seq_shape,
            tile_size=tile_size,
            device=device,
        )
        num_blocks = int(variable_block_sizes.numel())
        seq_len = num_blocks * block_elems
        print(
            "shape_mode:",
            {
                "raw_latent_shape": raw_latent_shape,
                "patch_size": patch_size,
                "dit_seq_shape": dit_seq_shape,
                "tile_size": tile_size,
                "num_tiles": num_tiles,
                "block_elems": block_elems,
                "num_blocks": num_blocks,
                "seq_len": seq_len,
                "min_block": int(variable_block_sizes.min().item()),
                "max_block": int(variable_block_sizes.max().item()),
            },
        )
    else:
        # Legacy mode: 1D block sequence with only the last block being partial.
        # Keep legacy tile size stable even if user passes --tile-size.
        tile_size = (4, 4, 4)
        block_elems = 64
        seq_len = block_elems * args.num_blocks
        num_blocks = args.num_blocks

    if not (0 < args.partial_block_size <= block_elems):
        raise ValueError("partial_block_size must be in (0, 64].")

    if args.raw_latent_shape is None:
        variable_block_sizes = torch.full(
            (num_blocks,),
            block_elems,
            dtype=torch.int32,
            device=device,
        )
        if args.partial_block_size < block_elems:
            variable_block_sizes[-1] = int(args.partial_block_size)

    # Build a block map that exercises the partial block.
    block_map = _make_block_map(
        batch=args.batch,
        heads=args.heads,
        num_q_blocks=num_blocks,
        num_kv_blocks=num_blocks,
        attend_last_only=args.attend_last_only,
        device=device,
    )

    # Enable anomaly detection to match the issue report.
    torch.autograd.set_detect_anomaly(True)

    # Print dispatch diagnostics (whether we *would* use SM90 ops and whether forcing is on).
    import fastvideo_kernel.block_sparse_attn as vsa_mod
    sm90_fwd, sm90_bwd = vsa_mod._get_sm90_ops()
    print(
        "dispatch_debug:",
        {
            "is_sm90": vsa_mod._is_sm90(),
            "force_triton": vsa_mod._force_triton(),
            "sm90_fwd_available": sm90_fwd is not None,
            "sm90_bwd_available": sm90_bwd is not None,
        },
    )

    if args.use_video_sparse_attn:
        from fastvideo_kernel import video_sparse_attn
    else:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

    # Try multiple input scales; some numerical issues only show up at larger magnitudes.
    for scale in args.scales:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if args.raw_latent_shape is not None:
            q, k, v = _build_qkv_from_block_sizes(
                batch=args.batch,
                heads=args.heads,
                num_blocks=num_blocks,
                block_elems=block_elems,
                dim=args.head_dim,
                dtype=dtype,
                device=device,
                block_sizes=variable_block_sizes,
                scale=scale,
            )
        else:
            # Build inputs, apply "padding" zeros, then enable grads.
            # This avoids in-place writes on leaf tensors that already require grad.
            q = torch.randn(args.batch,
                            args.heads,
                            seq_len,
                            args.head_dim,
                            device=device,
                            dtype=dtype)
            k = torch.randn(args.batch,
                            args.heads,
                            seq_len,
                            args.head_dim,
                            device=device,
                            dtype=dtype)
            v = torch.randn(args.batch,
                            args.heads,
                            seq_len,
                            args.head_dim,
                            device=device,
                            dtype=dtype)

            with torch.no_grad():
                q.mul_(scale)
                k.mul_(scale)
                v.mul_(scale)
                _zero_invalid_tokens(q, args.partial_block_size)
                _zero_invalid_tokens(k, args.partial_block_size)
                _zero_invalid_tokens(v, args.partial_block_size)

            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

        print("\n" + "-" * 80)
        print(f"scale={scale}")
        print(_summarize_tensor(variable_block_sizes, "variable_block_sizes"))

        try:
            if args.use_video_sparse_attn:
                # video_sparse_attn expects full-length q/k/v and uses block_map internally
                # only for the sparse branch. We pass block_size that matches VSA tiling.
                o = video_sparse_attn(
                    q,
                    k,
                    v,
                    variable_block_sizes,
                    variable_block_sizes,
                    topk=args.topk,
                    block_size=tile_size if args.raw_latent_shape is not None else (4, 4, 4),
                    compress_attn_weight=None,
                )
            else:
                o, _aux = block_sparse_attn(q, k, v, block_map,
                                            variable_block_sizes)

            # Simple scalar loss
            loss = (o.float()**2).mean()

            dq, dk, dv = torch.autograd.grad(loss, (q, k, v), retain_graph=False)

            has_nan = _any_nan((dq, dk, dv))
            print(_summarize_tensor(o, "out"))
            print(_summarize_tensor(dq, "dq"))
            print(_summarize_tensor(dk, "dk"))
            print(_summarize_tensor(dv, "dv"))
            if has_nan:
                raise RuntimeError("Detected NaN in gradients.")
            print("OK: grads are finite")
        except Exception as e:
            print("FAILED:", repr(e))
            raise


if __name__ == "__main__":
    main()

