# SPDX-License-Identifier: Apache-2.0
"""Rank-reduce the MiniMax-H3 AdaLN modulation projections.

Every AdaLN projection consumes ``silu(time_embedder(time_proj(t)))`` where ``t``
is a scalar timestep, so its reachable input set is a smooth 1-D curve. Sampling
that curve and truncating its SVD yields a shared basis ``V`` such that

    W @ u(t)  ==  (W @ V) @ (V.T @ u(t))

to within the truncation error. Pre-multiplying every block's ``[96768, 2688]``
projection by ``V`` leaves ``[96768, r]``, removing ~39% of the model's
parameters with no change to the forward's arithmetic structure.

Unlike ComfyUI's "pruned" checkpoints this keeps ``time_embedder`` and stores the
basis as a real projection instead of replacing the timestep MLP with a 1025-point
lookup table. That costs 14.5M parameters (0.03 GB of the ~26 GB saved) and buys
exactness at arbitrary ``t`` -- no grid interpolation and no clamped domain, which
matters if the schedule is ever changed (different flow shift, distilled sigmas).

Writes a diffusers-format component directory that ``TransformerLoader`` picks up
directly: ``adaln_rank`` in ``config.json`` flows onto the arch config through
``update_model_arch``. No flag is needed at inference -- a checkpoint carrying
``adaln_rank`` selects the factorized path, one without it stays full rank.

Usage::

    python tools/minimax_h3/fit_adaln_basis.py \
        --src  /path/to/MiniMax-H3/transformer \
        --dst  /path/to/MiniMax-H3-r16/transformer --rank 16

    # then assemble a model dir whose other components point at the original
    ln -s /path/to/MiniMax-H3/{text_encoder,tokenizer,processor,vae,audio_vae,\
scheduler,audio_scheduler,modular_model_index.json} /path/to/MiniMax-H3-r16/

    python examples/inference/basic/basic_minimax_h3_t2v.py \
        --model-path /path/to/MiniMax-H3-r16 --prompt "..."

Use ``--report-only`` to measure the fit error at a candidate rank without
writing a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
ADALN_SUFFIX = "adaln_proj.linear.weight"
NORM_OUT_WEIGHT = "norm_out.linear.weight"
TIME_EMBEDDER_KEYS = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Official transformer component dir.")
    p.add_argument("--dst", required=True, help="Output component dir.")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grid", type=int, default=4096, help="Timestep samples used to fit the basis.")
    p.add_argument("--freq-dim", type=int, default=256)
    p.add_argument("--report-only", action="store_true", help="Fit and report error without writing.")
    return p.parse_args()


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusers get_timestep_embedding, flip_sin_to_cos=True, downscale_freq_shift=0."""
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half
    emb = t[:, None].double() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def load_keys(src: Path, index: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(src / shard), framework="pt") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def fit_basis(src: Path, index: dict[str, str], rank: int, grid: int,
              freq_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (V [time_embed_dim, rank], U [grid, time_embed_dim]) in float64."""
    embedder = load_keys(src, index, list(TIME_EMBEDDER_KEYS))
    # t is the DiT's timestep input: scheduler.timesteps = 1 - sigmas, so t in [0, 1].
    # Condition rows pin t to 0.999 / 1.0, so the endpoint must be included.
    t = torch.linspace(0.0, 1.0, grid, dtype=torch.float64)
    h = timestep_embedding(t, freq_dim)
    h = h @ embedder["time_embedder.linear_1.weight"].double().T + embedder["time_embedder.linear_1.bias"].double()
    h = torch.nn.functional.silu(h)
    temb = h @ embedder["time_embedder.linear_2.weight"].double().T + embedder["time_embedder.linear_2.bias"].double()
    u = torch.nn.functional.silu(temb)
    _, s, vh = torch.linalg.svd(u, full_matrices=False)
    residual = ((s[rank:]**2).sum() / (s**2).sum()).sqrt()
    print(f"basis: U={tuple(u.shape)} rank={rank} relative residual ||U-U_r||/||U|| = {residual:.3e}")
    return vh[:rank].T.contiguous(), u


def main() -> None:
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    index_map = json.loads((src / INDEX_NAME).read_text())["weight_map"]

    basis, u = fit_basis(src, index_map, args.rank, args.grid, args.freq_dim)

    # Worst-case induced error on the actual modulation outputs.
    worst = 0.0
    scale = 0.0
    for key in sorted(k for k in index_map if k.endswith(ADALN_SUFFIX) or k == NORM_OUT_WEIGHT):
        w = load_keys(src, index_map, [key])[key].double()
        ref = u @ w.T
        approx = (u @ basis) @ (w @ basis).T
        worst = max(worst, (ref - approx).abs().max().item())
        scale = max(scale, ref.abs().max().item())
    print(f"modulation error over all projections: max|err|={worst:.3e} "
          f"(|Wu|max={scale:.3f}, relative={worst / scale:.3e})")
    if args.report_only:
        return

    dst.mkdir(parents=True, exist_ok=True)
    new_index: dict[str, str] = {}
    total_before = total_after = 0
    for shard in sorted(set(index_map.values())):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(src / shard), framework="pt") as f:
            shard_keys = list(f.keys())
            for key in shard_keys:
                tensor = f.get_tensor(key)
                total_before += tensor.numel()
                if key.endswith(ADALN_SUFFIX) or key == NORM_OUT_WEIGHT:
                    # [out, time_embed_dim] @ [time_embed_dim, rank] -> [out, rank]
                    tensor = (tensor.double() @ basis).to(torch.float16)
                tensors[key] = tensor
                total_after += tensor.numel()
                new_index[key] = shard
        save_file(tensors, str(dst / shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(tensors)} tensors)")

    # ReplicatedLinear stores [out_features, in_features], so the basis is V.T.
    basis_shard = "diffusion_pytorch_model-adaln-basis.safetensors"
    save_file({"adaln_basis.weight": basis.T.to(torch.float16).contiguous()},
              str(dst / basis_shard),
              metadata={"format": "pt"})
    new_index["adaln_basis.weight"] = basis_shard
    total_after += basis.numel()

    (dst / INDEX_NAME).write_text(json.dumps({"metadata": {}, "weight_map": new_index}, indent=1))

    config = json.loads((src / "config.json").read_text())
    config["adaln_rank"] = args.rank
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for extra in src.glob("*.json"):
        if extra.name not in {INDEX_NAME, "config.json"}:
            shutil.copy2(extra, dst / extra.name)

    print(f"\nparameters: {total_before / 1e9:.3f}B -> {total_after / 1e9:.3f}B "
          f"({100 * (1 - total_after / total_before):.1f}% removed)")
    print(f"bf16 footprint: {total_before * 2 / 1e9:.1f} GB -> ~{total_after * 2 / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
