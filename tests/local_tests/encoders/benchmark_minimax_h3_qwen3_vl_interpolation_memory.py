# SPDX-License-Identifier: Apache-2.0
"""Measure the Qwen3-VL position-interpolation CUDA working set."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _parse_grid(value: str) -> tuple[int, int, int]:
    try:
        grid = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid must contain integer T,H,W values") from error
    if len(grid) != 3 or any(dimension <= 0 for dimension in grid):
        raise argparse.ArgumentTypeError("grid must contain three positive T,H,W values")
    return grid


def _revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _is_dirty(source_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(source_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _run_interpolation(
    module: Any,
    position_embedding: torch.Tensor,
    grid_thw: torch.Tensor,
    side: int,
    merge: int,
) -> tuple[torch.Tensor, str]:
    helper = getattr(module, "_interpolate_vision_position_embeddings", None)
    if helper is not None:
        return helper(position_embedding, grid_thw, side, merge), "standalone_helper"

    class VisionProxy:
        num_grid_per_side = side
        spatial_merge_size = merge
        pos_embed = torch.nn.Embedding.from_pretrained(position_embedding, freeze=True)

    vision_class = module.MiniMaxH3Qwen3VLVisionModel
    return vision_class._interpolate_position_embeddings(VisionProxy(), grid_thw), "legacy_model_method"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--grid", action="append", type=_parse_grid, default=None, metavar="T,H,W")
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--num-grid-per-side", type=int, default=48)
    parser.add_argument("--spatial-merge-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1737)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the interpolation memory benchmark")
    if args.hidden_size <= 0 or args.num_grid_per_side <= 0 or args.spatial_merge_size <= 0:
        raise ValueError("hidden size, grid side, and merge size must be positive")

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("fastvideo.models.encoders.minimax_h3_qwen3_vl")

    device = torch.device("cuda")
    grids = args.grid or [(15, 42, 74)]
    grid_thw = torch.tensor(grids, dtype=torch.long, device=device)
    torch.manual_seed(args.seed)
    position_embedding = torch.randn(
        args.num_grid_per_side**2,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output, implementation = _run_interpolation(
            module,
            position_embedding,
            grid_thw,
            args.num_grid_per_side,
            args.spatial_merge_size,
        )
    torch.cuda.synchronize()

    properties = torch.cuda.get_device_properties(device)
    record = {
        "source_root": str(source_root),
        "source_revision": _revision(source_root),
        "source_is_dirty": _is_dirty(source_root),
        "implementation": implementation,
        "torch_version": torch.__version__,
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "grid_thw": grids,
        "position_embedding_shape": list(position_embedding.shape),
        "position_embedding_dtype": str(position_embedding.dtype),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_sum_fp32": output.float().sum().item(),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline_allocated,
        "incremental_peak_reserved_bytes": torch.cuda.max_memory_reserved() - baseline_reserved,
    }
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
