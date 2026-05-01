# SPDX-License-Identifier: Apache-2.0
"""Verify weights are bit-exact between upstream and FastVideo paths.

If they're not, the per-block parity drift could come from weight
mismatches (conversion-script truncation, bf16-cast-then-load, etc.)
rather than op-ordering. Run before concluding "bf16 noise".
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _find_base_shard_dir() -> Path | None:
    """Return the local path to GAIR/daVinci-MagiHuman/base/ with shards present, or None."""
    override = os.getenv("MAGI_HUMAN_BASE_SHARD_DIR")
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            repo_id="GAIR/daVinci-MagiHuman",
            allow_patterns=[
                "base/*.safetensors",
                "base/model.safetensors.index.json",
            ],
        )
        candidate = Path(snap) / "base"
        if candidate.is_dir() and any(candidate.glob("*.safetensors")):
            return candidate
        return None
    except Exception:
        return None


def main() -> None:
    if not torch.cuda.is_available():
        print("Need CUDA.")
        return

    upstream_src = REPO_ROOT / "daVinci-MagiHuman"
    if not upstream_src.exists():
        print("daVinci-MagiHuman/ missing.")
        return

    base_shard_dir = _find_base_shard_dir()
    if base_shard_dir is None:
        print("GAIR/daVinci-MagiHuman base shards not available locally.")
        return

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_DIFFUSERS_PATH",
        REPO_ROOT / "converted_weights" / "magi_human_base",
    ))
    transformer_dir = converted_dir / "transformer"

    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs, load_upstream_dit,
    )
    install_stubs()

    device = torch.device("cuda:0")

    # Load both, dumping weight tensors to dicts for comparison.
    print("Loading upstream...")
    up = load_upstream_dit(base_shard_dir, device=device, dtype=None)
    up_state = {k: v.detach().cpu() for k, v in up.state_dict().items()}
    del up
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print("Loading FastVideo...")
    from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from safetensors.torch import load_file
    fv = MagiHumanDiT(MagiHumanVideoConfig())
    state = {}
    for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))):
        state.update(load_file(shard))
    fv.load_state_dict(state, strict=False)
    fv_state = {k: v.detach().cpu() for k, v in fv.state_dict().items()}

    # Compare overlapping keys.
    up_keys = set(up_state.keys())
    fv_keys = set(fv_state.keys())
    only_up = up_keys - fv_keys
    only_fv = fv_keys - up_keys
    common = up_keys & fv_keys
    print(f"Keys: common={len(common)}, only_upstream={len(only_up)}, only_fastvideo={len(only_fv)}")
    if only_up:
        print(f"  Only upstream (sample): {sorted(only_up)[:5]}")
    if only_fv:
        print(f"  Only fv (sample): {sorted(only_fv)[:5]}")

    bit_exact = 0
    diff_keys = []
    shape_mismatch = []
    dtype_mismatch = []
    for k in sorted(common):
        a, b = up_state[k], fv_state[k]
        if a.shape != b.shape:
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
            continue
        if a.dtype != b.dtype:
            dtype_mismatch.append((k, a.dtype, b.dtype))
        d = (a.float() - b.float()).abs()
        max_d = d.max().item()
        if max_d == 0.0:
            bit_exact += 1
        else:
            diff_keys.append((k, max_d, d.mean().item(), tuple(a.shape), str(a.dtype)))

    print(f"\nWeight comparison ({len(common)} keys):")
    print(f"  bit-exact: {bit_exact}")
    print(f"  with diff: {len(diff_keys)}")
    print(f"  shape mismatch: {len(shape_mismatch)}")
    print(f"  dtype mismatch: {len(dtype_mismatch)}")

    if dtype_mismatch:
        print("\nDtype mismatches:")
        for k, da, db in dtype_mismatch[:10]:
            print(f"  {k}: up={da} fv={db}")

    if shape_mismatch:
        print("\nShape mismatches:")
        for k, sa, sb in shape_mismatch[:10]:
            print(f"  {k}: up={sa} fv={sb}")

    if diff_keys:
        print("\nTop weight diffs (max-diff sorted):")
        diff_keys.sort(key=lambda x: -x[1])
        for k, max_d, mean_d, shape, dtype in diff_keys[:15]:
            print(f"  {dtype} {str(shape):<40} max={max_d:.6e} mean={mean_d:.6e}  {k}")


if __name__ == "__main__":
    main()
