# SPDX-License-Identifier: Apache-2.0
"""Report the WanTrack "gate" — patch_embedding.weight[:, 36:] — from a DCP checkpoint.

The overfit stages (A/B) exist to grow this slot from EXACTLY ZERO up to roughly the scale the
successful 1.3B merged init reached (std ~0.011), co-adapting it with track_encoder. If it is
still ~0 there is no bootstrap happening and step C would merge nothing; if it explodes past the
pretrained channel scale (~0.0124) the track pathway is overwhelming the image prior.

Loads ONLY the tensors of interest (DCP resharding is per-tensor), so it costs ~MBs, not the
~200GB a full checkpoint load would.

Usage: python data_pipeline/check_gate_growth.py <checkpoint-dir> [more dirs...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

BASE_IN = 36
# NOTE: the live module nests the conv, so a TRAINED checkpoint stores
# 'patch_embedding.proj.weight', while the converter writes 'patch_embedding.weight' into the
# init safetensors. Try both — silently matching neither is how the earlier LWLR experiment
# ended up measuring nothing.
PE_KEYS = ["patch_embedding.proj.weight", "patch_embedding.weight"]
KEYS = PE_KEYS + [
    "track_encoder.proj.weight",
    "track_encoder.temporal_conv.weight",
]


def _find(md_keys, want: str) -> str | None:
    """DCP nests model tensors under a role prefix (e.g. 'student.'); match on the suffix."""
    exact = [k for k in md_keys if k == want]
    if exact:
        return exact[0]
    cands = [k for k in md_keys if k.endswith(want)]
    return cands[0] if cands else None


def report(ckpt: Path) -> None:
    dcp_dir = ckpt / "dcp" if (ckpt / "dcp").is_dir() else ckpt
    reader = FileSystemReader(str(dcp_dir))
    md = reader.read_metadata()
    planned = md.state_dict_metadata
    sd = {}
    resolved = {}
    for want in KEYS:
        k = _find(list(planned.keys()), want)
        if k is None:
            continue
        meta = planned[k]
        sd[k] = torch.empty(tuple(meta.size), dtype=meta.properties.dtype)
        resolved[want] = k
    if not sd:
        print(f"{ckpt.name}: none of {KEYS} found in checkpoint metadata")
        return
    dcp.load(sd, checkpoint_id=str(dcp_dir))

    out = [f"{ckpt.name:>18s}"]
    pe_k = next((resolved[k] for k in PE_KEYS if k in resolved), None)
    if pe_k is None:
        out.append("!! patch_embedding NOT FOUND — cannot report gate")
    else:
        pe = sd[pe_k].float()
        out.append(f"gate[:, {BASE_IN}:] std={pe[:, BASE_IN:].std():.6f}")
        out.append(f"pretrained std={pe[:, :BASE_IN].std():.6f}")
    for w in ("track_encoder.proj.weight", "track_encoder.temporal_conv.weight"):
        if w in resolved:
            out.append(f"{w.split('.')[1]}_norm={sd[resolved[w]].float().norm():.4f}")
    print("  ".join(out))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for key, default in [("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                         ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29555")]:
        os.environ.setdefault(key, default)
    print(f"reference: 1.3B merged-from-overfit gate std = 0.011418 (grown from 0.000000)")
    for p in sys.argv[1:]:
        try:
            report(Path(p))
        except Exception as e:  # keep going across checkpoints
            print(f"{Path(p).name}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
