#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read /mnt/lustre/vlm-s4duan/gradient_analysis/summary.json and print a compact table
plus a "magnitude ratio" analysis (track/base) at bs=N averaging.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/lustre/vlm-s4duan/gradient_analysis/summary.json")
    with open(path) as f:
        stats = json.load(f)

    layers = [
        "patch_embedding.proj.weight[:,:36]",
        "patch_embedding.proj.weight[:,36:]",
        "track_encoder.temporal_conv.weight",
        "track_encoder.proj.weight",
        "blocks.0.to_q.weight",
        "blocks.14.ffn.fc_in.weight",
        "blocks.29.to_out.weight",
    ]

    print(f"{'model':<14} {'layer':<44} {'per_sample':>12} {'mean_grad':>12} {'snr':>8}")
    print("-" * 96)
    for model, s in stats.items():
        for lay in layers:
            if lay not in s:
                continue
            v = s[lay]
            print(f"{model:<14} {lay:<44} {v['avg_per_sample_norm']:>12.4e} "
                  f"{v['mean_grad_norm']:>12.4e} {v['snr']:>8.4f}")
        print("-" * 96)

    # Magnitude ratio: how big is bs=N averaged grad on the track slice vs base slice?
    print("\nMagnitude ratio (mean_grad_norm(track slice) / mean_grad_norm(base slice))")
    print(f"{'model':<14} {'track_slice / base_slice (patch_embed)':>44}")
    for model, s in stats.items():
        try:
            base = s["patch_embedding.proj.weight[:,:36]"]["mean_grad_norm"]
            trk = s["patch_embedding.proj.weight[:,36:]"]["mean_grad_norm"]
            ratio = trk / max(base, 1e-30)
        except KeyError:
            ratio = float("nan")
        print(f"{model:<14} {ratio:>44.6f}")


if __name__ == "__main__":
    main()
