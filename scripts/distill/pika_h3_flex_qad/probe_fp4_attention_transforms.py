# SPDX-License-Identifier: Apache-2.0
"""Per-layer FP4 attention error of exact Q/K transforms on real Pika H3 Flex activations.

Runs one single-GPU Pika H3 Flex generation (``basic_pika_h3_flex.py``, in-process) with every dense
``FlashAttentionImpl`` call wrapped: the BF16 output is returned unchanged, and next to it the deployed FA4 NVFP4
attention (Q/K per-16 NVFP4, P/V BF16) is evaluated on transformed inputs. The transforms leave Q K^T unchanged in
exact arithmetic:

* ``smooth_k``: subtract each head's token-mean of K (softmax is shift-invariant per query row);
* ``hadamard``: rotate Q and K by the same orthonormal head-dim Hadamard matrix (spreads outlier channels).

Reports the relative L2 distance to the BF16 output per variant, per sampler forward and per layer. Needs a Blackwell
GPU with FA4 NVFP4 (``FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1``). Every other argument goes to the
example unchanged:

    python scripts/distill/pika_h3_flex_qad/probe_fp4_attention_transforms.py --out probe.json -- \\
        --model-path <flex release> --task fl2va --image first.png --last-image last.png --prompt "..." \\
        --resolution 480p --duration 5
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any

import torch

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "inference" / "basic" / "basic_pika_h3_flex.py"
# name -> (smooth K, Hadamard-rotate Q and K)
VARIANTS = {"plain": (False, False), "smooth_k": (True, False), "hadamard": (False, True), "smooth_k+hadamard": (True, True)}


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    if n & (n - 1):
        raise ValueError(f"Hadamard rotation needs a power-of-two head dim, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device)


def transform(q: torch.Tensor, k: torch.Tensor, smooth: bool, rotation: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    """``[B, L, H, D]`` Q/K -> transformed BF16 Q/K with the same Q K^T up to a per-query-row shift."""
    qf, kf = q.float(), k.float()
    if smooth:
        kf = kf - kf.mean(dim=1, keepdim=True)
    if rotation is not None:
        qf, kf = qf @ rotation, kf @ rotation
    return qf.to(q.dtype), kf.to(k.dtype)


def fa4_nvfp4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    from fastvideo.attention.backends import flash_attn as fa

    q4, q_sf = fa._nvfp4_quantize_for_fa4(q)
    k4, k_sf = fa._nvfp4_quantize_for_fa4(k)
    out = fa.flash_attn_fp4_func(q4[:, :q.shape[1]], k4[:, :k.shape[1]], v, q_sf, k_sf, softmax_scale=scale, causal=False)
    return out[0] if isinstance(out, tuple) else out


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


class Probe:

    def __init__(self) -> None:
        self.layer_of: dict[int, int] = {}
        self.calls: collections.Counter[int] = collections.Counter()
        self.records: list[dict[str, Any]] = []
        self._rotations: dict[tuple[int, str], torch.Tensor] = {}

    def install(self) -> None:
        from fastvideo.attention.backends.flash_attn import FlashAttentionImpl

        original = FlashAttentionImpl.forward
        probe = self

        def forward(impl: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, metadata: Any) -> Any:
            out = original(impl, query, key, value, metadata)
            if getattr(metadata, "packed_sequence", None) is None and query.dim() == 4 and query.shape[-1] % 16 == 0:
                with torch.no_grad():
                    probe.measure(impl, query, key, value, out)
            return out

        FlashAttentionImpl.forward = forward

    def measure(self, impl: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
        layer = self.layer_of.setdefault(id(impl), len(self.layer_of))
        call = self.calls[id(impl)]
        self.calls[id(impl)] += 1
        key = (q.shape[-1], str(q.device))
        if key not in self._rotations:
            self._rotations[key] = hadamard(q.shape[-1], q.device)
        q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
        record: dict[str, Any] = {"layer": layer, "forward": call, "tokens": int(q.shape[1])}
        for name, (smooth, rotate) in VARIANTS.items():
            tq, tk = transform(q, k, smooth, self._rotations[key] if rotate else None)
            record[name] = rel_l2(fa4_nvfp4(tq, tk, v, impl.softmax_scale), out)
        self.records.append(record)

    def summary(self) -> dict[str, Any]:

        def mean(rows: list[dict[str, Any]], name: str) -> float:
            return sum(r[name] for r in rows) / max(len(rows), 1)

        by_forward = collections.defaultdict(list)
        by_layer = collections.defaultdict(list)
        for r in self.records:
            by_forward[r["forward"]].append(r)
            by_layer[r["layer"]].append(r)
        overall = {name: mean(self.records, name) for name in VARIANTS}
        best = min(overall, key=overall.get)
        layers = sorted(((layer, {n: mean(rows, n) for n in VARIANTS}) for layer, rows in by_layer.items()),
                        key=lambda item: -item[1][best])
        return {
            "calls": len(self.records),
            "layers": len(by_layer),
            "overall": overall,
            "best": best,
            "per_forward": {f: {n: mean(rows, n) for n in VARIANTS} for f, rows in sorted(by_forward.items())},
            "worst_layers_under_best": layers[:8],
            "per_layer": dict(sorted(layers)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSON report path")
    args, example_args = parser.parse_known_args()
    if example_args and example_args[0] == "--":
        example_args = example_args[1:]
    probe = Probe()
    probe.install()
    sys.argv = [str(EXAMPLE), *example_args]
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    report = {"args": example_args, "summary": probe.summary(), "records": probe.records}
    Path(args.out).write_text(json.dumps(report, indent=1))
    s = report["summary"]
    print(json.dumps({k: s[k] for k in ("calls", "layers", "overall", "best", "per_forward")}, indent=1))
    print("worst layers under", s["best"], json.dumps(s["worst_layers_under_best"]))


if __name__ == "__main__":
    main()
