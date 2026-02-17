#!/usr/bin/env python3
"""Extract kernel latency stats from profiler traces and write markdown."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelTarget:
    impl: str
    trace_path: Path
    kernel_name: str


@dataclass
class KernelStats:
    occurrence: int
    min_ms: float
    max_ms: float
    avg_ms: float


def compute_kernel_stats(trace_path: Path, kernel_name: str) -> KernelStats:
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    durations_us: list[float] = []
    for event in payload.get("traceEvents", []):
        if event.get("cat") != "kernel":
            continue
        if event.get("name") != kernel_name:
            continue
        dur = event.get("dur")
        if dur is None:
            continue
        durations_us.append(float(dur))

    if not durations_us:
        raise ValueError(
            f"No kernel events found for kernel in trace: {trace_path}\n"
            f"Kernel: {kernel_name}"
        )

    min_us = min(durations_us)
    max_us = max(durations_us)
    avg_us = sum(durations_us) / len(durations_us)

    return KernelStats(
        occurrence=len(durations_us),
        min_ms=min_us / 1000.0,
        max_ms=max_us / 1000.0,
        avg_ms=avg_us / 1000.0,
    )


def build_markdown(targets: list[KernelTarget], stats: list[KernelStats]) -> str:
    lines = [
        "# Kernel Latency Summary",
        "",
        "Profiler kernel `dur` is typically in microseconds; values below are converted to milliseconds.",
        "",
        "| Attention Impl | Trace File | Kernel Name | Occurrence | Min (ms) | Max (ms) | Avg (ms) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for target, stat in zip(targets, stats):
        lines.append(
            "| "
            f"{target.impl} | "
            f"`{target.trace_path.name}` | "
            f"`{target.kernel_name}` | "
            f"{stat.occurrence} | "
            f"{stat.min_ms:.6f} | "
            f"{stat.max_ms:.6f} | "
            f"{stat.avg_ms:.6f} |"
        )

    lines.append("")
    return "\n".join(lines)


def default_targets() -> list[KernelTarget]:
    return [
        KernelTarget(
            impl="vsa",
            trace_path=Path(
                "/home/hal-jundas/codes/FastVideo-demo/profile-ltx2/"
                "vsa-traces-0.7/hpc-rack-1-10_19420.1771310583080545350.pt.trace.json.gz"
            ),
            kernel_name="_attn_fwd_sparse",
        ),
        KernelTarget(
            impl="flash_attn",
            trace_path=Path(
                "/home/hal-jundas/codes/FastVideo-demo/profile-ltx2/"
                "traces_20260217_061722/hpc-rack-1-10_16077.1771309181463056803.pt.trace.json.gz"
            ),
            kernel_name=(
                "void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, "
                "false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, "
                "cutlass::bfloat16_t> >, false, false, false, false, true, true, false, "
                "false>(flash::Flash_fwd_params)"
            ),
        ),
    ]


def main() -> None:
    out_default = Path(__file__).with_name("kernel_latency_summary.md")

    parser = argparse.ArgumentParser(
        description="Extract kernel latency stats from two trace files into markdown."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=out_default,
        help=f"Output markdown path (default: {out_default})",
    )
    args = parser.parse_args()

    targets = default_targets()
    stats = [compute_kernel_stats(t.trace_path, t.kernel_name) for t in targets]
    markdown = build_markdown(targets, stats)

    args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
