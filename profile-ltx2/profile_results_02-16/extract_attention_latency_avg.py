#!/usr/bin/env python3
"""Extract average attention latency from VSA and FA logs.

Notes:
- Log lines are labeled "ms", but the user indicates these values are actually seconds.
- This script reports averages in seconds and also shows milliseconds for convenience.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@dataclass
class SeriesStats:
    total_count: int
    selected_count: int
    avg_selected_s: float
    min_selected_s: float
    max_selected_s: float


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def extract_values(path: Path, tag: str) -> list[float]:
    # Example line: >>>> VSA time: 0.009293003007769585 ms
    pat = re.compile(rf">>>>\s+{re.escape(tag)}\s+time:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = strip_ansi(raw)
            m = pat.search(line)
            if m:
                values.append(float(m.group(1)))
    if not values:
        raise ValueError(f"No '{tag} time' values found in {path}")
    return values


def summarize(values: list[float], min_s: float, max_s: float) -> SeriesStats:
    selected = [v for v in values if min_s <= v <= max_s]
    if not selected:
        raise ValueError(
            f"No values left after range filter [{min_s}, {max_s}] on {len(values)} samples"
        )
    return SeriesStats(
        total_count=len(values),
        selected_count=len(selected),
        avg_selected_s=sum(selected) / len(selected),
        min_selected_s=min(selected),
        max_selected_s=max(selected),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract average VSA/FA latency (seconds) from two log files."
    )
    parser.add_argument(
        "--vsa-log",
        type=Path,
        default=Path("/home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile.4.vsa.compile.log"),
    )
    parser.add_argument(
        "--fa-log",
        type=Path,
        default=Path("/home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile.4.compile.log"),
    )
    parser.add_argument("--vsa-min-s", type=float, default=0.005)
    parser.add_argument("--vsa-max-s", type=float, default=0.05)
    parser.add_argument("--fa-min-s", type=float, default=0.005)
    parser.add_argument("--fa-max-s", type=float, default=0.05)
    args = parser.parse_args()

    vsa_values = extract_values(args.vsa_log, "VSA")
    fa_values = extract_values(args.fa_log, "Vanilla")

    vsa = summarize(vsa_values, args.vsa_min_s, args.vsa_max_s)
    fa = summarize(fa_values, args.fa_min_s, args.fa_max_s)

    print("Latency averages (values interpreted as seconds)")
    print(f"VSA log: {args.vsa_log}")
    print(
        f"  total={vsa.total_count}, selected={vsa.selected_count}, "
        f"avg={vsa.avg_selected_s:.9f} s ({vsa.avg_selected_s*1000:.6f} ms), "
        f"min={vsa.min_selected_s:.9f} s, max={vsa.max_selected_s:.9f} s"
    )
    print(f"FA log:  {args.fa_log}")
    print(
        f"  total={fa.total_count}, selected={fa.selected_count}, "
        f"avg={fa.avg_selected_s:.9f} s ({fa.avg_selected_s*1000:.6f} ms), "
        f"min={fa.min_selected_s:.9f} s, max={fa.max_selected_s:.9f} s"
    )


if __name__ == "__main__":
    main()
