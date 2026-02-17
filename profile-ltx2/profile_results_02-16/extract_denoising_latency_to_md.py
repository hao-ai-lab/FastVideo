#!/usr/bin/env python3
"""Extract denoising-stage latencies from two logs and write a markdown table."""

from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass


EXEC_RE = re.compile(r"Execution completed in ([0-9]+(?:\.[0-9]+)?) ms")
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@dataclass
class LogStats:
    runs: list[str]
    avg_all_rounds: str
    avg_excl_warmup: str


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(path: pathlib.Path) -> LogStats:
    runs: list[str] = []
    denoise_col = 4  # 0-based in split row; fallback if stage names are absent
    avg_all_rounds = ""
    avg_excl_warmup = ""

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line).strip()

            m = EXEC_RE.search(line)
            if m and "LTX2DenoisingStage" in line:
                runs.append(m.group(1))
                continue

            if line.startswith("Stage names,"):
                parts = [p.strip() for p in line.split(",")]
                for idx, name in enumerate(parts):
                    if "denois" in name.lower():
                        denoise_col = idx
                        break
                continue

            if line.startswith("Avg (all rounds),"):
                parts = [p.strip() for p in line.split(",")]
                if denoise_col < len(parts):
                    avg_all_rounds = parts[denoise_col]
                continue

            if line.startswith("Avg (excl. warmup),"):
                parts = [p.strip() for p in line.split(",")]
                if denoise_col < len(parts):
                    avg_excl_warmup = parts[denoise_col]

    if not runs:
        raise ValueError(f"No denoising run latencies found in: {path}")
    if not avg_all_rounds or not avg_excl_warmup:
        raise ValueError(f"Missing denoising averages in summary: {path}")

    return LogStats(
        runs=runs,
        avg_all_rounds=avg_all_rounds,
        avg_excl_warmup=avg_excl_warmup,
    )


def build_markdown(log1_name: str, stats1: LogStats, log2_name: str, stats2: LogStats) -> str:
    if len(stats1.runs) != len(stats2.runs):
        raise ValueError(
            "Run counts do not match between logs: "
            f"{log1_name} has {len(stats1.runs)}, {log2_name} has {len(stats2.runs)}"
        )

    lines = [
        "# Denoising Stage Latency",
        "",
        "Denoising stage: `LTX2DenoisingStage` (ms)",
        "",
        f"| Metric | `{log1_name}` (ms) | `{log2_name}` (ms) |",
        "|---|---:|---:|",
    ]

    for i, (v1, v2) in enumerate(zip(stats1.runs, stats2.runs), start=1):
        lines.append(f"| run_{i:02d} | {v1} | {v2} |")

    lines.append(
        f"| avg_all_rounds | {stats1.avg_all_rounds} | {stats2.avg_all_rounds} |"
    )
    lines.append(
        f"| avg_excl_warmup | {stats1.avg_excl_warmup} | {stats2.avg_excl_warmup} |"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LTX2 denoising-stage latency runs and averages from two logs "
            "into a single markdown table."
        )
    )
    parser.add_argument("log1", type=pathlib.Path, help="First log file path")
    parser.add_argument("log2", type=pathlib.Path, help="Second log file path")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("denoising_stage_latency.md"),
        help="Output markdown path (default: ./denoising_stage_latency.md)",
    )
    args = parser.parse_args()

    stats1 = parse_log(args.log1)
    stats2 = parse_log(args.log2)
    markdown = build_markdown(args.log1.name, stats1, args.log2.name, stats2)
    args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
