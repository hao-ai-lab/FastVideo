#!/usr/bin/env python3
"""Append compiled-vs-noncompiled denoising latency tables to a markdown file."""

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
    denoise_col = 4
    avg_all_rounds = ""
    avg_excl_warmup = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
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
        raise ValueError(f"No denoising runs found in: {path}")
    if not avg_all_rounds or not avg_excl_warmup:
        raise ValueError(f"Missing denoising averages in: {path}")

    return LogStats(
        runs=runs,
        avg_all_rounds=avg_all_rounds,
        avg_excl_warmup=avg_excl_warmup,
    )


def build_comparison_table(
    title: str,
    noncompiled_name: str,
    noncompiled_stats: LogStats,
    compiled_name: str,
    compiled_stats: LogStats,
) -> str:
    if len(noncompiled_stats.runs) != len(compiled_stats.runs):
        raise ValueError(
            f"Run count mismatch for {title}: "
            f"{noncompiled_name} has {len(noncompiled_stats.runs)}, "
            f"{compiled_name} has {len(compiled_stats.runs)}"
        )

    lines = [
        f"## {title}",
        "",
        "Denoising stage: `LTX2DenoisingStage` (ms)",
        "",
        f"| Metric | `{noncompiled_name}` (non-compiled) | `{compiled_name}` (compiled) |",
        "|---|---:|---:|",
    ]

    for i, (nonc, comp) in enumerate(
        zip(noncompiled_stats.runs, compiled_stats.runs), start=1
    ):
        lines.append(f"| run_{i:02d} | {nonc} | {comp} |")

    lines.append(
        "| avg_all_rounds | "
        f"{noncompiled_stats.avg_all_rounds} | {compiled_stats.avg_all_rounds} |"
    )
    lines.append(
        "| avg_excl_warmup | "
        f"{noncompiled_stats.avg_excl_warmup} | {compiled_stats.avg_excl_warmup} |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Append two denoising latency comparison tables (regular and VSA): "
            "non-compiled vs compiled."
        )
    )
    parser.add_argument(
        "--regular-noncompiled",
        type=pathlib.Path,
        default=script_dir / "profile.4.log",
        help="Regular non-compiled log path",
    )
    parser.add_argument(
        "--regular-compiled",
        type=pathlib.Path,
        default=script_dir / "profile.4.compile.log",
        help="Regular compiled log path",
    )
    parser.add_argument(
        "--vsa-noncompiled",
        type=pathlib.Path,
        default=repo_root / "profile.4.vsa.trace.log",
        help="VSA non-compiled log path",
    )
    parser.add_argument(
        "--vsa-compiled",
        type=pathlib.Path,
        default=script_dir / "profile.4.vsa.compile.log",
        help="VSA compiled log path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=script_dir / "denoising_stage_latency.md",
        help="Markdown file to append to",
    )
    args = parser.parse_args()

    regular_nonc = parse_log(args.regular_noncompiled)
    regular_comp = parse_log(args.regular_compiled)
    try:
        vsa_nonc = parse_log(args.vsa_noncompiled)
    except ValueError:
        # Some runs keep the non-compiled VSA timings only in trace logs.
        if args.vsa_noncompiled.name == "profile.4.vsa.log":
            fallback = args.vsa_noncompiled.with_name("profile.4.vsa.trace.log")
            vsa_nonc = parse_log(fallback)
            args.vsa_noncompiled = fallback
        else:
            raise
    vsa_comp = parse_log(args.vsa_compiled)

    block = "\n".join(
        [
            "",
            "## Additional Comparison",
            "",
            build_comparison_table(
                "Regular Attention: Non-Compiled vs Compiled",
                args.regular_noncompiled.name,
                regular_nonc,
                args.regular_compiled.name,
                regular_comp,
            ),
            build_comparison_table(
                "VSA Attention: Non-Compiled vs Compiled",
                args.vsa_noncompiled.name,
                vsa_nonc,
                args.vsa_compiled.name,
                vsa_comp,
            ),
        ]
    )

    with args.output.open("a", encoding="utf-8") as out:
        out.write(block)


if __name__ == "__main__":
    main()
