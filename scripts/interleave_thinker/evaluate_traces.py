#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate saved Interleave prompt-set traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastvideo.entrypoints.interleave.trace_eval import (
    evaluate_interleave_traces,
    interleave_trace_evaluation_to_dict,
    write_interleave_trace_evaluation,
    write_interleave_trace_html_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Interleave trace.json files or prompt-set output dirs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Trace file, summary.json, or prompt-set output directory. Multiple inputs are allowed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write aggregate metrics JSON to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Optional HTML report path with per-trace rows and final-image thumbnails.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Interleave Trace Evaluation",
        help="Title for the optional HTML report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_interleave_traces([Path(path) for path in args.paths])
    if args.output:
        write_interleave_trace_evaluation(summary, args.output)
        print(f"Wrote metrics: {args.output}")
    else:
        print(json.dumps(interleave_trace_evaluation_to_dict(summary), indent=2, sort_keys=True))
    if args.html_output:
        write_interleave_trace_html_report(summary, args.html_output, title=args.title)
        print(f"Wrote HTML report: {args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
