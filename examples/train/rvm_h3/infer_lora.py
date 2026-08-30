#!/usr/bin/env python3
"""Run FastH3 inference with an exported RVM LoRA adapter."""

from __future__ import annotations

import argparse

from examples.inference.basic.fasth3 import build_parser, run, validate_args


def main() -> None:
    parser: argparse.ArgumentParser = build_parser(description=__doc__)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    args = validate_args(parser, parser.parse_args())
    run(args)


if __name__ == "__main__":
    main()
