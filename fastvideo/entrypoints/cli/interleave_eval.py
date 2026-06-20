# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint for Interleave prompt-set evaluation."""

from __future__ import annotations

import argparse
import os
from typing import cast

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.interleave.config import load_interleave_run_config
from fastvideo.entrypoints.interleave.evaluation import run_interleave_prompt_set_config
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)
_VALIDATED_INTERLEAVE_EVAL_CONFIG_ATTR = "_fastvideo_validated_interleave_eval_config"


class InterleaveEvalSubcommand(CLISubcommand):
    """Runs a config-driven Interleave prompt set and writes aggregate metrics."""

    def __init__(self) -> None:
        self.name = "interleave-eval"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        run_config = getattr(args, _VALIDATED_INTERLEAVE_EVAL_CONFIG_ATTR, None)
        if run_config is None:
            run_config = load_interleave_run_config(
                args.config,
                overrides=getattr(args, "_unknown", None),
                output_dir=args.output_dir,
                require_instruction=False,
            )
        logger.info("CLI interleave eval config: %s", run_config)

        summary = run_interleave_prompt_set_config(
            run_config,
            args.prompts,
            output_dir=args.output_dir,
            summary_path=args.summary_path,
            limit=args.limit,
            resume=args.resume,
        )
        print(f"Summary: {summary.summary_path}")
        print(f"Samples: {summary.num_samples}")
        print(f"Success: {summary.num_success}/{summary.num_samples}")
        print(f"Success rate: {summary.success_rate:.4f}")

    def validate(self, args: argparse.Namespace) -> None:
        if not args.config:
            raise ValueError("fastvideo interleave-eval requires --config PATH")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")
        if not args.prompts:
            raise ValueError("fastvideo interleave-eval requires --prompts PATH")
        if not os.path.exists(args.prompts):
            raise ValueError(f"Prompt set not found: {args.prompts}")
        if args.limit is not None and args.limit <= 0:
            raise ValueError(f"--limit must be > 0; got {args.limit}")
        setattr(
            args,
            _VALIDATED_INTERLEAVE_EVAL_CONFIG_ATTR,
            load_interleave_run_config(
                args.config,
                overrides=getattr(args, "_unknown", None),
                output_dir=args.output_dir,
                require_instruction=False,
            ),
        )

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "interleave-eval",
            help="Run native Interleave planner/generator/critic orchestration over a prompt set",
            usage="fastvideo interleave-eval --config CONFIG --prompts PROMPTS [--dotted.override VALUE]",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Path to a nested interleave run config JSON or YAML file. Required.",
        )
        parser.add_argument(
            "--prompts",
            type=str,
            default="",
            required=False,
            help="Prompt set path. JSONL, JSON, and plain text files are supported.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Directory for per-sample traces and the default summary.json.",
        )
        parser.add_argument(
            "--summary-path",
            type=str,
            default=None,
            help="Optional path for the aggregate summary JSON.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Optional maximum number of prompt rows to run.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Skip prompt rows whose trace.json already exists.",
        )
        return cast(FlexibleArgumentParser, parser)


def cmd_init() -> list[CLISubcommand]:
    return [InterleaveEvalSubcommand()]
