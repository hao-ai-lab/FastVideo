# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint for native InterleaveThinker orchestration."""

from __future__ import annotations

import argparse
import os
from typing import cast

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.interleave.config import load_interleave_run_config
from fastvideo.entrypoints.interleave.runner import run_interleave_config
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)
_VALIDATED_INTERLEAVE_RUN_CONFIG_ATTR = "_fastvideo_validated_interleave_run_config"


class InterleaveRunSubcommand(CLISubcommand):
    """Runs a native planner -> generator -> critic interleave trace."""

    def __init__(self) -> None:
        self.name = "interleave-run"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        run_config = getattr(args, _VALIDATED_INTERLEAVE_RUN_CONFIG_ATTR, None)
        if run_config is None:
            run_config = load_interleave_run_config(
                args.config,
                overrides=getattr(args, "_unknown", None),
                prompt=args.prompt,
                input_image=args.input_image,
                output_dir=args.output_dir,
                trace_path=args.trace_path,
            )
        logger.info("CLI interleave run config: %s", run_config)

        result = run_interleave_config(run_config)
        final_path = result.trace.final_image.file_path if result.trace.final_image is not None else None
        if final_path:
            print(f"Image: {final_path}")
        print(f"Trace: {result.trace_path}")
        print(f"Success: {result.trace.success}")

    def validate(self, args: argparse.Namespace) -> None:
        if not args.config:
            raise ValueError("fastvideo interleave-run requires --config PATH; "
                             "use a nested interleave config plus optional dotted overrides")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")
        setattr(
            args,
            _VALIDATED_INTERLEAVE_RUN_CONFIG_ATTR,
            load_interleave_run_config(
                args.config,
                overrides=getattr(args, "_unknown", None),
                prompt=args.prompt,
                input_image=args.input_image,
                output_dir=args.output_dir,
                trace_path=args.trace_path,
            ),
        )

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "interleave-run",
            help="Run native InterleaveThinker planner/generator/critic orchestration",
            usage="fastvideo interleave-run --config CONFIG [--prompt TEXT] [--dotted.override VALUE]",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Path to a nested interleave run config JSON or YAML file. Required.",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default=None,
            help="Override interleave.instruction for this run.",
        )
        parser.add_argument(
            "--input-image",
            type=str,
            default=None,
            help="Override interleave.initial_image_path for image-conditioned planning/editing.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Override interleave.output_dir.",
        )
        parser.add_argument(
            "--trace-path",
            type=str,
            default=None,
            help="Override interleave.trace_path.",
        )
        return cast(FlexibleArgumentParser, parser)


def cmd_init() -> list[CLISubcommand]:
    return [InterleaveRunSubcommand()]
