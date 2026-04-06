# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import os
from typing import cast

from fastvideo import VideoGenerator
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.inference_config import build_generate_run_config
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class GenerateSubcommand(CLISubcommand):
    """The `generate` subcommand for the FastVideo CLI"""

    def __init__(self) -> None:
        self.name = "generate"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        run_config = build_generate_run_config(
            args,
            overrides=getattr(args, "_unknown", None),
        )
        logger.info("CLI generate config: %s", run_config)

        generator = VideoGenerator.from_config(run_config.generator)
        generator.generate(run_config.request)

    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        if not args.config:
            raise ValueError("fastvideo generate requires --config PATH; use a nested "
                             "run config plus optional dotted overrides")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        generate_parser = subparsers.add_parser(
            "generate",
            help="Run inference on a model",
            usage="fastvideo generate --config RUN_CONFIG [--dotted.override VALUE]")

        generate_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Path to a nested run config JSON or YAML file. Required.",
        )

        return cast(FlexibleArgumentParser, generate_parser)


def cmd_init() -> list[CLISubcommand]:
    return [GenerateSubcommand()]
