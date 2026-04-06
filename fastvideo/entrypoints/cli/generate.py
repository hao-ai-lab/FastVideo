# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import os
from typing import cast

from fastvideo import VideoGenerator
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.inference_config import build_generate_run_config
from fastvideo.entrypoints.cli.utils import RaiseNotImplementedAction
from fastvideo.fastvideo_args import FastVideoArgs
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
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

        if args.config and not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        generate_parser = subparsers.add_parser(
            "generate",
            help="Run inference on a model",
            usage="fastvideo generate (--model-path MODEL_PATH_OR_ID --prompt PROMPT) | --config CONFIG_FILE [OPTIONS]")

        generate_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config JSON or YAML file. If provided, --model-path and --prompt are optional."
        )

        generate_parser = FastVideoArgs.add_cli_args(generate_parser)
        generate_parser = SamplingParam.add_cli_args(generate_parser)

        generate_parser.add_argument(
            "--text-encoder-configs",
            action=RaiseNotImplementedAction,
            help="JSON array of text encoder configurations (NOT YET IMPLEMENTED)",
        )

        return cast(FlexibleArgumentParser, generate_parser)


def cmd_init() -> list[CLISubcommand]:
    return [GenerateSubcommand()]
