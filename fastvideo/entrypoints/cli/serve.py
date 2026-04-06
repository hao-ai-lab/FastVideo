# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import os
from typing import cast

from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class ServeSubcommand(CLISubcommand):
    """Starts an OpenAI-compatible API server."""

    def __init__(self) -> None:
        self.name = "serve"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        if not args.config:
            raise ValueError("fastvideo serve requires --config PATH; use nested config "
                             "plus optional dotted or flat overrides")
        serve_config = build_serve_config(
            args,
            overrides=getattr(args, "_unknown", None),
        )
        if serve_config.default_request != GenerationRequest():
            raise NotImplementedError("ServeConfig.default_request is not wired into the OpenAI "
                                      "server yet")

        from fastvideo.entrypoints.openai.api_server import (
            run_server, )

        logger.info("CLI serve config: %s", serve_config)
        logger.info(
            "Server will listen on %s:%d",
            serve_config.server.host,
            serve_config.server.port,
        )

        fastvideo_args = generator_config_to_fastvideo_args(serve_config.generator)
        run_server(
            fastvideo_args,
            host=serve_config.server.host,
            port=serve_config.server.port,
            output_dir=serve_config.server.output_dir,
        )

    def validate(self, args: argparse.Namespace) -> None:
        if not args.config:
            raise ValueError("fastvideo serve requires --config PATH; use nested config "
                             "plus optional dotted or flat overrides")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        from fastvideo.entrypoints.openai.api_server import (
            DEFAULT_HOST,
            DEFAULT_OUTPUT_DIR,
            DEFAULT_PORT,
        )

        serve_parser = subparsers.add_parser(
            "serve",
            help="Start an OpenAI-compatible HTTP server",
            usage=("fastvideo serve --config SERVE_CONFIG "
                   "[--host HOST] [--port PORT] [OPTIONS]"),
        )

        serve_parser.add_argument(
            "--host",
            type=str,
            default=DEFAULT_HOST,
            help=f"Host to bind the server to (default: {DEFAULT_HOST})",
        )
        serve_parser.add_argument(
            "--port",
            type=int,
            default=DEFAULT_PORT,
            help=f"Port to listen on (default: {DEFAULT_PORT})",
        )
        serve_parser.add_argument(
            "--output-dir",
            type=str,
            default=DEFAULT_OUTPUT_DIR,
            help=("Directory for generated outputs "
                  f"(default: {DEFAULT_OUTPUT_DIR})"),
        )
        serve_parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Path to a nested config JSON or YAML file. Required.",
        )

        serve_parser = FastVideoArgs.add_cli_args(serve_parser)
        return cast(FlexibleArgumentParser, serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
