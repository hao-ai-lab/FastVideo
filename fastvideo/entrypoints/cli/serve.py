# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import dataclasses
import os
from typing import cast

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
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
        excluded_args = {
            "subparser", "config", "dispatch_function", "host", "port",
        }

        provided = getattr(args, '_provided', set())
        cli_kwargs = {}
        for k, v in vars(args).items():
            if k in excluded_args:
                continue
            if k == '_provided':
                continue
            if k in provided and v is not None:
                cli_kwargs[k] = v

        if 'model_path' not in cli_kwargs and args.model_path is not None:
            cli_kwargs['model_path'] = args.model_path

        if not cli_kwargs.get('model_path'):
            raise ValueError("model_path must be provided via --model-path")

        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8000)

        logger.info("CLI serve args: %s", cli_kwargs)
        logger.info("Server will listen on %s:%d", host, port)

        fastvideo_args = FastVideoArgs.from_kwargs(**cli_kwargs)

        from fastvideo.entrypoints.openai.api_server import run_server
        run_server(fastvideo_args, host=host, port=port)
    def validate(self, args: argparse.Namespace) -> None:
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start an OpenAI-compatible HTTP server",
            usage=(
                "fastvideo serve --model-path MODEL_PATH_OR_ID "
                "[--host HOST] [--port PORT] [OPTIONS]"
            ),
        )

        serve_parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind the server to (default: 0.0.0.0)",
        )
        serve_parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to listen on (default: 8000)",
        )
        serve_parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Read CLI options from a config JSON or YAML file.",
        )

        serve_parser = FastVideoArgs.add_cli_args(serve_parser)
        return cast(FlexibleArgumentParser, serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
