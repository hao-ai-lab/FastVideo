# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import os
from typing import cast

from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)
_VALIDATED_SERVE_CONFIG_ATTR = "_fastvideo_validated_serve_config"


class ServeSubcommand(CLISubcommand):
    """Starts an OpenAI-compatible API server."""

    def __init__(self) -> None:
        self.name = "serve"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        serve_config = getattr(args, _VALIDATED_SERVE_CONFIG_ATTR, None)
        if serve_config is None:
            serve_config = build_serve_config(
                args,
                overrides=getattr(args, "_unknown", None),
            )

        logger.info("CLI serve config: %s", serve_config)

        # A `streaming:` block selects the WebSocket/Dynamo runtime;
        # its deps stay out of REST-only deployments via lazy import.
        if serve_config.streaming is not None:
            from fastvideo.entrypoints.streaming.server import (
                run_server as run_streaming_server, )
            run_streaming_server(serve_config)
            return

        from fastvideo.entrypoints.openai.api_server import (
            run_server, )

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
            default_request=serve_config.default_request,
        )

    def validate(self, args: argparse.Namespace) -> None:
        if not args.config:
            raise ValueError("fastvideo serve requires --config PATH; use a nested "
                             "serve config plus optional dotted overrides")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")
        setattr(
            args,
            _VALIDATED_SERVE_CONFIG_ATTR,
            build_serve_config(
                args,
                overrides=getattr(args, "_unknown", None),
            ),
        )

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start an OpenAI-compatible HTTP server",
            usage="fastvideo serve --config SERVE_CONFIG [--dotted.override VALUE]",
        )
        serve_parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Path to a nested config JSON or YAML file. Required.",
        )
        return cast(FlexibleArgumentParser, serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
