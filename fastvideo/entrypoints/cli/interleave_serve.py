# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint for the InterleaveThinker-compatible image service."""

from __future__ import annotations

import argparse
import os
from typing import cast

from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.inference_config import build_serve_config
from fastvideo.entrypoints.interleave.server import run_server
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)
_VALIDATED_INTERLEAVE_SERVE_CONFIG_ATTR = "_fastvideo_validated_interleave_serve_config"


class InterleaveServeSubcommand(CLISubcommand):
    """Starts an InterleaveThinker-compatible FastVideo image service."""

    def __init__(self) -> None:
        self.name = "interleave-serve"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        serve_config = getattr(args, _VALIDATED_INTERLEAVE_SERVE_CONFIG_ATTR, None)
        if serve_config is None:
            serve_config = build_serve_config(
                args,
                overrides=getattr(args, "_unknown", None),
            )

        logger.info("CLI interleave serve config: %s", serve_config)
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
            raise ValueError("fastvideo interleave-serve requires --config PATH; "
                             "use a nested serve config plus optional dotted overrides")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")
        setattr(
            args,
            _VALIDATED_INTERLEAVE_SERVE_CONFIG_ATTR,
            build_serve_config(
                args,
                overrides=getattr(args, "_unknown", None),
            ),
        )

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "interleave-serve",
            help="Start an InterleaveThinker-compatible image service",
            usage="fastvideo interleave-serve --config SERVE_CONFIG [--dotted.override VALUE]",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Path to a nested serve config JSON or YAML file. Required.",
        )
        return cast(FlexibleArgumentParser, parser)


def cmd_init() -> list[CLISubcommand]:
    return [InterleaveServeSubcommand()]
