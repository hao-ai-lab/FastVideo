# SPDX-License-Identifier: Apache-2.0
"""``fastvideo router-serve`` CLI subcommand.

Launches the streaming router from a YAML config. Separate from
``fastvideo serve`` because the router is an orthogonal process: it
fronts one or more running servers rather than hosting a generator
itself.
"""
from __future__ import annotations

import argparse
import os
from typing import cast

from fastvideo.api.parser import load_raw_config
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.streaming.router.config import (
    ReplicaEndpoint,
    RouterConfig,
)
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class RouterServeSubcommand(CLISubcommand):
    """Start the multi-replica WebSocket router."""

    def __init__(self) -> None:
        self.name = "router-serve"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        config = _load_router_config(args.config)
        logger.info(
            "router listening on %s:%d (%d replicas, %d primary)",
            config.host,
            config.port,
            len(config.replicas),
            sum(1 for r in config.replicas if r.primary),
        )
        from fastvideo.entrypoints.streaming.router.main import run_router

        run_router(config)

    def validate(self, args: argparse.Namespace) -> None:
        if not args.config:
            raise ValueError("fastvideo router-serve requires --config PATH")
        if not os.path.exists(args.config):
            raise ValueError(f"Router config file not found: {args.config}")

    def subparser_init(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "router-serve",
            help="Start the streaming router (multi-replica load balancer)",
            usage="fastvideo router-serve --config ROUTER_CONFIG",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Path to a YAML/JSON router config. Required.",
        )
        return cast(FlexibleArgumentParser, parser)


def _load_router_config(path: str) -> RouterConfig:
    raw = load_raw_config(path)
    router_raw = raw.get("router") if isinstance(raw, dict) else None
    if not isinstance(router_raw, dict):
        raise ValueError(f"Router config {path!r} must have a top-level `router:` block")

    replicas_raw = router_raw.get("replicas", [])
    replicas = [
        ReplicaEndpoint(
            url=r["url"],
            name=r.get("name"),
            primary=bool(r.get("primary", False)),
            weight=float(r.get("weight", 1.0)),
        ) for r in replicas_raw if isinstance(r, dict) and r.get("url")
    ]
    if not replicas:
        raise ValueError("Router config must list at least one replica under `router.replicas`")

    health_check = router_raw.get("health_check") or {}
    return RouterConfig(
        host=str(router_raw.get("host", "0.0.0.0")),
        port=int(router_raw.get("port", 9000)),
        replicas=replicas,
        health_check_path=str(health_check.get("path", "/health")),
        health_check_interval_seconds=float(health_check.get("interval_seconds", 5.0)),
        health_check_timeout_seconds=float(health_check.get("timeout_seconds", 2.0)),
        failure_threshold=int(health_check.get("failure_threshold", 3)),
        recovery_threshold=int(health_check.get("recovery_threshold", 2)),
    )


def cmd_init() -> list[CLISubcommand]:
    return [RouterServeSubcommand()]


__all__ = ["RouterServeSubcommand", "cmd_init"]
