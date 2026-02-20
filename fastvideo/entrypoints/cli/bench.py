# Adapted from SGLang (https://github.com/sgl-project/sglang)
"""
Runs benchmark against a running FastVideo OpenAI-compatible server.

Example usage:
    fastvideo bench --dataset vbench --num-prompts 20 --port 8000
"""

import argparse
from typing import cast

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class BenchSubcommand(CLISubcommand):
    """The ``bench`` subcommand — runs serving benchmarks."""

    def __init__(self) -> None:
        self.name = "bench"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        import asyncio
        from fastvideo.entrypoints.cli.bench_serving import benchmark
        asyncio.run(benchmark(args))

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            "bench",
            help="Benchmark a running FastVideo server",
            usage="fastvideo bench [--dataset vbench] [--num-prompts N] [--port PORT]",
        )

        bench_parser.add_argument(
            "--backend", type=str, default=None,
            help="DEPRECATED: will be ignored.",
        )
        bench_parser.add_argument(
            "--base-url", type=str, default=None,
            help="Base URL of the server (e.g., http://localhost:8000). Overrides host/port.",
        )
        bench_parser.add_argument("--host", type=str, default="localhost")
        bench_parser.add_argument("--port", type=int, default=8000)
        bench_parser.add_argument("--model", type=str, default="default")
        bench_parser.add_argument(
            "--dataset", type=str, default="vbench",
            choices=["vbench", "random"],
        )
        bench_parser.add_argument(
            "--task", type=str, default=None,
            choices=["text-to-video", "image-to-video", "text-to-image", "image-to-image", "video-to-video"],
        )
        bench_parser.add_argument("--dataset-path", type=str, default=None)
        bench_parser.add_argument("--num-prompts", type=int, default=10)
        bench_parser.add_argument("--max-concurrency", type=int, default=1)
        bench_parser.add_argument("--request-rate", type=float, default=float("inf"))
        bench_parser.add_argument("--width", type=int, default=None)
        bench_parser.add_argument("--height", type=int, default=None)
        bench_parser.add_argument("--num-frames", type=int, default=None)
        bench_parser.add_argument("--fps", type=int, default=None)
        bench_parser.add_argument("--output-file", type=str, default=None)
        bench_parser.add_argument("--disable-tqdm", action="store_true")
        bench_parser.add_argument(
            "--log-level", type=str, default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )

        return cast(FlexibleArgumentParser, bench_parser)


def cmd_init() -> list[CLISubcommand]:
    return [BenchSubcommand()]
