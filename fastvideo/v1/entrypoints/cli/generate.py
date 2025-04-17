# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
from typing import List, cast

from fastvideo.v1.entrypoints.cli import utils
from fastvideo.v1.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.utils import FlexibleArgumentParser


class GenerateSubcommand(CLISubcommand):
    """The `generate` subcommand for the FastVideo CLI"""

    def __init__(self) -> None:
        self.name = "generate"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        excluded_args = [
            'subparser', 'config', 'num_gpus', 'master_port',
            'dispatch_function'
        ]

        # Create a filtered dictionary of arguments
        filtered_args = {
            k: v
            for k, v in vars(args).items()
            if k not in excluded_args and v is not None
        }

        main_args = []

        for key, value in filtered_args.items():
            # Convert underscores to dashes in argument names
            arg_name = f"--{key.replace('_', '-')}"

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    main_args.append(arg_name)
            else:
                main_args.append(arg_name)
                main_args.append(str(value))

        utils.launch_distributed(args.num_gpus,
                                 main_args,
                                 master_port=args.master_port)

    def validate(self, args: argparse.Namespace) -> None:
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

        if args.master_port is not None and (args.master_port < 1024
                                             or args.master_port > 65535):
            raise ValueError("Master port must be between 1024 and 65535")

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        generate_parser = subparsers.add_parser(
            "generate",
            help="Run inference on a model",
            usage=
            "fastvideo generate --model-path MODEL_PATH_OR_ID --prompt PROMPT [OPTIONS]"
        )

        generate_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config YAML file.")

        generate_parser.add_argument("--master-port",
                                     type=int,
                                     default=None,
                                     help="Port for the master process")

        generate_parser = FastVideoArgs.add_cli_args(generate_parser)

        return cast(FlexibleArgumentParser, generate_parser)


def cmd_init() -> List[CLISubcommand]:
    return [GenerateSubcommand()]
