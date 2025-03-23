import argparse
import os
import sys
import yaml
from typing import List

# Use absolute imports with your package name
from fastvideo.v1.entrypoints.cli import utils
from fastvideo.v1.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.v1.utils import FlexibleArgumentParser
from fastvideo.v1.inference_args import InferenceArgs


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the FastVideo CLI"""

    def __init__(self):
        self.name = "serve"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        if args.model_path and args.model_path != args.model:
            raise ValueError(
                "With `fastvideo serve`, you should provide the model as a "
                "positional argument instead of via the `--model_path` option.")
        
        excluded_args = ['subparser', 'config', 'model', 'num_gpus', 'master_port', 'dispatch_function']
        
        # Create a filtered dictionary of arguments
        filtered_args = {k: v for k, v in vars(args).items() 
                        if k not in excluded_args and v is not None}
        
        main_args = ["--model-path", args.model]
        
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

        utils.launch_distributed(args.num_gpus, main_args, master_port=args.master_port)

    def validate(self, args: argparse.Namespace) -> None:
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

        if args.master_port is not None and (args.master_port < 1024 or args.master_port > 65535):
            raise ValueError("Master port must be between 1024 and 65535")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Run inference on a model",
            usage="fastvideo serve <model_path> [options]"
        )

        serve_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config YAML file."
        )

        serve_parser.add_argument(
            "model",
            type=str,
            help="Model to serve (will be passed as --model-path to the inference script)"
        )
        serve_parser.add_argument(
            "--num-gpus",
            type=int,
            help="Number of GPUs to use",
            required=True
        )
        serve_parser.add_argument(
            "--master-port",
            type=int,
            default=None,
            help="Port for the master process"
        )

        serve_parser = InferenceArgs.add_cli_args(serve_parser)

        return serve_parser


def cmd_init() -> List[CLISubcommand]:
    return [ServeSubcommand()] 