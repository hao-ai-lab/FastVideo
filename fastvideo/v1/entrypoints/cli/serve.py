import argparse
import os
import sys
from typing import List

# Use absolute imports with your package name
from fastvideo.v1.entrypoints.cli import utils
from fastvideo.v1.entrypoints.cli.cli_types import CLISubcommand


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the FastVideo CLI"""
    
    def __init__(self):
        self.name = "serve"
        super().__init__()
    
    def cmd(self, args: argparse.Namespace) -> None:
        # Prepare the arguments for the inference script
        main_args = []
        
        # Transform the positional model argument to --model-path
        main_args.extend(["--model-path", args.model])
        
        # Add num_gpus if specified
        if args.num_gpus is not None:
            main_args.extend(["--num-gpus", str(args.num_gpus)])
                
        # Add any unknown arguments that were passed
        if hasattr(args, 'unknown_args') and args.unknown_args:            
            main_args.extend(args.unknown_args)
        
        # Pass the arguments to the distributed launcher
        utils.launch_distributed(args.num_gpus, main_args, master_port=args.master_port)
    
    def validate(self, args: argparse.Namespace) -> None:
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")
        
        if args.master_port is not None and (args.master_port < 1024 or args.master_port > 65535):
            raise ValueError("Master port must be between 1024 and 65535")
    
    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        train_parser = subparsers.add_parser(
            "serve",
            help="Run inference on a model",
            # Allow unknown arguments to be passed through
            add_help=True
        )
        train_parser.add_argument(
            "model",
            type=str,
            help="Model to serve (will be passed as --model-path to the inference script)"
        )
        train_parser.add_argument(
            "--num-gpus",
            type=int,
            default=None,  # Will be set in utils.py if None
            help="Number of GPUs to use"
        )
        train_parser.add_argument(
            "--master-port",
            type=int,
            default=None,  # Default to None, will use a random port
            help="Port for the master process (default: random)"
        )
        
        return train_parser


def cmd_init() -> List[CLISubcommand]:
    return [ServeSubcommand()] 