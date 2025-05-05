# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import json
import os
import dataclasses
from typing import List, Dict, Any, cast
import sys

from fastvideo.v1.entrypoints.cli import utils
from fastvideo.v1.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.configs.sample.base import SamplingParam
from fastvideo.v1.utils import FlexibleArgumentParser
from fastvideo import VideoGenerator


class GenerateSubcommand(CLISubcommand):
    """The `generate` subcommand for the FastVideo CLI"""

    def __init__(self) -> None:
        self.name = "generate"
        super().__init__()
        
        # Define which arguments belong to which group
        self.init_arg_names = self._get_init_arg_names()
        self.generation_arg_names = self._get_generation_arg_names()

    def _get_init_arg_names(self) -> List[str]:
        """Get names of arguments for VideoGenerator initialization"""
        # Get all field names from FastVideoArgs
        return [field.name for field in dataclasses.fields(FastVideoArgs)]

    def _get_generation_arg_names(self) -> List[str]:
        """Get names of arguments for generate_video method"""
        # Get all field names from SamplingParam
        return [field.name for field in dataclasses.fields(SamplingParam)]

    def cmd(self, args: argparse.Namespace) -> None:
        excluded_args = [
            'subparser', 'config', 'num_gpus', 'master_port',
            'dispatch_function'
        ]

        print(f"argsx: {args}")

        filtered_args = {}
        for k, v in vars(args).items():
            if k not in excluded_args and v is not None:
                filtered_args[k] = v

        # Extract nested arguments
        # filtered_args = {}
        # for k, v in vars(args).items():
        #     if k not in excluded_args and v is not None:
        #         if '.' in k:
        #             # Handle nested arguments
        #             parts = k.split('.')
        #             current = filtered_args
        #             for part in parts[:-1]:
        #                 if part not in current:
        #                     current[part] = {}
        #                 current = current[part]
        #             current[parts[-1]] = v
        #         else:
        #             filtered_args[k] = v

        # filtered_args = {
        #     k: v
        #     for k, v in filtered_args.items()
        #     if k not in excluded_args and v is not None
        # }
        
        
        print(f"filtered_args: {filtered_args}")
        # CLI args take precedence over config file args
        merged_args = {**filtered_args}
        
        # Validate that required arguments are present in the merged args
        if 'model_path' not in merged_args or not merged_args['model_path']:
            raise ValueError("model_path must be provided either in config file or via --model-path")
        
        if 'prompt' not in merged_args or not merged_args['prompt']:
            raise ValueError("prompt must be provided either in config file or via --prompt")
        
        print(f"init_arg_names: {self.init_arg_names}")
        print(f"generation_arg_names: {self.generation_arg_names}")
        # Now separate arguments for initialization and generation
        init_args = {k: v for k, v in merged_args.items() if k in self.init_arg_names}
        generation_args = {k: v for k, v in merged_args.items() if k in self.generation_arg_names}
        
        # Get model path
        model_path = init_args.pop('model_path')
        
        # Get prompt from args or config
        prompt = generation_args.pop('prompt')
        
        # Print args for debugging
        print(f"init_args: {init_args}")
        print(f"generation_args: {generation_args}")
        
        # Create generator using from_pretrained
        generator = VideoGenerator.from_pretrained(
            model_path=model_path,
            **init_args
        )
        
        # Generate the video
        generator.generate_video(prompt=prompt, **generation_args)

    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

        if args.master_port is not None and (args.master_port < 1024
                                             or args.master_port > 65535):
            raise ValueError("Master port must be between 1024 and 65535")
        
        # Check if config file exists if provided
        if args.config:
            if not os.path.exists(args.config):
                raise ValueError(f"Config file not found: {args.config}")

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        generate_parser = subparsers.add_parser(
            "generate",
            help="Run inference on a model",
            usage=
            "fastvideo generate (--model-path MODEL_PATH_OR_ID --prompt PROMPT) | --config CONFIG_FILE [OPTIONS]"
        )

        generate_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config JSON or YAML file. If provided, --model-path and --prompt are optional."
        )

        generate_parser.add_argument("--master-port",
                                     type=int,
                                     default=None,
                                     help="Port for the master process")

        # Add model initialization arguments
        generate_parser = FastVideoArgs.add_cli_args(generate_parser)
        
        # Add generation-specific arguments
        generate_parser = SamplingParam.add_cli_args(generate_parser)

        # Add argument for text encoder configs as a JSON string
        generate_parser.add_argument(
            "--text-encoder-configs",
            type=str,
            dest="text_encoder_configs",
            default=None,
            help="JSON array of text encoder configurations (e.g., '[{\"prefix\":\"llama\",\"quant_config\":null},{\"prefix\":\"clip\",\"quant_config\":null}]')",
        )

        return cast(FlexibleArgumentParser, generate_parser)


def cmd_init() -> List[CLISubcommand]:
    return [GenerateSubcommand()]
