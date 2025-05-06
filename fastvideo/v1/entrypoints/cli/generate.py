# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import dataclasses
import os
from typing import Any, Dict, List, Optional, cast

from fastvideo import PipelineConfig, VideoGenerator
from fastvideo.v1.configs.sample.base import SamplingParam
from fastvideo.v1.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.v1.entrypoints.cli.utils import RaiseNotImplementedAction
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.utils import FlexibleArgumentParser


class GenerateSubcommand(CLISubcommand):
    """The `generate` subcommand for the FastVideo CLI"""

    def __init__(self) -> None:
        self.name = "generate"
        super().__init__()
        self.init_arg_names = self._get_init_arg_names()
        self.generation_arg_names = self._get_generation_arg_names()

    def _get_init_arg_names(self) -> List[str]:
        """Get names of arguments for VideoGenerator initialization"""
        return ["num_gpus", "tp_size", "sp_size", "model_path"]

    def _get_generation_arg_names(self) -> List[str]:
        """Get names of arguments for generate_video method"""
        return [field.name for field in dataclasses.fields(SamplingParam)]

    def cmd(self, args: argparse.Namespace) -> None:
        excluded_args = ['subparser', 'config', 'dispatch_function']

        FastVideoArgs.from_cli_args(args)
        filtered_args = {}
        for k, v in vars(args).items():
            if k not in excluded_args and v is not None:
                filtered_args[k] = v

        merged_args = {**filtered_args}

        if 'model_path' not in merged_args or not merged_args['model_path']:
            raise ValueError(
                "model_path must be provided either in config file or via --model-path"
            )

        if 'prompt' not in merged_args or not merged_args['prompt']:
            raise ValueError(
                "prompt must be provided either in config file or via --prompt")

        init_args = {
            k: v
            for k, v in merged_args.items() if k in self.init_arg_names
        }
        generation_args = {
            k: v
            for k, v in merged_args.items() if k in self.generation_arg_names
        }

        pipeline_config = PipelineConfig.from_pretrained(
            merged_args['model_path'])

        update_config_from_args(pipeline_config.dit_config, merged_args,
                                "dit_config")
        update_config_from_args(pipeline_config.vae_config, merged_args,
                                "vae_config")
        update_config_from_args(pipeline_config, merged_args)

        model_path = init_args.pop('model_path')
        prompt = generation_args.pop('prompt')

        generator = VideoGenerator.from_pretrained(
            model_path=model_path, **init_args, pipeline_config=pipeline_config)

        generator.generate_video(prompt=prompt, **generation_args)

    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

        if args.config and not os.path.exists(args.config):
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
            help=
            "Read CLI options from a config JSON or YAML file. If provided, --model-path and --prompt are optional."
        )

        generate_parser = FastVideoArgs.add_cli_args(generate_parser)
        generate_parser = SamplingParam.add_cli_args(generate_parser)

        generate_parser.add_argument(
            "--text-encoder-configs",
            action=RaiseNotImplementedAction,
            help=
            "JSON array of text encoder configurations (NOT YET IMPLEMENTED)",
        )

        return cast(FlexibleArgumentParser, generate_parser)


def cmd_init() -> List[CLISubcommand]:
    return [GenerateSubcommand()]


def update_config_from_args(config: Any,
                            args_dict: Dict[str, Any],
                            prefix: Optional[str] = None) -> None:
    """
    Update configuration object from arguments dictionary.
    
    Args:
        config: The configuration object to update
        args_dict: Dictionary containing arguments
        prefix: Prefix for the configuration parameters in the args_dict.
               If None, assumes direct attribute mapping without prefix.
    """
    # Handle top-level attributes (no prefix)
    if prefix is None:
        for key, value in args_dict.items():
            if hasattr(config, key) and value is not None:
                if key == "text_encoder_precisions" and isinstance(value, list):
                    setattr(config, key, tuple(value))
                else:
                    setattr(config, key, value)
        return

    # Handle nested attributes with prefix
    prefix_with_dot = f"{prefix}."
    for key, value in args_dict.items():
        if key.startswith(prefix_with_dot) and value is not None:
            attr_name = key[len(prefix_with_dot):]
            if hasattr(config, attr_name):
                setattr(config, attr_name, value)
