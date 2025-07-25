# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import dataclasses
import os
from typing import cast

from fastvideo import VideoGenerator
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.utils import RaiseNotImplementedAction
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class GenerateSubcommand(CLISubcommand):
    """The `generate` subcommand for the FastVideo CLI"""

    def __init__(self) -> None:
        self.name = "generate"
        super().__init__()
        self.init_arg_names = self._get_init_arg_names()
        self.generation_arg_names = self._get_generation_arg_names()

    def _get_init_arg_names(self) -> list[str]:
        """Get names of arguments for VideoGenerator initialization"""
        return ["num_gpus", "tp_size", "sp_size", "model_path"]

    def _get_generation_arg_names(self) -> list[str]:
        """Get names of arguments for generate_video method"""
        return [field.name for field in dataclasses.fields(SamplingParam)]

    def cmd(self, args: argparse.Namespace) -> None:
        excluded_args = ['subparser', 'config', 'dispatch_function']

        provided_args = {}
        for k, v in vars(args).items():
            if (k not in excluded_args and v is not None
                    and hasattr(args, '_provided') and k in args._provided):
                provided_args[k] = v

        if 'model_path' in vars(args) and args.model_path is not None:
            provided_args['model_path'] = args.model_path

        if 'prompt' in vars(args) and args.prompt is not None:
            provided_args['prompt'] = args.prompt

        merged_args = {**provided_args}

        logger.info('CLI Args: %s', merged_args)

        if 'model_path' not in merged_args or not merged_args['model_path']:
            raise ValueError(
                "model_path must be provided either in config file or via --model-path"
            )

        # Check if either prompt or prompt_txt is provided
        has_prompt = 'prompt' in merged_args and merged_args['prompt']
        has_prompt_txt = 'prompt_txt' in merged_args and merged_args[
            'prompt_txt']

        if not (has_prompt or has_prompt_txt):
            raise ValueError("Either prompt or prompt_txt must be provided")

        if has_prompt and has_prompt_txt:
            raise ValueError(
                "Cannot provide both 'prompt' and 'prompt_txt'. Use only one of them."
            )

        init_args = {
            k: v
            for k, v in merged_args.items()
            if k not in self.generation_arg_names
        }
        generation_args = {
            k: v
            for k, v in merged_args.items() if k in self.generation_arg_names
        }

        model_path = init_args.pop('model_path')
        prompt = generation_args.pop('prompt', None)

        generator = VideoGenerator.from_pretrained(model_path=model_path,
                                                   **init_args)

        # Call generate_video - it handles both single and batch modes
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


def cmd_init() -> list[CLISubcommand]:
    return [GenerateSubcommand()]
