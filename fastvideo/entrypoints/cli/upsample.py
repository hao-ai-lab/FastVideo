# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import cast

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.upsample import upscale_video_file
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser, StoreBoolean

logger = init_logger(__name__)


class UpsampleSubcommand(CLISubcommand):
    """The `upsample` subcommand for the FastVideo CLI."""

    def __init__(self) -> None:
        self.name = "upsample"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        upscale_video_file(
            input_video=args.input_video,
            output_video=args.output_video,
            vae_path=args.vae_path,
            upsampler_path=args.upsampler_path,
            precision=args.precision,
            device=args.device,
            max_frames=args.max_frames,
            trim_frames=args.trim_frames,
            pad_frames=args.pad_frames,
            crop_multiple=args.crop_multiple,
            output_fps=args.output_fps,
        )

    def validate(self, args: argparse.Namespace) -> None:
        if not os.path.exists(args.input_video):
            raise ValueError(f"Input video not found: {args.input_video}")
        if args.crop_multiple is not None and args.crop_multiple < 0:
            raise ValueError("crop_multiple must be >= 0")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if args.trim_frames and args.pad_frames:
            raise ValueError(
                "Only one of --trim-frames or --pad-frames can be enabled")

    def subparser_init(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "upsample",
            help="Upscale an existing video using the LTX-2 spatial upsampler",
            usage=
            ("fastvideo upsample --input-video INPUT.mp4 --output-video OUTPUT.mp4 "
             "[--vae-path PATH] [--upsampler-path PATH]"),
        )

        parser.add_argument(
            "--input-video",
            type=str,
            required=True,
            help="Path to the input video file",
        )
        parser.add_argument(
            "--output-video",
            type=str,
            required=True,
            help="Path to save the upscaled video",
        )
        parser.add_argument(
            "--vae-path",
            type=str,
            default="converted/ltx2_diffusers/vae",
            help="Path to LTX-2 VAE weights (diffusers-style)",
        )
        parser.add_argument(
            "--upsampler-path",
            type=str,
            default="converted/ltx2_spatial_upscaler",
            help="Path to LTX-2 spatial upsampler weights",
        )
        parser.add_argument(
            "--precision",
            type=str,
            default="bf16",
            choices=["fp32", "fp16", "bf16"],
            help="Precision to use for VAE + upsampler",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Torch device string (e.g. cuda, cuda:0, cpu)",
        )
        parser.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="Maximum number of frames to read from the input video",
        )
        parser.add_argument(
            "--trim-frames",
            action=StoreBoolean,
            default=True,
            help="Trim frames to satisfy the 1+8k requirement",
        )
        parser.add_argument(
            "--pad-frames",
            action=StoreBoolean,
            default=False,
            help=
            "Pad frames to satisfy the 1+8k requirement (repeats last frame)",
        )
        parser.add_argument(
            "--crop-multiple",
            type=int,
            default=32,
            help=
            "Center-crop to make H/W divisible by this value (0 to disable)",
        )
        parser.add_argument(
            "--output-fps",
            type=float,
            default=None,
            help="Override output video FPS (defaults to input FPS)",
        )

        return cast(FlexibleArgumentParser, parser)


def cmd_init() -> list[CLISubcommand]:
    return [UpsampleSubcommand()]
