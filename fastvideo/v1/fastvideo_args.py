# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of FastVideo Inference."""

import argparse
import dataclasses
from contextlib import contextmanager
from typing import List, Optional

from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import FlexibleArgumentParser

logger = init_logger(__name__)


@dataclasses.dataclass
class FastVideoArgs:
    # Model and path configuration
    model_path: str

    # Distributed executor backend
    distributed_executor_backend: str = "mp"

    inference_mode: bool = True  # if False == training mode

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: Optional[str] = None

    # Parallelism
    num_gpus: int = 1
    tp_size: Optional[int] = None
    sp_size: Optional[int] = None
    dist_timeout: Optional[int] = None  # timeout for torch.distributed

    # Video generation parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 117
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    embedded_cfg_scale: float = 6.0
    flow_shift: Optional[float] = None

    output_type: str = "pil"

    # Model configuration
    precision: str = "bf16"

    # VAE configuration
    vae_precision: str = "fp16"
    vae_tiling: bool = True
    vae_sp: bool = False
    vae_scale_factor: Optional[int] = None

    # DiT configuration
    num_channels_latents: Optional[int] = None

    # Image encoder configuration
    image_encoder_precision: str = "fp32"

    # Text encoder configuration
    text_encoder_precision: str = "fp16"
    text_len: int = 256
    hidden_state_skip_layer: int = 2

    # Secondary text encoder
    text_encoder_precision_2: str = "fp16"
    text_len_2: int = 77

    # Flow Matching parameters
    flow_solver: str = "euler"
    denoise_type: str = "flow"  # Deprecated. Will use scheduler_config.json

    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    enable_torch_compile: bool = False

    # Scheduler options
    scheduler_type: str = "euler"  # Deprecated. Will use the param in scheduler_config.json

    neg_prompt: Optional[str] = None
    num_videos: int = 1
    fps: int = 24
    use_cpu_offload: bool = False
    disable_autocast: bool = False

    # Logging
    log_level: str = "info"

    # Inference parameters
    image_path: Optional[str] = None
    prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"
    seed: int = 1024
    device_str: Optional[str] = None
    device = None

    def __post_init__(self):
        pass

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help=
            "The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--dit-weight",
            type=str,
            help="Path to the DiT model weights",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )

        # distributed_executor_backend
        parser.add_argument(
            "--distributed-executor-backend",
            type=str,
            choices=["mp"],
            default=FastVideoArgs.distributed_executor_backend,
            help="The distributed executor backend to use",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=FastVideoArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=FastVideoArgs.revision,
            help=
            "The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=FastVideoArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=FastVideoArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sequence-parallel-size",
            "--sp-size",
            type=int,
            default=FastVideoArgs.sp_size,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=FastVideoArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Video generation parameters
        parser.add_argument(
            "--height",
            type=int,
            default=FastVideoArgs.height,
            help="Height of generated video",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=FastVideoArgs.width,
            help="Width of generated video",
        )
        parser.add_argument(
            "--num-frames",
            type=int,
            default=FastVideoArgs.num_frames,
            help="Number of frames to generate",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=FastVideoArgs.num_inference_steps,
            help="Number of inference steps",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=FastVideoArgs.guidance_scale,
            help="Guidance scale for classifier-free guidance",
        )
        parser.add_argument(
            "--guidance-rescale",
            type=float,
            default=FastVideoArgs.guidance_rescale,
            help="Guidance rescale for classifier-free guidance",
        )
        parser.add_argument(
            "--embedded-cfg-scale",
            type=float,
            default=FastVideoArgs.embedded_cfg_scale,
            help="Embedded CFG scale",
        )
        parser.add_argument(
            "--flow-shift",
            "--shift",
            type=float,
            default=FastVideoArgs.flow_shift,
            help="Flow shift parameter",
        )
        parser.add_argument(
            "--output-type",
            type=str,
            default=FastVideoArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )

        parser.add_argument(
            "--precision",
            type=str,
            default=FastVideoArgs.precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for the model",
        )

        # VAE configuration
        parser.add_argument(
            "--vae-precision",
            type=str,
            default=FastVideoArgs.vae_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for VAE",
        )
        parser.add_argument(
            "--vae-tiling",
            action="store_true",
            default=FastVideoArgs.vae_tiling,
            help="Enable VAE tiling",
        )
        parser.add_argument(
            "--vae-sp",
            action="store_true",
            help="Enable VAE spatial parallelism",
        )

        parser.add_argument(
            "--text-encoder-precision",
            type=str,
            default=FastVideoArgs.text_encoder_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for text encoder",
        )
        parser.add_argument(
            "--text-len",
            type=int,
            default=FastVideoArgs.text_len,
            help="Maximum text length",
        )

        # Image encoder config
        parser.add_argument(
            "--image-encoder-precision",
            type=str,
            default=FastVideoArgs.image_encoder_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for image encoder",
        )

        # Secondary text encoder

        parser.add_argument(
            "--text-encoder-precision-2",
            type=str,
            default=FastVideoArgs.text_encoder_precision_2,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for secondary text encoder",
        )
        parser.add_argument(
            "--text-len-2",
            type=int,
            default=FastVideoArgs.text_len_2,
            help="Maximum secondary text length",
        )

        # Flow Matching parameters
        parser.add_argument(
            "--flow-solver",
            type=str,
            default=FastVideoArgs.flow_solver,
            help="Solver for flow matching",
        )
        parser.add_argument(
            "--denoise-type",
            type=str,
            default=FastVideoArgs.denoise_type,
            help="Denoise type for noised inputs",
        )

        # STA (Spatial-Temporal Attention) parameters
        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help=
            "Use torch.compile for speeding up STA inference without teacache",
        )

        # Scheduler options
        parser.add_argument(
            "--scheduler-type",
            type=str,
            default=FastVideoArgs.scheduler_type,
            help="Type of scheduler to use",
        )

        # HunYuan specific parameters
        parser.add_argument(
            "--neg-prompt",
            type=str,
            default=FastVideoArgs.neg_prompt,
            help="Negative prompt for sampling",
        )
        parser.add_argument(
            "--num-videos",
            type=int,
            default=FastVideoArgs.num_videos,
            help="Number of videos to generate per prompt",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=FastVideoArgs.fps,
            help="Frames per second for output video",
        )
        parser.add_argument(
            "--use-cpu-offload",
            action="store_true",
            help="Use CPU offload for the model load",
        )
        parser.add_argument(
            "--disable-autocast",
            action="store_true",
            help=
            "Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=FastVideoArgs.log_level,
            help="The logging level of all loggers.",
        )

        # Inference parameters
        prompt_group = parser.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument(
            "--prompt",
            type=str,
            help="Text prompt for video generation",
        )
        prompt_group.add_argument(
            "--prompt-path",
            type=str,
            help="Path to a text file containing the prompt",
        )

        parser.add_argument("--image-path",
                            type=str,
                            help="Path to the image for I2V generation")

        parser.add_argument(
            "--output-path",
            type=str,
            default=FastVideoArgs.output_path,
            help="Directory to save generated videos",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=FastVideoArgs.seed,
            help="Random seed for reproducibility",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "FastVideoArgs":
        args.tp_size = args.tensor_parallel_size
        args.sp_size = args.sequence_parallel_size
        args.flow_shift = getattr(args, "shift", args.flow_shift)

        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs = {}
        for attr in attrs:
            # Handle renamed attributes or those with multiple CLI names
            if attr == 'tp_size' and hasattr(args, 'tensor_parallel_size'):
                kwargs[attr] = args.tensor_parallel_size
            elif attr == 'sp_size' and hasattr(args, 'sequence_parallel_size'):
                kwargs[attr] = args.sequence_parallel_size
            elif attr == 'flow_shift' and hasattr(args, 'shift'):
                kwargs[attr] = args.shift
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                default_value = getattr(cls, attr, None)
                kwargs[attr] = getattr(args, attr, default_value)

        return cls(**kwargs)

    def check_fastvideo_args(self) -> None:
        """Validate inference arguments for consistency"""
        if self.tp_size is None:
            self.tp_size = self.num_gpus
        if self.sp_size is None:
            self.sp_size = self.num_gpus

        if self.num_gpus < max(self.tp_size, self.sp_size):
            self.num_gpus = max(self.tp_size, self.sp_size)

        if self.tp_size != self.sp_size:
            raise ValueError(
                f"tp_size ({self.tp_size}) must be equal to sp_size ({self.sp_size})"
            )

        # Validate VAE spatial parallelism with VAE tiling
        if self.vae_sp and not self.vae_tiling:
            raise ValueError(
                "Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True."
            )
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a text file")


_current_fastvideo_args = None


def prepare_fastvideo_args(argv: List[str]) -> FastVideoArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    FastVideoArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    fastvideo_args = FastVideoArgs.from_cli_args(raw_args)
    fastvideo_args.check_fastvideo_args()
    global _current_fastvideo_args
    _current_fastvideo_args = fastvideo_args
    return fastvideo_args


@contextmanager
def set_current_fastvideo_args(fastvideo_args: FastVideoArgs):
    """
    Temporarily set the current fastvideo config.
    Used during model initialization.
    We save the current fastvideo config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the fastvideo config to determine how to dispatch.
    """
    global _current_fastvideo_args
    old_fastvideo_args = _current_fastvideo_args
    try:
        _current_fastvideo_args = fastvideo_args
        yield
    finally:
        _current_fastvideo_args = old_fastvideo_args


def get_current_fastvideo_args() -> FastVideoArgs:
    if _current_fastvideo_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the fastvideo config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current fastvideo args is not set.")
    return _current_fastvideo_args
