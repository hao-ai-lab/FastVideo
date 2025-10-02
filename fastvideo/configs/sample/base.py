# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SamplingParam:
    """
    Sampling parameters for video generation.
    """
    # All fields below are copied from ForwardBatch
    data_type: str = "video"

    # Image inputs
    image_path: str | None = None

    # Video inputs
    video_path: str | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    prompt_path: str | None = None
    output_path: str = "outputs/"
    output_video_name: str | None = None

    # Batch info
    num_videos_per_prompt: int = 1
    seed: int = 1024

    # Original dimensions (before VAE scaling)
    num_frames: int = 125
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus
    height: int = 720
    width: int = 1280
    fps: int = 24

    # Denoising parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    boundary_ratio: float | None = None

    # TeaCache parameters
    enable_teacache: bool = False

    # Misc
    save_video: bool = True
    return_frames: bool = False
    return_trajectory_latents: bool = False  # returns all latents for each timestep
    return_trajectory_decoded: bool = False  # returns decoded latents for each timestep

    def __post_init__(self) -> None:
        self.data_type = "video" if self.num_frames > 1 else "image"

    def check_sampling_param(self):
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def update(self, source_dict: dict[str, Any]) -> None:
        for key, value in source_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.exception("%s has no attribute %s",
                                 type(self).__name__, key)

        self.__post_init__()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SamplingParam":
        from fastvideo.configs.sample.registry import (
            get_sampling_param_cls_for_name)
        sampling_cls = get_sampling_param_cls_for_name(model_path)
        if sampling_cls is not None:
            sampling_param: SamplingParam = sampling_cls()
        else:
            logger.warning(
                "Couldn't find an optimal sampling param for %s. Using the default sampling param.",
                model_path)
            sampling_param = cls()

        return sampling_param

    @staticmethod
    def add_cli_args(parser: Any) -> Any:
        """Add CLI arguments for SamplingParam fields"""
        parser.add_argument(
            "--prompt",
            type=str,
            default=SamplingParam.prompt,
            help="Text prompt for video generation",
        )
        parser.add_argument(
            "--negative-prompt",
            type=str,
            default=SamplingParam.negative_prompt,
            help="Negative text prompt for video generation",
        )
        parser.add_argument(
            "--prompt-path",
            type=str,
            default=SamplingParam.prompt_path,
            help="Path to a text file containing the prompt",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=SamplingParam.output_path,
            help="Path to save the generated video",
        )
        parser.add_argument(
            "--output-video-name",
            type=str,
            default=SamplingParam.output_video_name,
            help="Name of the output video",
        )
        parser.add_argument(
            "--num-videos-per-prompt",
            type=int,
            default=SamplingParam.num_videos_per_prompt,
            help="Number of videos to generate per prompt",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=SamplingParam.seed,
            help="Random seed for generation",
        )
        parser.add_argument(
            "--num-frames",
            type=int,
            default=SamplingParam.num_frames,
            help="Number of frames to generate",
        )
        parser.add_argument(
            "--height",
            type=int,
            default=SamplingParam.height,
            help="Height of generated video",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=SamplingParam.width,
            help="Width of generated video",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=SamplingParam.fps,
            help="Frames per second for saved video",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=SamplingParam.num_inference_steps,
            help="Number of denoising steps",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=SamplingParam.guidance_scale,
            help="Classifier-free guidance scale",
        )
        parser.add_argument(
            "--guidance-rescale",
            type=float,
            default=SamplingParam.guidance_rescale,
            help="Guidance rescale factor",
        )
        parser.add_argument(
            "--boundary-ratio",
            type=float,
            default=SamplingParam.boundary_ratio,
            help="Boundary timestep ratio",
        )
        parser.add_argument(
            "--save-video",
            action="store_true",
            default=SamplingParam.save_video,
            help="Whether to save the video to disk",
        )
        parser.add_argument(
            "--no-save-video",
            action="store_false",
            dest="save_video",
            help="Don't save the video to disk",
        )
        parser.add_argument(
            "--return-frames",
            action="store_true",
            default=SamplingParam.return_frames,
            help="Whether to return the raw frames",
        )
        parser.add_argument(
            "--image-path",
            type=str,
            default=SamplingParam.image_path,
            help="Path to input image for image-to-video generation",
        )
        parser.add_argument(
            "--video_path",
            type=str,
            default=SamplingParam.video_path,
            help="Path to input video for video-to-video generation",
        )
        parser.add_argument(
            "--moba-config-path",
            type=str,
            default=None,
            help=
            "Path to a JSON file containing V-MoBA specific configurations.",
        )
        parser.add_argument(
            "--return-trajectory-latents",
            action="store_true",
            default=SamplingParam.return_trajectory_latents,
            help="Whether to return the trajectory",
        )
        parser.add_argument(
            "--return-trajectory-decoded",
            action="store_true",
            default=SamplingParam.return_trajectory_decoded,
            help="Whether to return the decoded trajectory",
        )
        return parser


@dataclass
class CacheParams:
    cache_type: str = "none"
