# SPDX-License-Identifier: Apache-2.0
import copy
from dataclasses import dataclass, field, fields
from typing import Any

from fastvideo.logger import init_logger
from fastvideo.utils import StoreBoolean

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
    pil_image: Any | None = None

    # Video inputs
    video_path: str | None = None

    # Action control inputs (Matrix-Game)
    mouse_cond: Any | None = None  # Shape: (B, T, 2)
    keyboard_cond: Any | None = None  # Shape: (B, T, K)
    grid_sizes: Any | None = None  # Shape: (3,) [F,H,W]

    # Camera control inputs (HYWorld)
    pose: str | None = None  # Camera trajectory: pose string (e.g., 'w-31') or JSON file path
    prompt_attention_mask: list = field(default_factory=list)
    negative_attention_mask: list = field(default_factory=list)

    # Camera/action control inputs (GameCraft)
    camera_states: Any | None = None  # Plücker coordinates [B, T_video, 6, H, W]
    camera_trajectory: str | None = None
    action_list: list[str] | None = None
    action_speed_list: list[float] | None = None
    gt_latents: Any | None = None  # Ground truth latents [B, 16, T, H, W]
    conditioning_mask: Any | None = None  # Mask [B, 1, T, H, W]

    # Camera control inputs (LingBotWorld)
    c2ws_plucker_emb: Any | None = None  # Plucker embedding: [B, C, F_lat, H_lat, W_lat]

    # Refine inputs (LongCat 480p->720p upscaling)
    # Path-based refine (load stage1 video from disk, e.g. MP4)
    refine_from: str | None = None  # Path to stage1 video (480p output from distill)
    t_thresh: float = 0.5  # Threshold for timestep scheduling in refinement
    spatial_refine_only: bool = False  # If True, only spatial (no temporal doubling)
    num_cond_frames: int = 0  # Number of conditioning frames
    # In-memory refine input (for two-stage pipeline where stage1 frames are already in memory)
    # This mirrors LongCat's demo where a list of frames (e.g. np.ndarray or PIL.Image)
    # is passed directly to the refinement pipeline instead of reloading from disk.
    stage1_video: Any | None = None

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
    height: int = 720
    width: int = 1280
    height_sr: int = 1072
    width_sr: int = 1920
    fps: int = 24

    # Denoising parameters
    num_inference_steps: int = 50
    num_inference_steps_sr: int = 50
    guidance_scale: float = 1.0
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    boundary_ratio: float | None = None
    sigmas: list[float] | None = None

    # TeaCache parameters
    enable_teacache: bool = False

    # GEN3C camera control
    trajectory_type: str | None = None
    movement_distance: float | None = None
    camera_rotation: str | None = None

    # LTX2 multi-modal CFG and STG.
    #
    # ``ltx2_cfg_scale_video`` / ``ltx2_cfg_scale_audio`` default to
    # ``1.0`` (CFG off) rather than the LTX-2-specific on-values so
    # non-LTX2 models don't silently trip the CFG-force in
    # ``ForwardBatch.__post_init__`` (``ltx2_cfg_scale_video != 1.0 or
    # ltx2_cfg_scale_audio != 1.0`` → ``do_classifier_free_guidance = True``).
    # LTX-2 presets that need text-CFG on (e.g. ``ltx2_base``) set these
    # explicitly in their ``defaults`` dict.
    #
    # The other LTX-2 multi-modal knobs (``ltx2_modality_scale_*``,
    # ``ltx2_rescale_scale``, ``ltx2_stg_*``) keep their LTX-2 on-values
    # as class defaults because they are only read by LTX-2 pipeline
    # stages — no cross-family leak path. They remain "model-specific
    # fields in shared schema" tech debt (see PR plan.md) to be migrated
    # to preset-owned overrides.
    ltx2_cfg_scale_video: float = 1.0
    ltx2_cfg_scale_audio: float = 1.0
    ltx2_modality_scale_video: float = 3.0
    ltx2_modality_scale_audio: float = 3.0
    ltx2_rescale_scale: float = 0.7
    ltx2_stg_scale_video: float = 1.0
    ltx2_stg_scale_audio: float = 1.0
    ltx2_stg_blocks_video: list[int] = field(default_factory=lambda: [29])
    ltx2_stg_blocks_audio: list[int] = field(default_factory=lambda: [29])

    # Misc
    save_video: bool = True
    return_frames: bool = True
    return_trajectory_latents: bool = False  # returns all latents for each timestep
    return_trajectory_decoded: bool = False  # returns decoded latents for each timestep

    def __post_init__(self) -> None:
        self.data_type = "video" if self.num_frames > 1 else "image"

    def check_sampling_param(self):
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def update(self, source_dict: dict[str, Any]) -> None:
        valid_fields = {f.name for f in fields(self)}
        for key, value in source_dict.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                logger.error("%s has no field %s", type(self).__name__, key)

        self.__post_init__()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SamplingParam":
        sampling_param = cls._from_preset(model_path)
        if sampling_param is not None:
            return sampling_param

        logger.warning(
            "Couldn't find a preset for %s."
            " Using the default sampling param.",
            model_path,
        )
        return cls()

    @classmethod
    def _from_preset(
        cls,
        model_path: str,
    ) -> "SamplingParam | None":
        """Build a SamplingParam from preset defaults.

        Returns ``None`` when no preset is configured for
        *model_path*, letting the caller fall back to the legacy
        subclass lookup.
        """
        from fastvideo.registry import get_preset_selection

        try:
            preset_name, model_family = get_preset_selection(model_path)
        except (ValueError, RuntimeError):
            return None
        if preset_name is None or model_family is None:
            return None

        from fastvideo.api.presets import get_preset

        preset = get_preset(preset_name, model_family)
        sp = cls()
        valid_fields = {f.name for f in fields(cls)}
        for key, value in preset.defaults.items():
            if key in valid_fields:
                setattr(sp, key, copy.deepcopy(value))
        sp.__post_init__()
        return sp

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
            default=False,
            help="Whether to return the raw frames",
        )
        parser.add_argument(
            "--image-path",
            type=str,
            default=SamplingParam.image_path,
            help="Path to input image for image-to-video generation",
        )
        parser.add_argument(
            "--video-path",
            type=str,
            default=SamplingParam.video_path,
            help="Path to input video for video-to-video generation",
        )
        parser.add_argument(
            "--refine-from",
            type=str,
            default=SamplingParam.refine_from,
            help="Path to stage1 video for refinement (LongCat 480p->720p)",
        )
        parser.add_argument(
            "--t-thresh",
            type=float,
            default=SamplingParam.t_thresh,
            help="Threshold for timestep scheduling in refinement (default: 0.5)",
        )
        parser.add_argument(
            "--spatial-refine-only",
            action=StoreBoolean,
            default=SamplingParam.spatial_refine_only,
            help="Only perform spatial super-resolution (no temporal doubling)",
        )
        parser.add_argument(
            "--num-cond-frames",
            type=int,
            default=SamplingParam.num_cond_frames,
            help="Number of conditioning frames for refinement",
        )
        parser.add_argument(
            "--moba-config-path",
            type=str,
            default=None,
            help="Path to a JSON file containing V-MoBA specific configurations.",
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
