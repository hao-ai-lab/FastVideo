# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from fastvideo.logger import init_logger

if TYPE_CHECKING:
    from fastvideo.api.schema import ContinuationState

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
    last_image: Any | None = None
    references: list[Any] | None = None

    # Video inputs
    video_path: str | None = None

    # Optional pre-generated diffusion latents. Used by parity/debug harnesses
    # and advanced callers that need deterministic latent reuse.
    latents: Any | None = None
    audio_latents: Any | None = None

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

    # Camera control inputs (LingBotWorld and LingBotWorld2)
    c2ws_plucker_emb: Any | None = None  # Plucker embedding: [B, C, F_lat, H_lat, W_lat]
    action_path: str | None = None  # Directory containing poses.npy and intrinsics.npy

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
    max_sequence_length: int | None = None
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
    batch_cfg: bool = False
    guidance_scale_2: float | None = None
    # Z-Image CFG controls. ``cfg_normalization=True`` caps the guided
    # prediction norm at the positive-prediction norm; ``cfg_truncation``
    # disables CFG above the normalized-noise threshold.
    cfg_normalization: bool = False
    cfg_truncation: float | None = 1.0
    # Embedded guidance (FLUX): do not treat ``guidance_scale > 1`` as classic CFG.
    use_embedded_guidance: bool = False
    # Per-request embedded guidance scale for guidance-distilled DiTs; None uses
    # pipeline_config.embedded_cfg_scale.
    embedded_cfg_scale: float | None = None
    # Diffusers-style true CFG for FLUX when > 1 (requires negative prompt encoding).
    true_cfg_scale: float = 1.0
    guidance_rescale: float = 0.0
    boundary_ratio: float | None = None
    sigmas: list[float] | None = None

    # TeaCache parameters
    enable_teacache: bool = False

    # GEN3C camera control
    trajectory_type: str | None = None
    movement_distance: float | None = None
    camera_rotation: str | None = None

    # LTX-2 multi-modal CFG and STG.
    # Class-level defaults match the *distilled* LTX-2 schedule
    # (mirrors ``FastVideo-internal/.../LTX2DistilledSamplingParam``):
    # the distilled model expects neutral guidance scales — modality 1,
    # rescale 0, STG 0 — and explicit-CFG callers (full LTX-2) opt back
    # in by selecting the ``LTX2_BASE`` preset, which overrides these
    # to mod=3.0 / rescale=0.7 / stg=1.0 in its ``defaults`` dict.
    # cfg_scale defaults stay at 1.0 (CFG off) so
    # ``ForwardBatch.__post_init__`` doesn't force CFG on non-LTX-2
    # models that never override these fields.
    ltx2_cfg_scale_video: float = 1.0
    ltx2_cfg_scale_audio: float = 1.0
    ltx2_modality_scale_video: float = 1.0
    ltx2_modality_scale_audio: float = 1.0
    ltx2_rescale_scale: float = 0.0
    ltx2_stg_scale_video: float = 0.0
    ltx2_stg_scale_audio: float = 0.0
    ltx2_stg_blocks_video: list[int] = field(default_factory=lambda: [29])
    ltx2_stg_blocks_audio: list[int] = field(default_factory=lambda: [29])

    # LTX-2 image / video / continuation conditioning. These flow from
    # request ``extensions`` through ``sampling_param.update(...)``
    # onto the ForwardBatch fields of the same name. ``ltx2_image_crf``
    # gates the conditioning-image H.264 re-encode; the streaming
    # session controller passes ``ltx2_image_crf=0.0`` because it
    # conditions on already-decoded VAE-quality frames.
    ltx2_images: list[tuple[str, int, float]] | None = None
    ltx2_image_crf: float = 33.0
    ltx2_conditioning_latent_stage1: Any | None = None
    ltx2_conditioning_latent_stage2: Any | None = None
    ltx2_video_conditions: list[tuple[list[str], int, float]] | None = None

    # Stable Audio (T2A): clip start/end in seconds. Honored by
    # `StableAudioConditioningStage` + `StableAudioDecodingStage`. Other
    # families ignore them.
    audio_start_in_s: float | None = None
    audio_end_in_s: float | None = None

    # Stable Audio audio-to-audio (variation):
    #   `init_audio` -- a path or `[B, C, samples]` waveform at the model
    #                   sample rate; the pipeline encodes it via the VAE
    #                   and uses it as the starting latent.
    #   `init_audio_strength` -- 0..1, higher = closer to the reference
    #                            (matches the convention of Stability's
    #                            commercial Stable Audio 2.0 UI). 1.0 ~=
    #                            VAE round-trip, 0.0 ~= plain T2A.
    #   `init_noise_level` -- legacy raw `sigma_max` override (0.3..500,
    #                         higher = more freedom). Kept for callers
    #                         that already use it; prefer `init_audio_strength`.
    init_audio: Any = None
    init_audio_strength: float | None = None
    init_noise_level: float | None = None

    # Stable Audio inpainting (RePaint-style): `inpaint_audio` is the
    # reference clip, `inpaint_mask` is a [samples] tensor in {0, 1} where
    # 1 means *keep the reference* and 0 means *regenerate*.
    inpaint_audio: Any = None
    inpaint_mask: Any = None

    # Continuation state carried across streaming/multi-segment calls.
    continuation_state: ContinuationState | None = None
    # When True, the pipeline returns a ContinuationState on the result so
    # the caller can resume from the generated segment.
    return_continuation_state: bool = False

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
        unknown = [key for key in source_dict if key not in valid_fields]
        if unknown:
            raise ValueError(f"{type(self).__name__}.update() received unknown field(s): "
                             f"{sorted(unknown)}. All kwargs must correspond to declared "
                             f"SamplingParam fields. If a kwarg is meant to flow into "
                             f"ForwardBatch.extra (e.g. LTX2 audio conditioning), route it "
                             f"via VideoGenerator._BATCH_EXTRA_PASSTHROUGH_KEYS instead.")
        for key, value in source_dict.items():
            setattr(self, key, value)

        self.__post_init__()

    @classmethod
    def from_pretrained(cls, model_path: str) -> SamplingParam:
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
    ) -> SamplingParam | None:
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


@dataclass
class CacheParams:
    cache_type: str = "none"
