# SPDX-License-Identifier: Apache-2.0
"""
Sampling parameters for HunyuanGameCraft video generation.

GameCraft generates game-like videos with camera/action control.
Default parameters are based on the official implementation.
"""
from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanGameCraftSamplingParam(SamplingParam):
    """Sampling parameters for HunyuanGameCraft video generation.
    
    Supports camera/action conditioning via:
    - camera_trajectory: Plücker coordinates for camera motion
    - action_list: List of actions (e.g., ["forward", "left", "right"])
    - action_speed_list: Speed multipliers for each action
    
    Default resolution is 704x1280 (same as HunyuanVideo).
    Default frame count is 33 video frames -> 9 latent frames.
    """
    
    # Number of denoising steps
    num_inference_steps: int = 50
    
    # Video dimensions
    # 33 video frames -> 9 latent frames (4x temporal compression)
    num_frames: int = 33
    height: int = 704
    width: int = 1280
    fps: int = 24
    
    # Guidance scale - official GameCraft uses CFG with guidance_scale=6.0
    guidance_scale: float = 6.0
    
    # Negative prompt for CFG (empty string = unconditional)
    negative_prompt: str = ""
    
    # Camera/Action conditioning
    # Camera states as Plücker coordinates [B, T_video, 6, H, W]
    camera_states: Any | None = None
    
    # Camera trajectory file/identifier (alternative to camera_states)
    camera_trajectory: str | None = None
    
    # Action list for camera motion (e.g., ["forward", "left"])
    action_list: list[str] | None = None
    
    # Speed multipliers for each action
    action_speed_list: list[float] | None = None
    
    # History frame conditioning (for autoregressive generation)
    # Ground truth latents for conditioning [B, 16, T, H, W]
    gt_latents: Any | None = None
    
    # Mask for conditioning (1=use gt, 0=generate) [B, 1, T, H, W]
    conditioning_mask: Any | None = None
    
    # Number of conditioning frames (for autoregressive) - maps to num_cond_frames
    num_cond_frames: int = 0
    
    # TeaCache parameters (if enabled)
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                7.33226126e+02, -4.01131952e+02, 6.75869174e+01,
                -3.14987800e+00, 9.61237896e-02
            ],
        )
    )
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Validate action lists
        if self.action_list is not None and self.action_speed_list is not None:
            if len(self.action_list) != len(self.action_speed_list):
                raise ValueError(
                    f"action_list length ({len(self.action_list)}) must match "
                    f"action_speed_list length ({len(self.action_speed_list)})"
                )


@dataclass
class HunyuanGameCraft65FrameSamplingParam(HunyuanGameCraftSamplingParam):
    """Sampling parameters for 65-frame GameCraft generation.
    
    65 video frames -> 17 latent frames (with first frame as key frame).
    This is useful for longer video generation.
    """
    num_frames: int = 65


@dataclass  
class HunyuanGameCraft129FrameSamplingParam(HunyuanGameCraftSamplingParam):
    """Sampling parameters for 129-frame GameCraft generation.
    
    129 video frames -> 33 latent frames.
    This is the maximum supported by the official implementation.
    """
    num_frames: int = 129
