# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Waypoint-1-Small world model."""

from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class WaypointSamplingParam(SamplingParam):
    """Sampling parameters for Waypoint-1-Small interactive world model.
    
    Waypoint generates 360p video at 60fps with control inputs.
    """
    
    # Video parameters (native resolution)
    height: int = 360
    width: int = 640
    num_frames: int = 240  # 4 seconds at 60fps
    fps: int = 60
    
    # Denoising parameters
    # Waypoint uses a fixed 4-step schedule
    num_inference_steps: int = 4
    guidance_scale: float = 1.0  # Typically no classifier-free guidance needed
    
    # No negative prompt for world models
    negative_prompt: str | None = None
    
    # Waypoint-specific parameters
    # Control inputs are passed separately during generation

