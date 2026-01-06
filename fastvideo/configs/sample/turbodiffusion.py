# SPDX-License-Identifier: Apache-2.0
"""
TurboDiffusion sampling parameters.

TurboDiffusion uses RCM (recurrent Consistency Model) scheduler for
1-4 step video generation with no classifier-free guidance.
"""
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class TurboDiffusionT2V_1_3B_SamplingParam(SamplingParam):
    """Sampling parameters for TurboDiffusion T2V 1.3B model.
    
    Uses 4-step RCM sampling with guidance_scale=1.0 (no CFG).
    """
    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Denoising stage - TurboDiffusion uses 1-4 steps with no CFG
    guidance_scale: float = 1.0
    num_inference_steps: int = 4

    # No negative prompt needed for TurboDiffusion (no CFG)
    negative_prompt: str | None = None


@dataclass
class TurboDiffusionT2V_14B_SamplingParam(SamplingParam):
    """Sampling parameters for TurboDiffusion T2V 14B model.
    
    Uses 4-step RCM sampling with guidance_scale=1.0 (no CFG).
    """
    # Video parameters (720p for 14B)
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16

    # Denoising stage - TurboDiffusion uses 1-4 steps with no CFG
    guidance_scale: float = 1.0
    num_inference_steps: int = 4

    # No negative prompt needed for TurboDiffusion (no CFG)
    negative_prompt: str | None = None


@dataclass
class TurboDiffusionI2V_A14B_SamplingParam(SamplingParam):
    """Sampling parameters for TurboDiffusion I2V A14B model.
    
    Uses 4-step RCM sampling with dual-model switching (high/low noise).
    """
    # Video parameters (720p for A14B I2V)
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16

    # Denoising stage - TurboDiffusion uses 1-4 steps with no CFG
    guidance_scale: float = 1.0
    num_inference_steps: int = 4

    # Note: boundary_ratio is set in the pipeline config (TurboDiffusionI2VConfig),
    # not here. This keeps sampling params and pipeline config separate.

    # No negative prompt needed for TurboDiffusion (no CFG)
    negative_prompt: str | None = None
