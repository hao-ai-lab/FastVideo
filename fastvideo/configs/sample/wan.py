# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class WanT2V_1_3B_SamplingParam(SamplingParam):
    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 3.0
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_inference_steps: int = 50


@dataclass
class WanT2V_14B_SamplingParam(SamplingParam):
    # Video parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 5.0
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_inference_steps: int = 50


@dataclass
class WanI2V_14B_480P_SamplingParam(WanT2V_1_3B_SamplingParam):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 40


@dataclass
class WanI2V_14B_720P_SamplingParam(WanT2V_14B_SamplingParam):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 40


@dataclass
class FastWanT2V480P_SamplingParam(WanT2V_1_3B_SamplingParam):
    # DMD parameters
    # dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 757, 522])
    num_inference_steps: int = 3
    num_frames: int = 61
    height: int = 448
    width: int = 832
    fps: int = 16


# =============================================
# ============= Wan2.1 Fun Models =============
# =============================================
@dataclass
class Wan2_1_Fun_1_3B_InP_SamplingParam(SamplingParam):
    """Sampling parameters for Wan2.1 Fun 1.3B InP model."""
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    negative_prompt: str | None = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    guidance_scale: float = 6.0
    num_inference_steps: int = 50


@dataclass
class Wan2_1_Fun_1_3B_Control_SamplingParam(SamplingParam):
    fps: int = 16
    num_frames: int = 49
    height: int = 832
    width: int = 480
    guidance_scale: float = 6.0


# =============================================
# ============= Wan2.2 TI2V Models =============
# =============================================
@dataclass
class Wan2_2_Base_SamplingParam(SamplingParam):
    """Sampling parameters for Wan2.2 TI2V 5B model."""
    negative_prompt: str | None = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


@dataclass
class Wan2_2_TI2V_5B_SamplingParam(Wan2_2_Base_SamplingParam):
    """Sampling parameters for Wan2.2 TI2V 5B model."""
    height: int = 704
    width: int = 1280
    num_frames: int = 121
    fps: int = 24
    guidance_scale: float = 5.0
    num_inference_steps: int = 50


@dataclass
class Wan2_2_T2V_A14B_SamplingParam(Wan2_2_Base_SamplingParam):
    guidance_scale: float = 4.0  # high_noise
    guidance_scale_2: float = 3.0  # low_noise
    num_inference_steps: int = 40
    fps: int = 16
    # NOTE(will): default boundary timestep is tracked by PipelineConfig, but
    # can be overridden during sampling


@dataclass
class Wan2_2_I2V_A14B_SamplingParam(Wan2_2_Base_SamplingParam):
    guidance_scale: float = 3.5  # high_noise
    guidance_scale_2: float = 3.5  # low_noise
    num_inference_steps: int = 40
    fps: int = 16
    # NOTE(will): default boundary timestep is tracked by PipelineConfig, but
    # can be overridden during sampling


@dataclass
class Wan2_2_Fun_A14B_Control_SamplingParam(
        Wan2_1_Fun_1_3B_Control_SamplingParam):
    num_frames: int = 81


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam(
        Wan2_1_Fun_1_3B_InP_SamplingParam):
    pass


@dataclass
class SelfForcingWan2_2_T2V_A14B_480P_SamplingParam(
        Wan2_2_T2V_A14B_SamplingParam):
    num_inference_steps: int = 8
    num_frames: int = 81
    height: int = 448
    width: int = 832
    fps: int = 16


@dataclass
class MatrixGame2_SamplingParam(SamplingParam):
    height: int = 352
    width: int = 640
    num_frames: int = 57
    fps: int = 25
    guidance_scale: float = 1.0
    num_inference_steps: int = 3
    negative_prompt: str | None = None
