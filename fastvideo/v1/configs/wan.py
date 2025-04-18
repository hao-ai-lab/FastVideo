from dataclasses import dataclass

from fastvideo.v1.configs.base import BaseConfig


@dataclass
class WanT2V480PConfig(BaseConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    use_cpu_offload: bool = True

    # Denoising stage
    guidance_scale: float = 3.0
    neg_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    flow_shift: int = 3
    num_inference_steps: int = 50

    # Text encoding stage
    text_len: int = 512

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precision: str = "fp32"

    # WanConfig-specific added parameters


@dataclass
class WanI2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 40

    # Precision for each component
    image_encoder_precision: str = "fp32"
