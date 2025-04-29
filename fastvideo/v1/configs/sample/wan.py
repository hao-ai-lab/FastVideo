from dataclasses import dataclass

from fastvideo.v1.configs.sample.base import SamplingParam


@dataclass
class WanT2V480PSamplingParam(SamplingParam):
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
class WanI2V480PSamplingParam(WanT2V480PSamplingParam):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 40
