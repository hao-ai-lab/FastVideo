from dataclasses import dataclass

from fastvideo.v1.configs.sample.base import SamplingParam


@dataclass
class FluxSamplingParam(SamplingParam):
    # Video parameters
    height: int = 1024
    width: int = 1024
    num_frames: int = 1

    # Denoising stage
    num_inference_steps: int = 50