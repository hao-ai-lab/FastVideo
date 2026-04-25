# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.api.sampling_param import SamplingParam


@dataclass
class DaVinciMagiHumanSamplingParam(SamplingParam):
    """Default sampling parameters for daVinci-MagiHuman T2V."""
    # 720p at 4s (temporal stride 4 → 33 latent frames = ~4.1s at 8fps latent)
    height: int = 720
    width: int = 1280
    num_frames: int = 33   # latent frames (output frames = 33*4 - 3 ≈ 129)
    fps: int = 24
    seed: int = 0

    # daVinci paper uses guidance_scale=7.0 for T2V
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
