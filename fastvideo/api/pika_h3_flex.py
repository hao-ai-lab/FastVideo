# SPDX-License-Identifier: Apache-2.0
"""Sampling defaults for Pika H3 Flex."""

from dataclasses import dataclass

from fastvideo.api.sampling_param import SamplingParam


@dataclass
class PikaH3FlexSamplingParam(SamplingParam):
    """Six re-noise forwards, no guidance, and the 768p bucket by default."""

    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    batch_cfg: bool = False
    negative_prompt: str = ""
    height: int = 768
    width: int = 1344
    num_frames: int = 124
    fps: int = 24
    seed: int = 0


__all__ = ["PikaH3FlexSamplingParam"]
