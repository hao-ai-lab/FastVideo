# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class LTXSamplingParam(SamplingParam):
    # Video parameters
    height: int = 512
    width: int = 704

    # Most defaults set in pipeline config
    num_inference_steps: int = 50
