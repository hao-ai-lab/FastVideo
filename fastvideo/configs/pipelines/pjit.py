# SPDX-License-Identifier: Apache-2.0
"""Single-image inference with Linum's pixel-space Pyramid-JiT checkpoint."""

from dataclasses import dataclass, field
from typing import ClassVar

from fastvideo.configs.models.dits.pjit import PJiTConfig
from fastvideo.configs.models.encoders.qwen3_5 import Qwen3_5TextConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class PJiTPipelineConfig(PipelineConfig):
    default_workload_type: ClassVar[str] = "t2i"
    dit_config: PJiTConfig = field(default_factory=PJiTConfig)
    dit_precision: str = "fp32"
    text_encoder_configs: tuple[Qwen3_5TextConfig, ...] = field(default_factory=lambda: (Qwen3_5TextConfig(), ))
    text_encoder_precisions: tuple[str, ...] = ("bf16", )
    qwen_model_path: str = "Qwen/Qwen3.5-4B"
    qwen_revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    vae_tiling: bool = False
    vae_sp: bool = False
    scheduler_step_in_fp32: bool = True
    # Linum's published CUDA noise draw uses the H100 SXM's 132 SMs.
    noise_sm_count: int | None = 132
    noise_scale: float = 4.0
    apg_momentum: float = -0.75
    apg_eta: float = 0.0
    apg_rescale: float = 10.0
