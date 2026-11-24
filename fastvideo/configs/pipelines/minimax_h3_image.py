# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax-H3-Image."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ImageConfig
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.configs.models.vaes.minimax_h3_image import MiniMaxH3ImageVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.pipelines.basic.minimax_h3_image.packing import MINIMAX_H3_IMAGE_MIN_GENERATION_AREA


def _text_encoder_configs() -> tuple[MiniMaxH3Qwen3VLConfig, ...]:
    # The video encoder default truncates at 1024. Image prompts truncate at 4096
    # in the conditioning stage; this keeps the component config in agreement
    # until a checkpoint field overwrites it.
    config = MiniMaxH3Qwen3VLConfig()
    config.arch_config.text_len = 4096
    return (config, )


@dataclass
class MiniMaxH3ImagePipelineConfig(PipelineConfig):
    """H3 image stack: same Qwen3-VL encoder, PDD DiT, FP32 image VAE."""

    dit_config: MiniMaxH3ImageConfig = field(default_factory=MiniMaxH3ImageConfig)
    vae_config: MiniMaxH3ImageVAEConfig = field(default_factory=MiniMaxH3ImageVAEConfig)
    text_encoder_configs: tuple[MiniMaxH3Qwen3VLConfig, ...] = field(default_factory=_text_encoder_configs)
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    # Images below this area are generated larger and resized back.
    min_generation_area: int = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.flow_shift is not None:
            raise ValueError("MiniMax-H3-Image uses the published sigma list; flow_shift must stay unset.")
        if self.min_generation_area <= 0:
            raise ValueError(f"min_generation_area must be positive, got {self.min_generation_area}.")
