# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.glm_image import GlmImageDiTConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes.glm_image import GlmImageVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def glm_image_t5_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()

    assert torch.isnan(hidden_state).sum() == 0, "T5 hidden states contain NaN"

    max_len = 512
    prompt_embeds = [u[:min(v, max_len)] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)

    return prompt_embeds_tensor


@dataclass
class GlmImageConfig(PipelineConfig):

    dit_config: DiTConfig = field(default_factory=GlmImageDiTConfig)
    dit_precision: str = "bf16"

    vae_config: VAEConfig = field(default_factory=GlmImageVAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = True
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (glm_image_t5_postprocess, ))

    flow_shift: float | None = 1.0
    embedded_cfg_scale: float = 7.5
