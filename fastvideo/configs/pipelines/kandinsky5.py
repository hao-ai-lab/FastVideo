# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import Kandinsky5VideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, CLIPTextConfig
from fastvideo.configs.models.encoders.reason1 import Reason1Config
from fastvideo.configs.models.vaes import HunyuanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text

# Byte-exact copy of the upstream Kandinsky5/diffusers template, including the
# "promt"/"scren" typos: the checkpoints were trained with this exact system  # codespell:ignore promt,scren
# prompt, and ENCODE_START_IDX below is the tokenized length of everything
# before the user prompt. Fixing the typos shifts user content to index 127
# and mis-conditions every generation.
KANDINSKY5_PROMPT_TEMPLATE = "\n".join([
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",  # codespell:ignore promt
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
    "Describe the location of the video, main characters or objects and their action.",
    "Describe the dynamism of the video and presented actions.",
    "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",  # codespell:ignore scren
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
    "Pay attention to the order of key actions shown in the scene.<|im_end|>",
    "<|im_start|>user\n{}<|im_end|>",
])
KANDINSKY5_PROMPT_TEMPLATE_ENCODE_START_IDX = 129


def kandinsky5_qwen_preprocess_text(prompt: str) -> str:
    if not prompt.strip():
        prompt = "."
    return KANDINSKY5_PROMPT_TEMPLATE.format(prompt)


def kandinsky5_qwen_postprocess_text(outputs: BaseEncoderOutput,
                                     mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if outputs.hidden_states is None:
        raise RuntimeError("Kandinsky5 Qwen prompt embeddings require hidden_states.")
    hidden_states = outputs.hidden_states[-1]
    prompt_embeds = hidden_states[:, KANDINSKY5_PROMPT_TEMPLATE_ENCODE_START_IDX:]
    mask = mask[:, KANDINSKY5_PROMPT_TEMPLATE_ENCODE_START_IDX:]
    if prompt_embeds.shape[1] == 0:
        prompt_embeds = hidden_states[:, -1:]
        mask = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
    return prompt_embeds, mask


def kandinsky5_clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    if outputs.pooler_output is None:
        raise RuntimeError("Kandinsky5 CLIP pooled output is required.")
    return outputs.pooler_output


@dataclass
class Kandinsky5T2VConfig(PipelineConfig):
    """Kandinsky-5.0 Lite text-to-video pipeline configuration."""

    dit_config: DiTConfig = field(default_factory=Kandinsky5VideoConfig)
    vae_config: VAEConfig = field(default_factory=HunyuanVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (Reason1Config(), CLIPTextConfig()))
    preprocess_text_funcs: tuple[Callable[[str], Any], ...] = field(
        default_factory=lambda: (kandinsky5_qwen_preprocess_text, preprocess_text))
    postprocess_text_funcs: tuple[Callable[..., Any], ...] = field(
        default_factory=lambda: (kandinsky5_qwen_postprocess_text, kandinsky5_clip_postprocess_text))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", "bf16"))
    text_encoder_max_lengths: tuple[int, ...] = field(
        default_factory=lambda: (KANDINSKY5_PROMPT_TEMPLATE_ENCODE_START_IDX + 512, 77))

    flow_shift: float | None = 5.0
    vae_tiling: bool = True

    def __post_init__(self) -> None:
        if len(self.text_encoder_configs) != 2:
            raise ValueError(f"Kandinsky5 pipeline requires exactly 2 text encoders (qwen and clip), "
                             f"but got {len(self.text_encoder_configs)} encoder(s).")

        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

        qwen_cfg = self.text_encoder_configs[0]
        qwen_cfg.arch_config.output_hidden_states = True
        qwen_cfg.arch_config.tokenizer_kwargs.update({
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        })

        clip_cfg = self.text_encoder_configs[1]
        clip_cfg.arch_config.tokenizer_kwargs.update({
            "padding": "max_length",
            "max_length": 77,
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt",
        })


@dataclass
class Kandinsky5I2VConfig(Kandinsky5T2VConfig):
    """Kandinsky-5.0 image-to-video pipeline configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # I2V needs the VAE encoder to encode the conditioning image.
        self.vae_config.load_encoder = True
