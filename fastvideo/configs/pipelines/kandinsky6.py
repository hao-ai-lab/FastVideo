# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, ModelConfig, VAEConfig
from fastvideo.configs.models.audio import BigVGANV2Config, Kandinsky6AudioVAEConfig
from fastvideo.configs.models.dits import Kandinsky6VideoAudioConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, CLIPTextConfig
from fastvideo.configs.models.encoders.reason1 import Reason1Config
from fastvideo.configs.models.vaes import HunyuanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text

# Same Qwen2.5-VL "prompt engineer" system template and 129-token crop as
# Kandinsky5 (fastvideo/configs/pipelines/kandinsky5.py) -- Kandinsky6 reuses
# the identical text-conditioning stack (Qwen2.5-VL token embeddings + CLIP
# pooled embedding), byte-exact including its two misspelled words: the
# checkpoints were trained with this exact system prompt, and
# ENCODE_START_IDX is the tokenized length of everything before the user
# prompt. Kept as an independent copy (not imported from kandinsky5.py) so
# the two model families' pipeline configs stay self-contained.
KANDINSKY6_PROMPT_TEMPLATE = "\n".join([
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",  # codespell:ignore promt
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
    "Describe the location of the video, main characters or objects and their action.",
    "Describe the dynamism of the video and presented actions.",
    "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",  # codespell:ignore scren
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
    "Pay attention to the order of key actions shown in the scene.<|im_end|>",
    "<|im_start|>user\n{}<|im_end|>",
])
KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX = 129


def kandinsky6_qwen_preprocess_text(prompt: str) -> str:
    # The diffusers reference encodes an empty prompt as an empty user turn
    # (the template's system half is never empty, so the Qwen tokenizer
    # never sees zero tokens) -- unlike the generic empty-prompt guard
    # (`text_encoding.py`'s `treat_empty_as_dot`, not opted into here),
    # substituting "." would tokenize differently and drift from the
    # reference's embeddings for prompt="".
    return KANDINSKY6_PROMPT_TEMPLATE.format(prompt)


def kandinsky6_qwen_postprocess_text(outputs: BaseEncoderOutput,
                                     mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if outputs.hidden_states is None:
        raise RuntimeError("Kandinsky6 Qwen prompt embeddings require hidden_states.")
    hidden_states = outputs.hidden_states[-1]
    prompt_embeds = hidden_states[:, KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX:]
    mask = mask[:, KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX:]
    if prompt_embeds.shape[1] == 0:
        prompt_embeds = hidden_states[:, -1:]
        mask = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
    return prompt_embeds, mask


def kandinsky6_clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    if outputs.pooler_output is None:
        raise RuntimeError("Kandinsky6 CLIP pooled output is required.")
    return outputs.pooler_output


@dataclass
class Kandinsky6TI2VAConfig(PipelineConfig):
    """Kandinsky6 TI2VA pipeline configuration.

    One pipeline (fastvideo.pipelines.basic.kandinsky6.Kandinsky6TI2VAPipeline)
    serves text-to-video+audio generation with an optional conditioning image:
    pure text when none is supplied at generation time, image+text (called
    "I2VA" in the diffusers reference) when one is -- mirroring the diffusers
    reference, which likewise has a single Kandinsky6TI2VAPipeline class
    (pipeline_kandinsky6_ti2va.py) taking an optional image argument, not a
    separate T2VA/I2VA subclass pair. See
    fastvideo/pipelines/stages/kandinsky6.py's Kandinsky6ImageEncodingStage.
    """

    dit_config: DiTConfig = field(default_factory=Kandinsky6VideoAudioConfig)
    vae_config: VAEConfig = field(default_factory=HunyuanVAEConfig)
    # Two separate checkpoint components, like LTX-2's audio_vae/vocoder pair: audio_vae_config (the
    # mel-VAE decoder, a thin nesting wrapper around FastVideo's existing MMAudioVAE) and
    # vocoder_config (BigVGAN-v2, reused unmodified from the existing mmaudio pipeline).
    audio_vae_config: ModelConfig = field(default_factory=Kandinsky6AudioVAEConfig)
    vocoder_config: ModelConfig = field(default_factory=BigVGANV2Config)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (Reason1Config(), CLIPTextConfig()))
    preprocess_text_funcs: tuple[Callable[[str], Any], ...] = field(
        default_factory=lambda: (kandinsky6_qwen_preprocess_text, preprocess_text))
    postprocess_text_funcs: tuple[Callable[..., Any], ...] = field(
        default_factory=lambda: (kandinsky6_qwen_postprocess_text, kandinsky6_clip_postprocess_text))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", "bf16"))
    # Diffusers default: 1024 user tokens for Qwen (max_length = msl + 129,
    # the template-prefix crop), 77 for CLIP (fixed, never overridden by a
    # request -- see Kandinsky6TextEncodingStage._resolve_max_length).
    text_encoder_max_lengths: tuple[int, ...] = field(
        default_factory=lambda: (KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX + 1024, 77))

    # Matches both official scheduler_config.json files (base and distilled
    # alike): shift 5.0; SchedulerLoader also applies this to a checkpoint's
    # own scheduler_config.json via set_shift(), so it is only a fallback for
    # a config missing the key.
    flow_shift: float | None = 5.0
    vae_tiling: bool = True

    # Scheduler constructor overrides; None keeps scheduler_config.json values.
    piflow_eps: float | None = None
    piflow_final_step_size_scale: float | None = None
    piflow_num_policy_substeps: int | None = None

    # Audio<->video latent-length alignment, matching the diffusers
    # reference's Kandinsky6TI2VAPipeline defaults exactly:
    # audio_latent_frames = ceil(((T_lat-1)*4+1) / sample_fps * audio_sample_rate / audio_downsample_factor).
    sample_fps: float = 24.0
    audio_sample_rate: int = 44100
    audio_downsample_factor: int = 1024

    def __post_init__(self) -> None:
        if len(self.text_encoder_configs) != 2:
            raise ValueError(f"Kandinsky6 pipeline requires exactly 2 text encoders (qwen and clip), "
                             f"but got {len(self.text_encoder_configs)} encoder(s).")
        if len(self.text_encoder_precisions) != 2:
            raise ValueError("Kandinsky6 pipeline requires exactly 2 text encoder precisions, "
                             f"but got {len(self.text_encoder_precisions)}.")
        if len(self.text_encoder_max_lengths) != 2:
            raise ValueError("Kandinsky6 pipeline requires exactly 2 text encoder max lengths, "
                             f"but got {len(self.text_encoder_max_lengths)}.")

        # The merged pipeline can receive an optional conditioning image on
        # any given call, so the VAE encoder must always be available (unlike
        # Kandinsky5, which only flips this on for the dedicated I2V config).
        self.vae_config.load_encoder = True
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
