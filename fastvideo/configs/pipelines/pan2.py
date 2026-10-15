# SPDX-License-Identifier: Apache-2.0
"""PAN2 pipeline configuration: text-to-video, and image-to-video from a first frame."""

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import PAN2VideoConfig
from fastvideo.configs.models.encoders import Qwen3_5TextConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.models.wan.vae_config import WanVAEConfig

PAN2_SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."


@dataclass
class PAN2PipelineConfig(PipelineConfig):
    """One configuration serves text-to-video and, given a first frame, image-to-video."""

    dit_config: DiTConfig = field(default_factory=PAN2VideoConfig)
    # The Wan2.2 48-channel VAE; its architecture and latent statistics come from vae/config.json at load time.
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (Qwen3_5TextConfig(), ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    flow_shift: float | None = 7.0
    # PAN2 samples on fp32 latents; keep the Euler update out of autocast.
    scheduler_step_in_fp32: bool = True

    # Prompt encoding: PAN2 conditions on the tokens after the chat template's user header, taken from the text
    # encoder's hidden states with its last two layers skipped (``hidden_states[-3]``).
    text_system_prompt: str = PAN2_SYSTEM_PROMPT
    text_encoder_skip_layers: int = 2
    text_max_sequence_length: int = 1536

    def __post_init__(self) -> None:
        # The encoder is needed for the image-to-video first frame.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.text_encoder_configs[0].arch_config.output_hidden_states = True
