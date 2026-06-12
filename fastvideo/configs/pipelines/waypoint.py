# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Waypoint-1-Small world model."""

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.waypoint_transformer import WaypointConfig
from fastvideo.configs.models.encoders import T5Config
from fastvideo.configs.models.vaes import WorldEngineVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class WaypointT2VConfig(PipelineConfig):
    """Configuration for Waypoint-1-Small text-to-video world model.
    
    Waypoint is an interactive world model that generates video frames
    conditioned on text prompts and controller inputs (mouse, keyboard, scroll).
    """

    # DiT configuration
    dit_config: DiTConfig = field(default_factory=WaypointConfig)

    # WorldEngineVAE: native DCAE VAE, loaded through the standard VAELoader path.
    vae_config: VAEConfig = field(default_factory=WorldEngineVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Text encoding - uses UMT5-XL (loaded from model subfolder). Waypoint encodes
    # the prompt through its own WaypointTextEncodingStage (the streaming path),
    # so the base postprocess_text_funcs is left at its default and never invoked.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))

    # Precision settings
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))

    # Waypoint-specific settings
    # Fixed sigma schedule (no flow shift). Match official Overworld transformer config.
    # 5 sigmas = 4 denoising steps (official: Overworld/Waypoint-1-Small transformer config.json).
    flow_shift: float | None = None
    scheduler_sigmas: list[float] = field(default_factory=lambda: [
        1.0,
        0.8609585762023926,
        0.729332447052002,
        0.3205108940601349,
        0.0,
    ])

    # Interactive generation settings
    is_causal: bool = True
    base_fps: int = 60

    # Control input settings
    n_buttons: int = 256

    # Faithful-to-official denoise: the official does no latent renormalization
    # and accumulates the rectified-flow ODE in bf16. The original port added a
    # smooth std-normalization + fp32 accumulation to stabilize the (then-buggy)
    # flat KV cache; with the correct per-layer cache they hurt fidelity.
    disable_latent_norm: bool = True
    bf16_denoise: bool = True

    def __post_init__(self):
        # Waypoint doesn't use standard VAE loading
        pass
