from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration for all pipeline architectures."""

    # Video parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 125
    fps: int = 24

    # Video generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 1024
    guidance_rescale: float = 0.0
    embedded_cfg_scale: float = 6.0
    flow_shift: Optional[float] = None
    use_cpu_offload: bool = False
    disable_autocast: bool = False

    # Model configuration
    precision: str = "bf16"

    # VAE configuration
    vae_precision: str = "fp16"
    vae_tiling: bool = True
    vae_sp: bool = True
    vae_scale_factor: Optional[int] = None

    # DiT configuration
    num_channels_latents: Optional[int] = None

    # Image encoder configuration
    image_encoder_precision: str = "fp32"

    # Text encoder configuration
    text_encoder_precision: str = "fp16"
    text_len: int = -1
    hidden_state_skip_layer: int = 0

    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    enable_torch_compile: bool = False

    neg_prompt: Optional[str] = None


@dataclass
class SlidingTileAttnConfig(BaseConfig):
    """Configuration for sliding tile attention."""

    # Override any BaseConfig defaults as needed
    # Add sliding tile specific parameters
    window_size: int = 16
    stride: int = 8

    # You can provide custom defaults for inherited fields
    height: int = 576
    width: int = 1024

    # Additional configuration specific to sliding tile attention
    pad_to_square: bool = False
    use_overlap_optimization: bool = True
