from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class BaseConfig:
    """Base configuration for all pipeline architectures."""

    # Video parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 125
    fps: int = 24

    # Inference parameters
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    seed: int = 1024

    # Model configuration
    num_gpus: int = 1
    sp_size: Optional[int] = None  # defaults to num_gpus
    tp_size: Optional[int] = None  # defaults to num_gpus
    vae_sp: bool = True

    # Output configuration
    output_path: str = "output_videos/"

    # Additional parameters can be added as a dict
    extra_params: Dict[str, Any] = field(default_factory=dict)


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
