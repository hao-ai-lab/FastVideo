# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.gen3c import Gen3CVideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5LargeConfig
from fastvideo.configs.models.vaes import CosmosVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_large_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess T5 Large text encoder outputs for GEN3C pipeline.
    
    Return raw last_hidden_state without truncation/padding.
    """
    hidden_state = outputs.last_hidden_state

    if hidden_state is None:
        raise ValueError("T5 Large outputs missing last_hidden_state")

    nan_count = torch.isnan(hidden_state).sum()
    if nan_count > 0:
        hidden_state = hidden_state.masked_fill(torch.isnan(hidden_state), 0.0)

    # Zero out embeddings beyond actual sequence length (vectorized)
    if outputs.attention_mask is not None:
        attention_mask = outputs.attention_mask
        lengths = attention_mask.sum(dim=1)
        max_len = hidden_state.shape[1]
        mask = torch.arange(
            max_len, device=hidden_state.device)[None, :] >= lengths[:, None]
        hidden_state[mask] = 0.0

    return hidden_state


@dataclass
class Gen3CConfig(PipelineConfig):
    """Configuration for GEN3C Video Generation Pipeline.
    
    GEN3C extends Cosmos with 3D cache for camera-controlled video generation.
    Key parameters:
    - frame_buffer_max: Number of 3D cache buffers (default: 2)
    - noise_aug_strength: Strength of noise augmentation per buffer
    - filter_points_threshold: Threshold for filtering unreliable depth points
    """

    dit_config: DiTConfig = field(default_factory=Gen3CVideoConfig)

    vae_config: VAEConfig = field(default_factory=CosmosVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5LargeConfig(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (t5_large_postprocess_text, ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", ))

    # GEN3C-specific conditioning parameters
    conditioning_strategy: str = "frame_replace"
    min_num_conditional_frames: int = 1
    max_num_conditional_frames: int = 2
    sigma_conditional: float = 0.0001
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 16  # GEN3C uses 16 latent frames (121 pixel frames)
    text_encoder_class: str = "T5"

    # Flow matching parameters
    embedded_cfg_scale: int = 6
    flow_shift: float = 1.0

    # GEN3C 3D Cache parameters
    frame_buffer_max: int = 2
    noise_aug_strength: float = 0.0
    filter_points_threshold: float = 1.0

    # Depth estimation settings
    use_moge_depth: bool = True
    moge_model_name: str = "Ruicheng/moge-vitl"

    # Camera trajectory settings
    default_trajectory_type: str = "clockwise"
    default_movement_distance: float = 0.03
    default_camera_rotation: str = "center_facing"

    # Video generation settings
    video_resolution: tuple[int, int] = (720, 1280)  # H, W
    num_frames: int = 121  # Default number of frames to generate

    # Generation frame rate
    fps: int = 24

    # Autoregressive generation settings
    autoregressive_chunk_frames: int = 121  # Frames per chunk
    autoregressive_overlap_frames: int = 1  # Overlap between chunks

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

        self._vae_latent_dim = 16

        # Validate frame buffer configuration matches DiT
        if hasattr(self.dit_config, 'arch_config'):
            arch_config = self.dit_config.arch_config
            if (hasattr(arch_config, 'frame_buffer_max')
                    and arch_config.frame_buffer_max != self.frame_buffer_max):
                raise ValueError(
                    f"frame_buffer_max mismatch: pipeline config has {self.frame_buffer_max}, "
                    f"DiT config has {arch_config.frame_buffer_max}")


@dataclass
class Gen3CInferenceConfig(Gen3CConfig):
    """Configuration for GEN3C inference with optimized defaults."""

    # Use smaller batch sizes for inference
    batch_size: int = 1

    # Enable gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = False

    # Inference-specific parameters
    guidance_scale: float = 6.0
    num_inference_steps: int = 50

    # Disable noise augmentation during inference
    noise_aug_strength: float = 0.0
