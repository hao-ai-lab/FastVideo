# SPDX-License-Identifier: Apache-2.0
"""Arch config for Wan2.2-Animate-14B (character animation / replacement).

Every value is transcribed verbatim from
``Wan-AI/Wan2.2-Animate-14B-Diffusers/transformer/config.json`` -- do not
"tidy" them. Field names match that file's keys one-for-one because
``update_model_arch`` overlays any matching key from the checkpoint's
config.json at load time; a renamed field silently stops receiving its
checkpoint value.

The tower itself is the Wan-I2V one (this class only switches on the I2V
knobs ``image_dim``/``added_kv_proj_dim``); why that is the right base is
explained on ``WanAnimate14BConfig`` in ``configs/pipelines/wan.py``. What
Animate adds on top:

* ``in_channels = 36 = 16 (noise) + 4 (mask) + 16 (conditional latent)``.
* ``pose_patch_embedding`` -- a second patchifier whose output is added to
  the video tokens (skipping the reference latent frame).
* A LIA-style motion encoder (``motion_*`` fields) turning 512x512 face crops
  into 20-dim motion codes. The checkpoint stores its conv/linear weights
  **unit-scale**: the model must apply the StyleGAN2 runtime factor
  (``1/sqrt(fan_in)``) in forward. Loading these into vanilla
  ``nn.Conv2d``/``nn.Linear`` forwards succeeds and is silently wrong.
* A face encoder + ``face_adapter`` cross-attention blocks. The checkpoint
  indexes the adapters densely (``face_adapter.0 .. .7``) with no record of
  which transformer block each serves; adapter ``i`` serves block
  ``i * inject_face_latents_blocks``. That correspondence exists only here,
  so it is asserted in ``__post_init__`` -- a wrong value loads cleanly and
  steers the wrong blocks.

Naming contract with the model (``fastvideo/models/dits/wan_animate.py``):
the Animate-specific modules keep checkpoint-identical parameter names
(``motion_encoder.*``, ``face_encoder.*``, ``face_adapter.*``) so the loader
passes them through verbatim; only ``pose_patch_embedding`` needs a mapping
entry, mirroring how the base ``patch_embedding`` is wrapped in ``.proj``.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig
from fastvideo.platforms import AttentionBackendEnum


def _dit_blocks_only(n: str, m) -> bool:
    """FSDP/compile unit = one transformer block. The base predicate
    ("blocks" anywhere in the name) would also match the motion encoder's
    tiny ``res_blocks.N`` convs and make each its own unit."""
    return n.startswith("blocks.") and n.split(".")[-1].isdigit()


@dataclass
class WanAnimateArchConfig(WanVideoArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_dit_blocks_only])
    # The face adapter's per-frame attention is implemented on these two only;
    # notably no VSA -- the checkpoint has no gate weights for VSA blocks.
    _supported_attention_backends: tuple[AttentionBackendEnum,
                                         ...] = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)

    # Appending last is safe despite first-match-wins: the base
    # `^patch_embedding\.` regex cannot match `pose_patch_embedding.*`.
    param_names_mapping: dict = field(default_factory=lambda: WanVideoArchConfig().param_names_mapping |
                                      {r"^pose_patch_embedding\.(.*)$": r"pose_patch_embedding.proj.\1"})
    # The relighting LoRA was trained on the official (native-naming) model,
    # whose I2V cross-attention carries k_img/v_img projections the base LoRA
    # mapping does not cover. Harmless for adapters that do not target them.
    lora_param_names_mapping: dict = field(
        default_factory=lambda: WanVideoArchConfig().lora_param_names_mapping | {
            r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
            r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
        })

    # --- I2V skeleton knobs (config.json: in_channels 36 / image_dim 1280) ---
    in_channels: int = 36
    out_channels: int = 16
    image_dim: int | None = 1280
    added_kv_proj_dim: int | None = 5120

    # --- Animate-specific (names match config.json keys exactly) ---
    # in_channels = 2 * latent_channels + 4 mask channels; asserted below.
    latent_channels: int = 16
    # None -> the model uses the LIA channel table keyed by feature-map size
    # (512: 32ch ... 4: 512ch). config.json ships null here.
    motion_encoder_channel_sizes: dict[str, int] | None = None
    motion_encoder_size: int = 512  # face crops are motion_encoder_size**2 RGB
    motion_style_dim: int = 512  # appearance vector width
    motion_dim: int = 20  # the identity-squeezing bottleneck
    motion_encoder_dim: int = 512  # per-frame motion vector width fed to the face encoder
    face_encoder_hidden_dim: int = 1024
    face_encoder_num_heads: int = 4  # 4 face tokens (+1 padding token) per latent frame
    inject_face_latents_blocks: int = 5  # adapter i serves block i * this
    # An inference-memory knob (face crops go through the motion encoder this
    # many frames at a time), but config.json ships it, so it lives here.
    motion_encoder_batch_size: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        # Re-checked after update_model_arch overlays the checkpoint's
        # config.json: a checkpoint this config cannot represent should fail
        # here, not as a shape error mid-load.
        assert self.in_channels == 2 * self.latent_channels + 4, (
            f"in_channels ({self.in_channels}) must be 2 * latent_channels "
            f"({self.latent_channels}) + 4 mask channels: noise | mask | conditional latent")
        assert self.inject_face_latents_blocks > 0, "inject_face_latents_blocks must be positive"
        assert self.num_layers % self.inject_face_latents_blocks == 0, (
            f"num_layers ({self.num_layers}) must be an exact multiple of the face-adapter "
            f"injection stride ({self.inject_face_latents_blocks}): the checkpoint stores adapters "
            f"densely (face_adapter.0..N) and adapter i serves block i * stride")


@dataclass
class WanAnimateConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanAnimateArchConfig)

    prefix: str = "WanAnimate"
