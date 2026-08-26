# SPDX-License-Identifier: Apache-2.0
"""Arch config for Wan2.2-Animate-14B (character animation / replacement).

Every architecture value here is transcribed verbatim from the official
Diffusers-format checkpoint manifest at
``Wan-AI/Wan2.2-Animate-14B-Diffusers/transformer/config.json`` -- do not
"tidy" them. Field names deliberately match that file's keys one-for-one:
``update_model_arch`` overlays any matching key from the checkpoint's
config.json onto this dataclass at load time, so a renamed field silently
stops receiving its checkpoint value.

Despite the Wan2.2 branding, Animate is a *dense* 40-layer / 5120-dim DiT on
the Wan2.1-I2V skeleton (single expert -- no MoE pair -- with the Wan2.1
16-channel VAE and the I2V CLIP image branch). The paper (arXiv:2509.14055)
keeps Wan-I2V's input structure to minimise distributional shift during
post-training, which is why this config subclasses ``WanVideoArchConfig``
with the I2V knobs (``image_dim``, ``added_kv_proj_dim``) switched on rather
than defining a new tower.

What Animate adds on top of the I2V skeleton:

* ``in_channels = 36 = 16 (noise) + 4 (mask) + 16 (conditional latent)`` --
  the Wan-I2V channel-concat conditioning, reused for the reference image,
  temporal-guidance frames and (in replacement mode) the background video.
* ``pose_patch_embedding`` -- a second patchifier whose output is *added* to
  the video tokens (skipping the reference latent frame).
* A LIA-style motion encoder (``motion_*`` fields) turning 512x512 face crops
  into 20-dim motion codes, recombined through a QR-orthonormalised basis.
  The checkpoint stores its conv/linear weights **unit-scale**: the model must
  apply the StyleGAN2 runtime factor (``1/sqrt(fan_in)``) in forward. Loading
  these into vanilla ``nn.Conv2d``/``nn.Linear`` forwards succeeds and is
  silently wrong by a per-layer constant.
* A face encoder + ``face_adapter`` cross-attention blocks. The checkpoint
  indexes the adapters densely (``face_adapter.0 .. .7``) with no record of
  which transformer block each serves; adapter ``i`` serves block
  ``i * inject_face_latents_blocks`` (0, 5, ..., 35). That correspondence
  exists only here, so it is asserted in ``__post_init__`` -- a wrong value
  loads cleanly and steers the wrong blocks.

Naming contract with the model (``fastvideo/models/dits/wan_animate.py``):
the Animate-specific modules keep checkpoint-identical parameter names
(``motion_encoder.*``, ``face_encoder.*``, ``face_adapter.*``) so the loader
passes them through verbatim; only ``pose_patch_embedding`` needs a mapping
entry, mirroring how the base ``patch_embedding`` is wrapped in ``.proj``.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig


def _animate_lora_param_names_mapping() -> dict:
    """Base Wan LoRA mapping (native -> diffusers naming) + the I2V image-KV heads.

    The relighting LoRA was trained on the official (native-naming) model,
    whose I2V cross-attention carries ``k_img``/``v_img`` projections the base
    mapping does not cover. Harmless when a given adapter does not target them.
    """
    mapping = dict(WanVideoArchConfig().lora_param_names_mapping)
    mapping[r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$"] = r"blocks.\1.attn2.add_k_proj.\2"
    mapping[r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$"] = r"blocks.\1.attn2.add_v_proj.\2"
    return mapping


def _animate_param_names_mapping() -> dict:
    """Base Wan (diffusers-naming) mapping + the one Animate-specific entry.

    The Animate checkpoint is diffusers-format, so the whole base-tower
    mapping from ``WanVideoArchConfig`` applies unchanged. ``motion_encoder``,
    ``face_encoder`` and ``face_adapter`` tensors keep their checkpoint names
    in the FastVideo model and need no entries (unmatched names load
    verbatim). ``pose_patch_embedding`` mirrors ``patch_embedding``'s
    ``.proj`` wrapping -- the anchored base regex cannot match it, hence the
    explicit entry.
    """
    mapping = dict(WanVideoArchConfig().param_names_mapping)
    mapping[r"^pose_patch_embedding\.(.*)$"] = r"pose_patch_embedding.proj.\1"
    return mapping


@dataclass
class WanAnimateArchConfig(WanVideoArchConfig):
    param_names_mapping: dict = field(default_factory=_animate_param_names_mapping)
    lora_param_names_mapping: dict = field(default_factory=_animate_lora_param_names_mapping)

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
    # Face crops are chunked through the motion encoder this many frames at a
    # time (inference-memory knob, not an architecture parameter).
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
            f"num_layers ({self.num_layers}) must divide evenly into face-adapter injection "
            f"stride ({self.inject_face_latents_blocks}): the checkpoint stores adapters densely "
            f"(face_adapter.0..N) and adapter i serves block i * stride")
        assert self.motion_encoder_size >= 8 and (self.motion_encoder_size & (self.motion_encoder_size - 1)) == 0, (
            f"motion_encoder_size ({self.motion_encoder_size}) must be a power of two >= 8: the "
            f"LIA appearance encoder halves the feature map from size down to 4x4")


@dataclass
class WanAnimateConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanAnimateArchConfig)

    prefix: str = "WanAnimate"
