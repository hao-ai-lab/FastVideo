# SPDX-License-Identifier: Apache-2.0
"""Arch config for Wan2.2-S2V-14B (speech-to-video).

Every architecture value here is transcribed verbatim from the official
checkpoint manifest at ``Wan-AI/Wan2.2-S2V-14B/config.json`` -- do not "tidy"
them. ``dim: 5120`` and ``num_heads: 40`` in that file become
``num_attention_heads=40`` / ``attention_head_dim=128`` here (5120 / 40).

Unlike every other Wan variant in this package, the S2V checkpoint ships in
*native* Wan format (``blocks.0.self_attn.q``), not diffusers format
(``blocks.0.attn1.to_q``). ``param_names_mapping`` therefore translates from
the native layout; see ``lora_param_names_mapping`` in ``wanvideo.py`` for the
same naming scheme applied to Wan LoRA adapters.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class WanS2VArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    # Only names that actually change need an entry: the loader passes anything
    # that matches no pattern through verbatim (models/loader/utils.py), which
    # covers the qk norms, the modulation tables and the whole audio encoder.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # --- transformer blocks: native Wan naming -> FastVideo naming ---
            # Self-attention params live flat on the block (matches wanvideo.py).
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.self_attn.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.self_attn.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.self_attn.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.self_attn.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.cross_attn.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.cross_attn.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.cross_attn.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.cross_attn.to_out.\2",
            # ffn is an nn.Sequential upstream; we name the two Linears.
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            # --- audio injection: 12 cross-attn gadgets beside the block tower ---
            r"^audio_injector\.injector\.(\d+)\.q\.(.*)$": r"audio_injector.injector.\1.to_q.\2",
            r"^audio_injector\.injector\.(\d+)\.k\.(.*)$": r"audio_injector.injector.\1.to_k.\2",
            r"^audio_injector\.injector\.(\d+)\.v\.(.*)$": r"audio_injector.injector.\1.to_v.\2",
            r"^audio_injector\.injector\.(\d+)\.o\.(.*)$": r"audio_injector.injector.\1.to_out.\2",
            # --- top-level embedders: nn.Sequential -> named Linears ---
            r"^text_embedding\.0\.(.*)$": r"text_embedding.fc_in.\1",
            r"^text_embedding\.2\.(.*)$": r"text_embedding.fc_out.\1",
            r"^time_embedding\.0\.(.*)$": r"time_embedding.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"time_embedding.fc_out.\1",
            r"^time_projection\.1\.(.*)$": r"time_projection.linear.\1",
        })

    # --- base Wan geometry (config.json: dim 5120 / num_heads 40) ---
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16  # config.json: in_dim
    out_channels: int = 16  # config.json: out_dim
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    rope_max_seq_len: int = 1024

    # --- S2V-specific ---
    cond_dim: int = 16
    audio_dim: int = 1024
    num_audio_token: int = 4
    enable_adain: bool = True
    adain_mode: str = "attn_norm"
    audio_inject_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39)
    zero_init: bool = True
    zero_timestep: bool = True
    enable_motioner: bool = False
    add_last_motion: bool = True
    enable_tsm: bool = False
    trainable_token_pos_emb: bool = False
    motion_token_num: int = 1024
    enable_framepack: bool = True
    framepack_drop_mode: str = "padd"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels
        # The checkpoint indexes the audio injectors densely (injector.0 .. .11)
        # with no record of which block each serves, so a duplicate or
        # out-of-range entry here loads cleanly and steers the wrong blocks.
        assert len(set(self.audio_inject_layers)) == len(self.audio_inject_layers), \
            "audio_inject_layers must not contain duplicates"
        assert all(0 <= i < self.num_layers for i in self.audio_inject_layers), \
            f"audio_inject_layers must index into the {self.num_layers} blocks"
        assert not (self.enable_motioner and self.enable_framepack), \
            "enable_motioner and enable_framepack are mutually exclusive: upstream picks one motion path"


@dataclass
class WanS2VConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanS2VArchConfig)

    prefix: str = "WanS2V"
