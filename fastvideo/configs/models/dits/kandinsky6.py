# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_kandinsky6_transformer_block(n: str, m) -> bool:
    return ("text_transformer_blocks" in n or "visual_transformer_blocks" in n) and n.split(".")[-1].isdigit()


@dataclass
class Kandinsky6ArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_kandinsky6_transformer_block])

    # NABLA block-sparse attention for attention_engine="nabla" checkpoints,
    # plus the dense backends every DiT supports. Kandinsky6 reuses the same
    # NABLA backend as Kandinsky5 (fastvideo/attention/backends/nabla.py) for
    # the video self-attention sub-layer only -- audio self-attention and the
    # video<->audio cross-attention are always dense.
    #
    # Note: the diffusers reference's own `engine`/`attention_engine` doesn't
    # actually gate NABLA at construction time -- every attention module can
    # receive `sparse_params` at *call* time regardless of this value (NABLA
    # dispatch is a per-call pipeline decision there, not a fixed model
    # property). FastVideo's LocalAttention needs its backend bound at
    # construction time, so this port keeps Kandinsky5's construction-time
    # gate instead: only build the extra NABLA-backed attention module when
    # attention_engine=="nabla".
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.NABLA_ATTN,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    # Map the current Diffusers checkpoint names to native FastVideo layers. The loader applies only
    # the FIRST matching rule: decoder renames must include the FFN rename in the same rule.
    # Text-tower attn is self-attention; decoder self/cross-attention names are unchanged.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(visual_transformer_blocks\.\d+)\.video_dec_block\.feed_forward\.net\.0\.proj\.(weight|bias)$":
            r"\1.videoT.feed_forward.mlp.fc_in.\2",
            r"^(visual_transformer_blocks\.\d+)\.video_dec_block\.feed_forward\.net\.2\.(weight|bias)$":
            r"\1.videoT.feed_forward.mlp.fc_out.\2",
            r"^(visual_transformer_blocks\.\d+)\.video_dec_block\.(.*)$": r"\1.videoT.\2",
            r"^(visual_transformer_blocks\.\d+)\.audio_dec_block\.feed_forward\.net\.0\.proj\.(weight|bias)$":
            r"\1.audioT.feed_forward.mlp.fc_in.\2",
            r"^(visual_transformer_blocks\.\d+)\.audio_dec_block\.feed_forward\.net\.2\.(weight|bias)$":
            r"\1.audioT.feed_forward.mlp.fc_out.\2",
            r"^(visual_transformer_blocks\.\d+)\.audio_dec_block\.(.*)$": r"\1.audioT.\2",
            r"^((?:video|audio)_text_transformer_blocks\.\d+)\.attn\.(.*)$": r"\1.self_attention.\2",
            r"^((?:video|audio)_text_transformer_blocks\.\d+)\.attn_norm\.(.*)$": r"\1.self_attention_norm.\2",
            r"^((?:video|audio)_time_embeddings)\.timestep_embedder\.linear_1\.(weight|bias)$": r"\1.in_layer.\2",
            r"^((?:video|audio)_time_embeddings)\.timestep_embedder\.linear_2\.(weight|bias)$": r"\1.out_layer.\2",
            r"^(.*feed_forward)\.net\.0\.proj\.(weight|bias)$": r"\1.mlp.fc_in.\2",
            r"^(.*feed_forward)\.net\.2\.(weight|bias)$": r"\1.mlp.fc_out.\2",
        })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Diffusers Kandinsky6Transformer3DModel config fields (mirror
    # transformer/config.json 1:1).
    in_visual_dim: int = 16
    out_visual_dim: int = 16
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    time_dim: int = 1024
    patch_size: tuple[int, int, int] = (1, 2, 2)
    model_dim: int = 4096
    ff_dim: int = 16384
    num_text_blocks: int = 4
    num_visual_blocks: int = 60
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    visual_cond: bool = True

    # RoPE (T, H, W) axis divisor from transformer/config.json. A fixed
    # value, not derived from height/width.
    scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0)

    # Always True: the DiT is the joint video+audio model. Kept so the
    # transformer/config.json key round-trips; False is rejected below.
    is_multimodal: bool = True
    out_audio_dim: int | None = None
    in_audio_dim: int = 20
    model_dim_a: int | None = None
    time_dim_a: int | None = None
    ff_dim_a: int | None = None
    axes_dims_a: tuple[int, int, int] | None = None
    audio_freqs_scaling: float = 1.0

    # transformer/config.json's "attention_engine" key, kept as a plain string
    # (not AttentionBackendEnum) so it round-trips. Only the value "nabla" is
    # treated specially, per the note above.
    attention_engine: str = "auto"
    attention_causal: bool | None = None
    attention_local: bool | None = None
    attention_glob: bool | None = None
    attention_window: int | None = None
    attention_P: float | None = None
    attention_wT: int | None = None
    attention_wW: int | None = None
    attention_wH: int | None = None
    attention_add_sta: bool | None = None
    attention_method: str | None = None

    # Parsed so the transformer/config.json key round-trips, but not read.
    # Diffusers pads text tokens to a fixed length and masks the padding in
    # attention; this port feeds the DiT the unpadded tokens with no mask
    # (as Kandinsky5 does), which is numerically equivalent.
    text_token_padding: bool = False

    # Video<->audio fused block knobs (Kandinsky6FusedTransformerDecoderBlock).
    ca_rope: bool = False
    cross_gates: bool = False
    fix_modulation: bool = False
    # >0 adds a learned embedding distinguishing generated vs. reference
    # visual tokens, consumed by IT2VA's tail_cond_first_frame conditioning.
    # Defaults to 2 (generated/reference) rather than diffusers' bare-DiT
    # default of 0: FastVideo ships one merged pipeline that serves both
    # T2VA (no image -> layer exists but unused) and IT2VA (image ->
    # tail_cond_first_frame needs it) from the same checkpoint/config.
    visual_token_type_num_embeddings: int = 2

    def __post_init__(self):
        super().__post_init__()
        if not self.is_multimodal:
            raise ValueError("Kandinsky6 only supports is_multimodal=True: the DiT is always the joint "
                             "video+audio model.")
        head_dim = sum(self.axes_dims)
        if self.model_dim % head_dim != 0:
            raise ValueError(f"model_dim ({self.model_dim}) must be divisible by head_dim ({head_dim})")
        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // head_dim
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim

        # Resolve the audio-tower dims once here (mirrors the `x or default`
        # resolution diffusers' Kandinsky6Transformer3DModel.__init__ does
        # inline) so the model constructor can read them unconditionally.
        self.model_dim_a = self.model_dim_a or self.model_dim
        self.time_dim_a = self.time_dim_a or self.time_dim
        self.ff_dim_a = self.ff_dim_a or self.ff_dim
        self.axes_dims_a = self.axes_dims_a or self.axes_dims
        head_dim_a = sum(self.axes_dims_a)
        if self.model_dim_a % head_dim_a != 0:
            raise ValueError(f"model_dim_a ({self.model_dim_a}) must be divisible by head_dim_a ({head_dim_a})")

        # Kandinsky6FusedTransformerDecoderBlock's cross-modal modulation is
        # driven by the *other* modality's time embedding by default
        # (fix_modulation=False, matching the diffusers reference): the
        # video-conditioning-on-audio modulation is built from time_dim but
        # invoked with the audio time embedding, and vice versa. That only
        # type-checks when the two time embeddings are the same width, so
        # reject the mismatched, fix_modulation=False combination here with a
        # clear message instead of a cryptic matmul shape error deep inside
        # the fused block.
        if not self.fix_modulation and self.time_dim_a != self.time_dim:
            raise ValueError(f"time_dim_a ({self.time_dim_a}) must equal time_dim ({self.time_dim}) unless "
                             "fix_modulation=True (Kandinsky6FusedTransformerDecoderBlock's cross-modal "
                             "modulation is driven by the other modality's time embedding by default).")


@dataclass
class Kandinsky6VideoAudioConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Kandinsky6ArchConfig)
    prefix: str = "Kandinsky6"
