# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Kandinsky5ArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [
            lambda n, m: ("text_transformer_blocks" in n
                          or "visual_transformer_blocks" in n)
            and n.split(".")[-1].isdigit()
        ])

    # Native FastVideo implementation uses the same parameter names as diffusers
    # except FFN internals: Diffusers FFN uses `in_layer/out_layer`, while
    # FastVideo uses MLP `fc_in/fc_out`.
    param_names_mapping: dict = field(default_factory=lambda: {
        r"^(.*feed_forward)\.in_layer\.(weight|bias)$":
        r"\1.mlp.fc_in.\2",
        r"^(.*feed_forward)\.out_layer\.(weight|bias)$":
        r"\1.mlp.fc_out.\2",
    })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Diffusers Kandinsky5Transformer3DModel config fields.
    in_visual_dim: int = 4
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    time_dim: int = 512
    out_visual_dim: int = 4
    patch_size: tuple[int, int, int] = (1, 2, 2)
    model_dim: int = 2048
    ff_dim: int = 5120
    num_text_blocks: int = 2
    num_visual_blocks: int = 32
    axes_dims: tuple[int, int, int] = (16, 24, 24)
    visual_cond: bool = False
    attention_type: str = "regular"
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

    def __post_init__(self):
        super().__post_init__()
        head_dim = sum(self.axes_dims)
        if self.model_dim % head_dim != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by head_dim ({head_dim})"
            )
        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // head_dim
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim


@dataclass
class Kandinsky5VideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Kandinsky5ArchConfig)
    prefix: str = "Kandinsky5"
