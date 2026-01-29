# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_double_block(n: str, m) -> bool:
    return "double" in n and str.isdigit(n.split(".")[-1])


def is_single_block(n: str, m) -> bool:
    return "single" in n and str.isdigit(n.split(".")[-1])


# def is_refiner_block(n: str, m) -> bool:
#     return "refiner" in n and str.isdigit(n.split(".")[-1])


def is_txt_in(n: str, m) -> bool:
    return n.split(".")[-1] == "txt_in"


@dataclass
class HYWorldArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block])

    _compile_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block, is_txt_in])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # 1. txt_in submodules (text embedder, refiner blocks):
            r"^txt_in\.t_embedder\.mlp\.0\.(.*)$":
            r"txt_in.t_embedder.mlp.fc_in.\1",
            r"^txt_in\.t_embedder\.mlp\.2\.(.*)$":
            r"txt_in.t_embedder.mlp.fc_out.\1",
            r"^txt_in\.c_embedder\.linear_1\.(.*)$":
            r"txt_in.c_embedder.fc_in.\1",
            r"^txt_in\.c_embedder\.linear_2\.(.*)$":
            r"txt_in.c_embedder.fc_out.\1",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.norm1\.(.*)$":
            r"txt_in.refiner_blocks.\1.norm1.\2",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.norm2\.(.*)$":
            r"txt_in.refiner_blocks.\1.norm2.\2",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_qkv\.(.*)$":
            r"txt_in.refiner_blocks.\1.self_attn_qkv.\2",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_proj\.(.*)$":
            r"txt_in.refiner_blocks.\1.self_attn_proj.\2",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc1\.(.*)$":
            r"txt_in.refiner_blocks.\1.mlp.fc_in.\2",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc2\.(.*)$":
            r"txt_in.refiner_blocks.\1.mlp.fc_out.\2",
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.adaLN_modulation\.1\.(.*)$":
            r"txt_in.refiner_blocks.\1.adaLN_modulation.linear.\2",

            # 2. time_in mappings (HYWorld uses TimestepEmbedder directly,
            #    but FastVideo model inherits HunyuanVideo15TimeEmbedding with timestep_embedder):
            r"^time_in\.mlp\.0\.(.*)$":
            r"time_in.timestep_embedder.mlp.fc_in.\1",
            r"^time_in\.mlp\.2\.(.*)$":
            r"time_in.timestep_embedder.mlp.fc_out.\1",

            # 3. action_in mappings:
            r"^action_in\.mlp\.0\.(.*)$":
            r"action_in.mlp.fc_in.\1",
            r"^action_in\.mlp\.2\.(.*)$":
            r"action_in.mlp.fc_out.\1",

            # 4. byt5_in -> txt_in_2 mappings:
            r"^byt5_in\.layernorm\.(.*)$":
            r"txt_in_2.norm.\1",
            r"^byt5_in\.fc1\.(.*)$":
            r"txt_in_2.linear_1.\1",
            r"^byt5_in\.fc2\.(.*)$":
            r"txt_in_2.linear_2.\1",
            r"^byt5_in\.fc3\.(.*)$":
            r"txt_in_2.linear_3.\1",

            # 5. cond_type_embedding -> cond_type_embed:
            r"^cond_type_embedding\.(.*)$":
            r"cond_type_embed.\1",

            # 6. vision_in -> image_embedder mappings:
            r"^vision_in\.proj\.0\.(.*)$":
            r"image_embedder.norm_in.\1",
            r"^vision_in\.proj\.1\.(.*)$":
            r"image_embedder.linear_1.\1",
            r"^vision_in\.proj\.3\.(.*)$":
            r"image_embedder.linear_2.\1",
            r"^vision_in\.proj\.4\.(.*)$":
            r"image_embedder.norm_out.\1",

            # 7. double_blocks mapping:
            r"^double_blocks\.(\d+)\.img_attn_q\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 0, 3),
            r"^double_blocks\.(\d+)\.img_attn_k\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 1, 3),
            r"^double_blocks\.(\d+)\.img_attn_v\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 2, 3),
            r"^double_blocks\.(\d+)\.txt_attn_q\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 0, 3),
            r"^double_blocks\.(\d+)\.txt_attn_k\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 1, 3),
            r"^double_blocks\.(\d+)\.txt_attn_v\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 2, 3),
            r"^double_blocks\.(\d+)\.img_mlp\.fc1\.(.*)$":
            r"double_blocks.\1.img_mlp.fc_in.\2",
            r"^double_blocks\.(\d+)\.img_mlp\.fc2\.(.*)$":
            r"double_blocks.\1.img_mlp.fc_out.\2",
            r"^double_blocks\.(\d+)\.txt_mlp\.fc1\.(.*)$":
            r"double_blocks.\1.txt_mlp.fc_in.\2",
            r"^double_blocks\.(\d+)\.txt_mlp\.fc2\.(.*)$":
            r"double_blocks.\1.txt_mlp.fc_out.\2",

            # 8. Final layer mapping:
            r"^final_layer\.adaLN_modulation\.1\.(.*)$":
            r"final_layer.adaLN_modulation.linear.\1",
        })

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Parameters from HY-WorldPlay config.json (loaded from checkpoint)
    patch_size: list | tuple | int = field(default_factory=lambda: [1, 1, 1])
    # Base latent channels - will be expanded in __post_init__ if concat_condition=True
    in_channels: int = 32
    concat_condition: bool = True
    out_channels: int = 32
    hidden_size: int = 2048
    heads_num: int = 16
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    mm_double_blocks_depth: int = 54
    mm_single_blocks_depth: int = 0
    rope_dim_list: list | tuple = field(default_factory=lambda: [16, 56, 56])
    qkv_bias: bool = True
    qk_norm: bool | str = True
    qk_norm_type: str = "rms"
    guidance_embed: bool = False
    use_meanflow: bool = False
    text_projection: str = "single_refiner"
    use_attention_mask: bool = True
    text_states_dim: int = 3584
    text_states_dim_2: int | None = None
    text_pool_type: str | None = None
    rope_theta: float = 256.0
    attn_mode: str = "flash"
    attn_param: str | None = None
    glyph_byT5_v2: bool = True
    vision_projection: str = "linear"
    vision_states_dim: int = 1152
    is_reshape_temporal_channels: bool = False
    use_cond_type_embedding: bool = True
    ideal_resolution: str = "480p"
    ideal_task: str = "i2v"
    task_type: str = "i2v"
    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["img_in", "txt_in", "time_in", "vector_in"])

    def __post_init__(self):
        super().__post_init__()
        # Convert HY-WorldPlay naming to FastVideo naming conventions
        self.num_attention_heads: int = self.heads_num
        self.attention_head_dim: int = self.hidden_size // self.heads_num
        self.num_layers: int = self.mm_double_blocks_depth
        self.num_single_layers: int = self.mm_single_blocks_depth
        self.num_refiner_layers: int = 2  # Default for HYWorld
        self.mlp_ratio: float = float(self.mlp_width_ratio)
        self.text_embed_dim: int = self.text_states_dim
        self.text_embed_2_dim: int = self.text_states_dim_2 if self.text_states_dim_2 else 1472
        self.image_embed_dim: int = self.vision_states_dim
        self.rope_axes_dim: tuple[int, ...] = tuple(self.rope_dim_list)
        self.num_channels_latents: int = self.out_channels
        self.target_size: int = 640

        # Handle concat_condition: when True, actual in_channels = base * 2 + 1
        # (base latent + condition latent + mask channel)
        # config.json has base in_channels (32), but img_in needs full (65)
        if self.concat_condition and self.in_channels == 32:
            if self.is_reshape_temporal_channels:
                self.in_channels = self.in_channels + self.in_channels // 2 + 1
            else:
                self.in_channels = self.in_channels * 2 + 1  # 32 * 2 + 1 = 65

        # Handle patch_size (can be list/tuple or int)
        if isinstance(self.patch_size, list | tuple):
            self.patch_size_t: int = self.patch_size[0]
            # assume square patch size for height and width
            patch_size_hw: int = self.patch_size[1]
            object.__setattr__(self, 'patch_size', patch_size_hw)
        else:
            self.patch_size_t = 1

        # Convert qk_norm to string format
        if isinstance(self.qk_norm, bool):
            if self.qk_norm:
                self.qk_norm = "rms_norm" if self.qk_norm_type == "rms" else self.qk_norm_type
            else:
                self.qk_norm = "none"


@dataclass
class HYWorldConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HYWorldArchConfig)

    prefix: str = "HYWorld"
