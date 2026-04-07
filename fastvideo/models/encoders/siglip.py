# SPDX-License-Identifier: Apache-2.0
"""
SigLIP Vision Encoder for FastVideo.

SigLIP (Sigmoid Loss for Language-Image Pre-training) is similar to CLIP
but uses sigmoid loss instead of contrastive loss, and doesn't use a CLS token.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.siglip import SiglipVisionArchConfig, SiglipVisionConfig
from fastvideo.distributed import divide, get_tp_world_size
from fastvideo.layers.activation import get_act_fn
from fastvideo.layers.linear import (ColumnParallelLinear, QKVParallelLinear,
                                     RowParallelLinear)
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.logger import init_logger
from fastvideo.models.encoders.base import ImageEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class SiglipVisionEmbeddings(nn.Module):
    """
    SigLIP vision embeddings - similar to CLIP but without class embedding.
    """

    def __init__(self, arch_config: SiglipVisionArchConfig):
        super().__init__()
        self.arch_config = arch_config
        self.embed_dim = arch_config.hidden_size
        self.image_size = arch_config.image_size
        self.patch_size = arch_config.patch_size
        # SigLIP uses valid padding, so non-divisible sizes work (edge pixels are ignored)

        self.patch_embedding = nn.Conv2d(
            in_channels=arch_config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # SigLIP uses valid padding
        )

        # Integer division - with valid padding, edge pixels are ignored
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, embed_dim, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention for SigLIP."""

    def __init__(
        self,
        arch_config: SiglipVisionArchConfig,
        enable_scale: bool = True,
        is_causal: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.arch_config = arch_config
        self.embed_dim = arch_config.hidden_size
        self.num_heads = arch_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        
        self.scale = self.head_dim**-0.5 if enable_scale else None
        self.dropout = arch_config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tp_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = LocalAttention(
            self.num_heads_per_partition,
            self.head_dim,
            self.num_heads_per_partition,
            softmax_scale=self.scale,
            causal=is_causal,
            supported_attention_backends=arch_config._supported_attention_backends,
        )

    def forward(self, hidden_states: torch.Tensor):
        """Input shape: Batch x Time x Channel"""
        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        
        query_states = query_states.reshape(
            query_states.shape[0], query_states.shape[1],
            self.num_heads_per_partition, self.head_dim
        )
        key_states = key_states.reshape(
            key_states.shape[0], key_states.shape[1],
            self.num_heads_per_partition, self.head_dim
        )
        value_states = value_states.reshape(
            value_states.shape[0], value_states.shape[1],
            self.num_heads_per_partition, self.head_dim
        )
        
        attn_output = self.attn(query_states, key_states, value_states)
        attn_output = attn_output.reshape(
            attn_output.shape[0], attn_output.shape[1],
            self.num_heads_per_partition * self.head_dim
        )
        attn_output, _ = self.out_proj(attn_output)
        return attn_output, None


class SiglipMLP(nn.Module):

    def __init__(
        self,
        arch_config: SiglipVisionArchConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.arch_config = arch_config
        self.activation_fn = get_act_fn(arch_config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            arch_config.hidden_size,
            arch_config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            arch_config.intermediate_size,
            arch_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(
        self,
        arch_config: SiglipVisionArchConfig,
        enable_scale: bool = True,
        is_causal: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = SiglipAttention(
            arch_config,
            enable_scale=enable_scale,
            is_causal=is_causal,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(
            arch_config.hidden_size, eps=arch_config.layer_norm_eps
        )
        self.mlp = SiglipMLP(
            arch_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(
            arch_config.hidden_size, eps=arch_config.layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SigLIP uses post-norm (like original ViT)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    """SigLIP encoder consisting of transformer layers."""

    def __init__(
        self,
        arch_config: SiglipVisionArchConfig,
        enable_scale: bool = True,
        is_causal: bool = False,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.arch_config = arch_config

        if num_hidden_layers_override is None:
            num_hidden_layers = arch_config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
            
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(
                arch_config=arch_config,
                enable_scale=enable_scale,
                is_causal=is_causal,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            )
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states_pool
        return [hidden_states]


class SiglipVisionTransformer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        arch_config = config.arch_config
        embed_dim = arch_config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(arch_config)

        self.encoder = SiglipEncoder(
            arch_config=arch_config,
            enable_scale=config.enable_scale,
            is_causal=config.is_causal,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = arch_config.num_hidden_layers
        if len(self.encoder.layers) > arch_config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # Post layer norm (applied to output)
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=arch_config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: list[int] | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        return_all_hidden_states = feature_sample_layers is not None
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states,
        )

        if not return_all_hidden_states:
            encoder_outputs = encoder_outputs[0]

        # Apply post-layernorm
        if self.post_layernorm is not None:
            if isinstance(encoder_outputs, list):
                encoder_outputs[-1] = self.post_layernorm(encoder_outputs[-1])
            else:
                encoder_outputs = self.post_layernorm(encoder_outputs)

        return encoder_outputs


class SiglipVisionModel(ImageEncoder):
    """SigLIP Vision Model for FastVideo."""
    
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.vision_model = SiglipVisionTransformer(
            config=config,
            quant_config=config.quant_config,
            num_hidden_layers_override=config.num_hidden_layers_override,
            require_post_norm=config.require_post_norm,
            prefix=f"{config.prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: list[int] | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        last_hidden_state = self.vision_model(pixel_values, feature_sample_layers)
        return BaseEncoderOutput(last_hidden_state=last_hidden_state)

    @property
    def device(self):
        return next(self.parameters()).device

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # Skip projection layers if any
            if name.startswith("visual_projection"):
                continue
                
            # Skip head if any
            if "head" in name:
                continue

            # Post layernorm handling
            if (name.startswith("vision_model.post_layernorm")
                    and self.vision_model.post_layernorm is None):
                continue

            # Omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            # Handle QKV projection weight mapping
            for (param_name, weight_name, shard_id) in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
            
        return loaded_params

# Entry point for model registry
EntryClass = SiglipVisionModel
