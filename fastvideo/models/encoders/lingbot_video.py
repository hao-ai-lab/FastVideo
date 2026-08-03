# SPDX-License-Identifier: Apache-2.0
"""Native LingBot-Video Qwen3-VL language and vision conditioning models."""

from collections.abc import Iterable
import itertools
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.encoders.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
)
from fastvideo.models.loader.weight_utils import default_weight_loader


def _vision_position_ids(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Lay out vision rotary positions in Qwen3-VL's spatial merge order."""
    position_ids = []
    for temporal, height, width in grid_thw.tolist():
        height_ids, width_ids = torch.meshgrid(
            torch.arange(height, device=grid_thw.device),
            torch.arange(width, device=grid_thw.device),
            indexing="ij",
        )
        block_shape = (height // merge_size, merge_size, width // merge_size, merge_size)
        height_ids = height_ids.reshape(block_shape).transpose(1, 2).flatten()
        width_ids = width_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack((height_ids, width_ids), dim=-1).repeat(temporal, 1))
    return torch.cat(position_ids, dim=0)


def _vision_bilinear_position_data(
    grid_thw: torch.Tensor,
    side: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL's interpolated absolute-position lookup data."""
    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for temporal, height, width in grid_thw.tolist():
        height_grid = torch.linspace(0, side - 1, height, device=grid_thw.device)
        width_grid = torch.linspace(0, side - 1, width, device=grid_thw.device)
        height_floor, width_floor = height_grid.int(), width_grid.int()
        height_ceil = (height_floor + 1).clamp(max=side - 1)
        width_ceil = (width_floor + 1).clamp(max=side - 1)
        height_frac, width_frac = height_grid - height_floor, width_grid - width_floor
        floor_offset, ceil_offset = height_floor * side, height_ceil * side
        corner_indices = (
            (floor_offset[:, None] + width_floor[None, :]).flatten(),
            (floor_offset[:, None] + width_ceil[None, :]).flatten(),
            (ceil_offset[:, None] + width_floor[None, :]).flatten(),
            (ceil_offset[:, None] + width_ceil[None, :]).flatten(),
        )
        corner_weights = (
            ((1 - height_frac)[:, None] * (1 - width_frac)[None, :]).flatten(),
            ((1 - height_frac)[:, None] * width_frac[None, :]).flatten(),
            (height_frac[:, None] * (1 - width_frac)[None, :]).flatten(),
            (height_frac[:, None] * width_frac[None, :]).flatten(),
        )
        height_order = torch.arange(height, device=grid_thw.device).view(height // merge_size, merge_size)
        width_order = torch.arange(width, device=grid_thw.device).view(width // merge_size, merge_size)
        reorder = (
            (height_order[:, :, None, None] * width + width_order[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(temporal)
        )
        for index in range(4):
            index_parts[index].append(corner_indices[index][reorder])
            weight_parts[index].append(corner_weights[index][reorder])
    indices = torch.stack([torch.cat(part) for part in index_parts])
    weights = torch.stack([torch.cat(part) for part in weight_parts])
    return indices, weights


class LingBotVideoQwen3VLVisionPatchEmbed(nn.Module):
    """Convert processor patch rows into Qwen3-VL vision tokens."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.hidden_size, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the released Conv3d patch projection in its parameter dtype."""
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


class LingBotVideoQwen3VLVisionMLP(nn.Module):
    """Qwen3-VL vision feed-forward block with tanh-approximate GELU."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the two released vision MLP projections."""
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states), approximate="tanh"))


class LingBotVideoQwen3VLVisionAttention(nn.Module):
    """Packed non-causal Qwen3-VL vision attention using native SDPA."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embeddings and attend independently within each packed image."""
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_dtype, key_dtype = query.dtype, key.dtype
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        query_float, key_float = query.float(), key.float()
        query = (query_float * cos + self._rotate_half(query_float) * sin).to(query_dtype)
        key = (key_float * cos + self._rotate_half(key_float) * sin).to(key_dtype)
        query, key, value = [tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value)]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        query_chunks, key_chunks, value_chunks = [torch.split(tensor, lengths, dim=2) for tensor in (query, key, value)]
        outputs = [
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            ).transpose(1, 2)
            for q, k, v in zip(query_chunks, key_chunks, value_chunks, strict=True)
        ]
        return self.proj(torch.cat(outputs, dim=1).reshape(sequence_length, -1).contiguous())

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        """Rotate the two halves used by Qwen3-VL's NeoX rotary layout."""
        first, second = hidden_states.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)


class LingBotVideoQwen3VLVisionBlock(nn.Module):
    """One released Qwen3-VL vision transformer block."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = LingBotVideoQwen3VLVisionAttention(config)
        self.mlp = LingBotVideoQwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention and MLP residuals in the official order."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class LingBotVideoQwen3VLVisionPatchMerger(nn.Module):
    """Merge each 2x2 vision-token block into language hidden width."""

    def __init__(self, config: SimpleNamespace, *, postshuffle_norm: bool) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.postshuffle_norm = postshuffle_norm
        norm_size = self.hidden_size if postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize at the released boundary, shuffle, project, and merge."""
        if self.postshuffle_norm:
            hidden_states = self.norm(hidden_states.view(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.linear_fc1(hidden_states)
        return self.linear_fc2(F.gelu(hidden_states))


class LingBotVideoQwen3VLVisionModel(nn.Module):
    """Native Qwen3-VL vision tower, including the three DeepStack outputs."""

    def __init__(self, config: SimpleNamespace) -> None:
        """Build the released patch, position, block, and merge structure."""
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = LingBotVideoQwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = nn.ModuleList(LingBotVideoQwen3VLVisionBlock(config) for _ in range(config.depth))
        self.merger = LingBotVideoQwen3VLVisionPatchMerger(config, postshuffle_norm=False)
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            LingBotVideoQwen3VLVisionPatchMerger(config, postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        )
        head_dim = config.hidden_size // config.num_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode processor patches into merged and DeepStack visual features."""
        indices, weights = _vision_bilinear_position_data(
            grid_thw,
            self.num_grid_per_side,
            self.spatial_merge_size,
        )
        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        hidden_states = self.patch_embed(pixel_values)
        position_embeds = (self.pos_embed(indices) * weights[:, :, None]).sum(0)
        hidden_states = hidden_states + position_embeds.to(hidden_states.dtype)
        rotary = (position_ids.unsqueeze(-1) * self.rotary_inv_freq).flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos, sin = rotary.cos(), rotary.sin()
        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, cos, sin)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.deepstack_merger_list[merger_index](hidden_states))
        return self.merger(hidden_states), deepstack_features


class LingBotVideoQwen3VLAttention(Qwen3Attention):
    """Qwen3-VL attention with the official masked repeat-K/V SDPA path."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        """Keep the interleaved multimodal RoPE section sizes with attention."""
        super().__init__(config=config, **kwargs)
        self.mrope_section = config.mrope_section
        inverse_frequency = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.int64, device="cpu").float()
                / self.head_dim
            )
        )
        self.register_buffer("mrope_inv_freq", inverse_frequency, persistent=False)

    def _apply_qwen3_vl_rope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply NeoX RoPE with Qwen3-VL's input-dtype multiply-add ordering."""
        if positions.ndim == 3:
            inverse_frequency = self.mrope_inv_freq.to(query.device)
            expanded_frequency = inverse_frequency[None, None, :, None].expand(
                3,
                positions.shape[1],
                -1,
                1,
            )
            expanded_positions = positions[:, :, None, :].float()
            frequencies = (expanded_frequency.float() @ expanded_positions.float()).transpose(2, 3)
            frequencies_t = frequencies[0]
            for dimension, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dimension] * 3
                index = slice(offset, length, 3)
                frequencies_t[..., index] = frequencies[dimension, ..., index]
            embeddings = torch.cat((frequencies_t, frequencies_t), dim=-1)
            cos = embeddings.cos().to(query.dtype).unsqueeze(2)
            sin = embeddings.sin().to(query.dtype).unsqueeze(2)
            return self._rotate(query, cos, sin), self._rotate(key, cos, sin)
        flat_positions = positions.flatten()
        cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, flat_positions)
        cos_half, sin_half = cos_sin.chunk(2, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1).to(query.dtype)
        sin = torch.cat((sin_half, sin_half), dim=-1).to(query.dtype)
        if flat_positions.numel() == query.shape[1]:
            cos = cos.view(1, query.shape[1], 1, self.head_dim)
            sin = sin.view(1, query.shape[1], 1, self.head_dim)
        else:
            cos = cos.view(*query.shape[:2], 1, self.head_dim)
            sin = sin.view(*query.shape[:2], 1, self.head_dim)

        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)

    @staticmethod
    def _rotate(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply Qwen3-VL's half-rotation multiply-add ordering."""
        first, second = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return tensor * cos + rotated * sin

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply fused projections, QK norm, RoPE, and causal grouped attention."""
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        batch_size, sequence_length = query.shape[:2]
        query = query.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self._apply_qwen3_vl_rope(positions, query, key)
        no_padding = attention_mask is None
        if no_padding:
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=sequence_length > 1,
                scale=self.scaling,
                enable_gqa=self.num_heads != self.num_kv_heads,
            ).transpose(1, 2)
        else:
            groups = self.num_heads // self.num_kv_heads
            key = (key[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(batch_size, sequence_length,
                                                                                self.num_heads, self.head_dim))
            value = (value[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(batch_size, sequence_length,
                                                                                    self.num_heads, self.head_dim))
            causal_mask = torch.ones(sequence_length, sequence_length, device=query.device, dtype=torch.bool).tril()
            key_mask = attention_mask.to(device=query.device, dtype=torch.bool)
            sdpa_mask = causal_mask[None, None, :, :] & key_mask[:, None, None, :]
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            ).transpose(1, 2)
        output, _ = self.o_proj(attention_output.reshape(batch_size, sequence_length, -1))
        return output


class LingBotVideoQwen3VLDecoderLayer(Qwen3DecoderLayer):
    """Qwen3-VL decoder layer with explicit official residual rounding order."""

    def __init__(self, config: Any, prefix: str) -> None:
        """Build the final Qwen3-VL attention once to avoid orphan parameters."""
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        quant_config = getattr(config, "quant_config", None)
        self.self_attn = LingBotVideoQwen3VLAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            bias=config.attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention and MLP with each residual sum rounded before normalization."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LingBotVideoQwen3VLTextModel(Qwen3ForCausalLM):
    """Load the Qwen3-VL language-model subset without its vision tower or LM head."""

    supports_hf_from_pretrained = False

    def __init__(self, config) -> None:
        """Construct the exact Qwen3-VL module graph without replacing base layers."""
        TextEncoder.__init__(self, config)
        self.quant_config = getattr(config, "quant_config", None)
        if getattr(config, "lora_config", None) is not None:
            max_loras = getattr(config.lora_config, "max_loras", 1)
            lora_vocab_size = getattr(config.lora_config, "lora_extra_vocab_size", 1)
            lora_vocab = lora_vocab_size * max_loras
        else:
            lora_vocab = 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=self.quant_config,
        )
        self.layers = nn.ModuleList(
            LingBotVideoQwen3VLDecoderLayer(config, prefix=f"{config.prefix}.layers.{index}")
            for index in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        """Run explicit Qwen3-VL layers and return the requested hidden-state tuple."""
        del kwargs
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        if attention_mask is not None and bool(attention_mask.to(torch.bool).all()):
            attention_mask = None
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        for layer_index, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states, )
            hidden_states = layer(position_ids, hidden_states, attention_mask)
            if deepstack_visual_embeds is not None and layer_index < len(deepstack_visual_embeds):
                if visual_pos_masks is None:
                    raise ValueError("visual_pos_masks is required with DeepStack features")
                hidden_states = hidden_states.clone()
                visual_states = deepstack_visual_embeds[layer_index].to(hidden_states.device, hidden_states.dtype)
                hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + visual_states
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states, )
        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Accept either official compound keys or converted native keys."""
        prefix = "model.language_model."
        language_weights = ((name[len(prefix):] if name.startswith(prefix) else name, tensor)
                            for name, tensor in weights
                            if name.startswith(prefix) or not name.startswith(("model.", "lm_head.")))
        return super().load_weights(language_weights)


class LingBotVideoQwen3VLModel(LingBotVideoQwen3VLTextModel):
    """Compound native Qwen3-VL encoder for LingBot-Video TI2V prompts."""

    def __init__(self, config: Any) -> None:
        """Add the released visual tower without changing native language keys."""
        super().__init__(config)
        vision_config = config.vision_config
        if isinstance(vision_config, dict):
            vision_config = SimpleNamespace(**vision_config)
        self.visual = LingBotVideoQwen3VLVisionModel(vision_config)
        self.image_token_id = config.image_token_id

    @staticmethod
    def _multimodal_vision_positions(
        start_position: int,
        grid_thw: torch.Tensor,
        merge_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create temporal, height, and width positions for one image grid."""
        temporal = grid_thw[0].item()
        height = grid_thw[1].item() // merge_size
        width = grid_thw[2].item() // merge_size
        temporal_grid, height_grid, width_grid = torch.meshgrid(
            torch.arange(temporal, device=device),
            torch.arange(height, device=device) + start_position,
            torch.arange(width, device=device) + start_position,
            indexing="ij",
        )
        positions = torch.stack((temporal_grid, height_grid, width_grid), dim=0).reshape(3, -1)
        positions[0] += start_position
        return positions

    def _get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build Qwen3-VL interleaved text/image MRoPE positions for each prompt."""
        merge_size = self.visual.spatial_merge_size
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iterator = iter(image_grid_thw)
        for batch_index, token_types in enumerate(mm_token_type_ids):
            active = attention_mask[batch_index].bool() if attention_mask is not None else None
            current_types = token_types[active] if active is not None else token_types
            groups = []
            for modality, group in itertools.groupby(enumerate(current_types.tolist()), lambda pair: pair[1]):
                entries = list(group)
                groups.append((modality, entries[0][0], entries[-1][0] + 1))
            current_position = 0
            position_parts = []
            for modality, start, end in groups:
                if modality == 0:
                    length = end - start
                    text_positions = torch.arange(length, device=input_ids.device).view(1, -1).expand(3, -1)
                    position_parts.append(text_positions + current_position)
                    current_position += length
                elif modality == 1:
                    grid = next(grid_iterator)
                    position_parts.append(
                        self._multimodal_vision_positions(
                            current_position,
                            grid,
                            merge_size,
                            input_ids.device,
                        )
                    )
                    current_position += max(grid[1], grid[2]).item() // merge_size
                else:
                    raise ValueError("LingBot-Video TI2V accepts image tokens but not video tokens")
            positions = torch.cat(position_parts, dim=1)
            if active is None:
                position_ids[:, batch_index] = positions
            else:
                position_ids[:, batch_index, active] = positions
        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        """Replace image placeholders, add DeepStack features, and encode the prompt."""
        if pixel_values is None:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        if input_ids is None or inputs_embeds is not None:
            raise ValueError("multimodal conditioning requires input_ids and no inputs_embeds")
        if image_grid_thw is None or mm_token_type_ids is None:
            raise ValueError("image_grid_thw and mm_token_type_ids are required with pixel_values")
        inputs_embeds = self.get_input_embeddings(input_ids)
        visual_features, deepstack_features = self.visual(
            pixel_values.to(self.visual.patch_embed.proj.weight.dtype),
            image_grid_thw,
        )
        visual_features = visual_features.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        expected = image_mask.sum().item()
        if expected != visual_features.shape[0]:
            raise ValueError(f"image tokens ({expected}) do not match visual features ({visual_features.shape[0]})")
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), visual_features)
        if position_ids is None:
            position_ids = self._get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask)
        return super().forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack_features,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load visual tensors directly and delegate fused language tensors."""
        visual_parameters = dict(self.visual.named_parameters())
        language_weights = []
        loaded = set()
        for name, tensor in weights:
            visual_name = None
            if name.startswith("model.visual."):
                visual_name = name[len("model.visual.") :]
            elif name.startswith("visual."):
                visual_name = name[len("visual.") :]
            if visual_name is None:
                language_weights.append((name, tensor))
            elif "rotary_pos_emb.inv_freq" not in visual_name:
                default_weight_loader(visual_parameters[visual_name], tensor)
                loaded.add(f"visual.{visual_name}")
        loaded.update(super().load_weights(language_weights))
        return loaded


EntryClass = [LingBotVideoQwen3VLTextModel, LingBotVideoQwen3VLModel]
