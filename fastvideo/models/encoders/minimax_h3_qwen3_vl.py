# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL conditioner used by MiniMax H3."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.distributed import get_tp_world_size
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ColumnParallelLinear, RowParallelLinear
from fastvideo.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MiniMaxH3Qwen3VLTextRotaryEmbedding(nn.Module):
    """Shared Qwen3-VL interleaved temporal/height/width rotary embedding."""

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        head_dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta**(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(config.mrope_section)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        inv_freq = self.inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        positions = position_ids[:, :, None, :].float()
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (inv_freq.float() @ positions).transpose(2, 3)
            interleaved = frequencies[0]
            for dimension, offset in enumerate((1, 2), start=1):
                stop = self.mrope_section[dimension] * 3
                interleaved[..., offset:stop:3] = frequencies[dimension, ..., offset:stop:3]
            embedding = torch.cat((interleaved, interleaved), dim=-1)
            cos = embedding.cos()
            sin = embedding.sin()
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


class MiniMaxH3Qwen3VLTextAttention(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig, prefix: str) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError(f"Qwen3-VL attention heads {self.total_num_heads} are not divisible by TP={tp_size}")
        if tp_size > self.total_num_kv_heads:
            raise ValueError(f"Qwen3-VL native K/V projections require TP={tp_size} to be no larger than "
                             f"the {self.total_num_kv_heads} KV heads")
        if self.total_num_kv_heads % tp_size:
            raise ValueError(f"Qwen3-VL KV heads {self.total_num_kv_heads} are not divisible by TP={tp_size}")

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        quant_config = getattr(config, "quant_config", None)
        self.q_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        batch_size, sequence_length = hidden_states.shape[:2]
        query = self.q_norm(query.view(batch_size, sequence_length, self.num_heads, self.head_dim)).transpose(1, 2)
        key = self.k_norm(key.view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        if attention_mask is None:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=sequence_length > 1,
                scale=self.scaling,
                enable_gqa=self.num_heads != self.num_kv_heads,
            )
        else:
            if self.num_heads != self.num_kv_heads:
                groups = self.num_heads // self.num_kv_heads
                key = key[:, :, None].expand(-1, -1, groups, -1, -1).reshape(batch_size, self.num_heads,
                                                                             sequence_length, self.head_dim)
                value = value[:, :, None].expand(-1, -1, groups, -1, -1).reshape(batch_size, self.num_heads,
                                                                                 sequence_length, self.head_dim)
            causal = torch.ones(sequence_length, sequence_length, device=query.device, dtype=torch.bool).tril()
            key_mask = attention_mask.to(device=query.device, dtype=torch.bool)
            mask = causal[None, None] & key_mask[:, None, None]
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        output, _ = self.o_proj(output)
        return output


class MiniMaxH3Qwen3VLTextMLP(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig, prefix: str) -> None:
        super().__init__()
        quant_config = getattr(config, "quant_config", None)
        self.gate_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class MiniMaxH3Qwen3VLTextDecoderLayer(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig, prefix: str) -> None:
        super().__init__()
        self.self_attn = MiniMaxH3Qwen3VLTextAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = MiniMaxH3Qwen3VLTextMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniMaxH3Qwen3VLLanguageModel(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        quant_config = getattr(config, "quant_config", None)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )
        self.layers = nn.ModuleList(
            MiniMaxH3Qwen3VLTextDecoderLayer(config, prefix=f"{config.prefix}.language_model.layers.{index}")
            for index in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiniMaxH3Qwen3VLTextRotaryEmbedding(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
    ) -> BaseEncoderOutput:
        if attention_mask is not None and bool(attention_mask.to(torch.bool).all()):
            attention_mask = None
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        for layer_index, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states, )
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
            if deepstack_visual_embeds is not None and layer_index < len(deepstack_visual_embeds):
                if visual_pos_masks is None:
                    raise ValueError("Qwen3-VL DeepStack features require visual token positions")
                mask = visual_pos_masks.to(hidden_states.device)
                visual = deepstack_visual_embeds[layer_index].to(hidden_states.device, hidden_states.dtype)
                updated = hidden_states[mask].clone() + visual
                hidden_states[mask] = updated
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states, )
        return BaseEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class MiniMaxH3Qwen3VLVisionPatchEmbed(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        self.in_channels = config.vision_in_channels
        self.temporal_patch_size = config.vision_temporal_patch_size
        self.patch_size = config.vision_patch_size
        self.hidden_size = config.vision_hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.hidden_size, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(pixels.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


class MiniMaxH3Qwen3VLVisionRotaryEmbedding(nn.Module):

    def __init__(self, dimension: int) -> None:
        super().__init__()
        # Transformers initializes this non-persistent constant on CPU before
        # moving the model.  FastVideo constructs native encoders inside a CUDA
        # device context, where evaluating the power directly differs by a few
        # fp32 ULPs and is amplified by the 64-layer language model.
        target_device = torch.empty(0).device
        exponents = torch.arange(0, dimension, 2, dtype=torch.float32, device="cpu") / dimension
        inv_freq = (1.0 / (10000.0**exponents)).to(target_device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(positions, self.inv_freq)


class MiniMaxH3Qwen3VLVisionPatchMerger(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig, use_postshuffle_norm: bool) -> None:
        super().__init__()
        self.hidden_size = config.vision_hidden_size * config.vision_spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.vision_hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.vision_out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.view(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = F.gelu(self.linear_fc1(hidden_states))
        return self.linear_fc2(hidden_states)


class MiniMaxH3Qwen3VLVisionMLP(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states), approximate="tanh"))


class MiniMaxH3Qwen3VLVisionAttention(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        self.num_heads = config.vision_num_heads
        self.head_dim = config.vision_hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config.vision_hidden_size, config.vision_hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.vision_hidden_size, config.vision_hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).view(sequence_length, 3, self.num_heads, self.head_dim).unbind(1)
        cos, sin = position_embeddings
        query_dtype, key_dtype = query.dtype, key.dtype
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        query = (query.float() * cos + _rotate_half(query.float()) * sin).to(query_dtype)
        key = (key.float() * cos + _rotate_half(key.float()) * sin).to(key_dtype)

        query_chunks = query.split(sequence_lengths, dim=0)
        key_chunks = key.split(sequence_lengths, dim=0)
        value_chunks = value.split(sequence_lengths, dim=0)
        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(query_chunks, key_chunks, value_chunks, strict=True):
            output = F.scaled_dot_product_attention(
                query_chunk.transpose(0, 1).unsqueeze(0),
                key_chunk.transpose(0, 1).unsqueeze(0),
                value_chunk.transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(output.transpose(1, 2))
        hidden_states = torch.cat(outputs, dim=1).reshape(sequence_length, -1)
        return self.proj(hidden_states)


class MiniMaxH3Qwen3VLVisionBlock(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.vision_hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.vision_hidden_size, eps=1e-6)
        self.attn = MiniMaxH3Qwen3VLVisionAttention(config)
        self.mlp = MiniMaxH3Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), sequence_lengths, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3Qwen3VLVisionModel(nn.Module):

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.vision_spatial_merge_size
        self.patch_embed = MiniMaxH3Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.vision_num_position_embeddings, config.vision_hidden_size)
        self.num_grid_per_side = int(config.vision_num_position_embeddings**0.5)
        head_dim = config.vision_hidden_size // config.vision_num_heads
        self.rotary_pos_emb = MiniMaxH3Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(MiniMaxH3Qwen3VLVisionBlock(config) for _ in range(config.vision_depth))
        self.merger = MiniMaxH3Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = tuple(config.vision_deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            MiniMaxH3Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes)

    def _rotary_positions(self, grid_thw: torch.Tensor) -> torch.Tensor:
        max_height_width = int(grid_thw[:, 1:].max().item())
        frequency_table = self.rotary_pos_emb(max_height_width)
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        position_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=frequency_table.device)
        offset = 0
        merge = self.spatial_merge_size
        for frames_tensor, height_tensor, width_tensor in grid_thw:
            frames, height, width = int(frames_tensor), int(height_tensor), int(width_tensor)
            merged_height, merged_width = height // merge, width // merge
            block_rows = torch.arange(merged_height, device=frequency_table.device)
            block_cols = torch.arange(merged_width, device=frequency_table.device)
            intra_rows = torch.arange(merge, device=frequency_table.device)
            intra_cols = torch.arange(merge, device=frequency_table.device)
            rows = block_rows[:, None, None, None] * merge + intra_rows[None, None, :, None]
            cols = block_cols[None, :, None, None] * merge + intra_cols[None, None, None, :]
            rows = rows.expand(merged_height, merged_width, merge, merge).reshape(-1)
            cols = cols.expand(merged_height, merged_width, merge, merge).reshape(-1)
            coordinates = torch.stack((rows, cols), dim=-1).repeat(frames, 1)
            position_ids[offset:offset + coordinates.shape[0]] = coordinates
            offset += coordinates.shape[0]
        return frequency_table[position_ids].flatten(1)

    def _interpolate_position_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        index_lists: list[list[int]] = [[] for _ in range(4)]
        weight_lists: list[list[float]] = [[] for _ in range(4)]
        merge = self.spatial_merge_size
        patch_counts: list[int] = []
        grids: list[tuple[int, int, int]] = []
        for frames_tensor, height_tensor, width_tensor in grid_thw:
            frames, height, width = int(frames_tensor), int(height_tensor), int(width_tensor)
            grids.append((frames, height, width))
            patch_counts.append(height * width)
            height_positions = torch.linspace(0, self.num_grid_per_side - 1, height)
            width_positions = torch.linspace(0, self.num_grid_per_side - 1, width)
            height_floor = height_positions.int()
            width_floor = width_positions.int()
            height_ceil = (height_floor + 1).clip(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clip(max=self.num_grid_per_side - 1)
            delta_height = height_positions - height_floor
            delta_width = width_positions - width_floor
            base_height = height_floor * self.num_grid_per_side
            base_height_ceil = height_ceil * self.num_grid_per_side
            indices = (
                (base_height[:, None] + width_floor[None]).flatten(),
                (base_height[:, None] + width_ceil[None]).flatten(),
                (base_height_ceil[:, None] + width_floor[None]).flatten(),
                (base_height_ceil[:, None] + width_ceil[None]).flatten(),
            )
            weights = (
                ((1 - delta_height)[:, None] * (1 - delta_width)[None]).flatten(),
                ((1 - delta_height)[:, None] * delta_width[None]).flatten(),
                (delta_height[:, None] * (1 - delta_width)[None]).flatten(),
                (delta_height[:, None] * delta_width[None]).flatten(),
            )
            for index in range(4):
                index_lists[index].extend(indices[index].tolist())
                weight_lists[index].extend(weights[index].tolist())

        index_tensor = torch.tensor(index_lists, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_lists,
                                     dtype=self.pos_embed.weight.dtype,
                                     device=self.pos_embed.weight.device)
        embeddings = self.pos_embed(index_tensor) * weight_tensor[:, :, None]
        embeddings = (embeddings[0] + embeddings[1] + embeddings[2] + embeddings[3]).split(patch_counts)
        permuted = []
        for embedding, (frames, height, width) in zip(embeddings, grids, strict=True):
            embedding = embedding.repeat(frames, 1)
            embedding = embedding.view(frames, height // merge, merge, width // merge, merge, -1)
            permuted.append(embedding.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(permuted)

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(pixels)
        hidden_states = hidden_states + self._interpolate_position_embeddings(grid_thw)
        rotary = self._rotary_positions(grid_thw).reshape(hidden_states.shape[0], -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        sequence_lengths = [
            int(value) for value in torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        ]
        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, sequence_lengths, position_embeddings)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.deepstack_merger_list[merger_index](hidden_states))
        return self.merger(hidden_states), deepstack_features


class MiniMaxH3Qwen3VLConditioner(TextEncoder):
    """FastVideo-native Qwen3-VL body without the unused language-model head."""

    supports_hf_from_pretrained = False

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__(config)
        self.visual = MiniMaxH3Qwen3VLVisionModel(config)
        self.language_model = MiniMaxH3Qwen3VLLanguageModel(config)

    @property
    def dtype(self) -> torch.dtype:
        parameter = next(self.parameters(), None)
        return torch.float32 if parameter is None else parameter.dtype

    @property
    def num_hidden_layers(self) -> int:
        return self.config.num_hidden_layers

    def _get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is None:
                return torch.arange(input_ids.shape[1],
                                    device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            positions = attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(attention_mask == 0, 1)
            return positions.unsqueeze(0).expand(3, -1, -1)

        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        position_ids = torch.ones(3,
                                  input_ids.shape[0],
                                  input_ids.shape[1],
                                  dtype=input_ids.dtype,
                                  device=input_ids.device)
        image_index = 0
        video_index = 0
        merge = self.config.vision_spatial_merge_size
        for batch_index, sequence in enumerate(input_ids):
            sequence = sequence[mask[batch_index] == 1]
            starts = torch.argwhere(sequence == self.config.vision_start_token_id).squeeze(1)
            vision_tokens = sequence[starts + 1]
            image_count = int((vision_tokens == self.config.image_token_id).sum())
            video_count = int((vision_tokens == self.config.video_token_id).sum())
            tokens = sequence.tolist()
            pieces = []
            start = 0
            remaining_images, remaining_videos = image_count, video_count
            for _ in range(image_count + video_count):
                image_end = tokens.index(self.config.image_token_id, start) if remaining_images else len(tokens) + 1
                video_end = tokens.index(self.config.video_token_id, start) if remaining_videos else len(tokens) + 1
                if image_end < video_end:
                    if image_grid_thw is None:
                        raise ValueError("Qwen3-VL image tokens require image_grid_thw")
                    frames, height, width = (int(value) for value in image_grid_thw[image_index])
                    image_index += 1
                    remaining_images -= 1
                    end = image_end
                else:
                    if video_grid_thw is None:
                        raise ValueError("Qwen3-VL video tokens require video_grid_thw")
                    frames, height, width = (int(value) for value in video_grid_thw[video_index])
                    video_index += 1
                    remaining_videos -= 1
                    end = video_end
                grid_frames, grid_height, grid_width = frames, height // merge, width // merge
                text_length = end - start
                offset = int(pieces[-1].max()) + 1 if pieces else 0
                pieces.append(torch.arange(text_length).view(1, -1).expand(3, -1) + offset)
                temporal = torch.arange(grid_frames).view(-1, 1).expand(-1, grid_height * grid_width).flatten()
                rows = torch.arange(grid_height).view(1, -1, 1).expand(grid_frames, -1, grid_width).flatten()
                columns = torch.arange(grid_width).view(1, 1, -1).expand(grid_frames, grid_height, -1).flatten()
                pieces.append(torch.stack((temporal, rows, columns)) + text_length + offset)
                start = end + grid_frames * grid_height * grid_width
            if start < len(tokens):
                offset = int(pieces[-1].max()) + 1 if pieces else 0
                pieces.append(torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + offset)
            positions = torch.cat(pieces, dim=1).to(position_ids.device)
            position_ids[:, batch_index, mask[batch_index] == 1] = positions
        return position_ids

    def _visual_features(
        self,
        pixels: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.visual(pixels.to(self.visual.patch_embed.proj.weight.dtype), grid_thw)

    @staticmethod
    def _placeholder_mask(
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        token_id: int,
        features: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        mask = input_ids == token_id
        expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[expanded].numel() != features.numel():
            raise ValueError(f"Qwen3-VL {label} features and placeholder tokens do not match: "
                             f"tokens={int(mask.sum())}, features={features.shape[0]}")
        return mask

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        del mm_token_type_ids, kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds is required")
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if input_ids is None and (pixel_values is not None or pixel_values_videos is not None):
            raise ValueError("Multimodal Qwen3-VL inputs require input_ids for placeholder matching")

        image_mask = None
        video_mask = None
        image_deepstack = None
        video_deepstack = None
        if pixel_values is not None:
            if input_ids is None or image_grid_thw is None:
                raise ValueError("pixel_values require input_ids and image_grid_thw")
            image_features, image_deepstack = self._visual_features(pixel_values, image_grid_thw)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self._placeholder_mask(input_ids, inputs_embeds, self.config.image_token_id, image_features,
                                                "image")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features)
        if pixel_values_videos is not None:
            if input_ids is None or video_grid_thw is None:
                raise ValueError("pixel_values_videos require input_ids and video_grid_thw")
            video_features, video_deepstack = self._visual_features(pixel_values_videos, video_grid_thw)
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = self._placeholder_mask(input_ids, inputs_embeds, self.config.video_token_id, video_features,
                                                "video")
            inputs_embeds = inputs_embeds.masked_scatter(video_mask.unsqueeze(-1), video_features)

        visual_mask = None
        deepstack_features = None
        if image_mask is not None and video_mask is not None:
            visual_mask = image_mask | video_mask
            deepstack_features = []
            image_joint = image_mask[visual_mask]
            video_joint = video_mask[visual_mask]
            assert image_deepstack is not None and video_deepstack is not None
            for image_features, video_features in zip(image_deepstack, video_deepstack, strict=True):
                combined = image_features.new_zeros((int(visual_mask.sum()), image_features.shape[-1]))
                combined[image_joint] = image_features
                combined[video_joint] = video_features
                deepstack_features.append(combined)
        elif image_mask is not None:
            visual_mask = image_mask
            deepstack_features = image_deepstack
        elif video_mask is not None:
            visual_mask = video_mask
            deepstack_features = video_deepstack

        if position_ids is None:
            if input_ids is None:
                sequence_length = inputs_embeds.shape[1]
                position_ids = torch.arange(sequence_length,
                                            device=inputs_embeds.device).view(1, 1,
                                                                              -1).expand(3, inputs_embeds.shape[0], -1)
            else:
                position_ids = self._get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        outputs = self.language_model(
            inputs_embeds,
            position_ids,
            attention_mask,
            output_hidden_states,
            visual_mask,
            deepstack_features,
        )
        outputs.attention_mask = attention_mask
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()

        for source_name, tensor in weights:
            if source_name == "lm_head.weight":
                continue
            name = source_name[6:] if source_name.startswith("model.") else source_name
            if name not in parameters:
                raise ValueError(f"Unexpected MiniMax-H3 Qwen3-VL checkpoint key: {source_name}")
            parameter = parameters[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(name)
        return loaded


EntryClass = MiniMaxH3Qwen3VLConditioner

__all__ = ["MiniMaxH3Qwen3VLConditioner"]
