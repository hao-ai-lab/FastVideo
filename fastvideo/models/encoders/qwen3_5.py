# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open
from tokenizers import Regex, pre_tokenizers
from torch import nn
from transformers import AutoTokenizer

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.qwen3_5 import Magi2Qwen35Config
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.utils import set_default_torch_dtype
from fastvideo.models.loader.weight_utils import default_weight_loader


_LANGUAGE_MODEL_PREFIX = "model.language_model."


def _iter_indexed_language_model_weights(
    model_path: Path,
    weight_map: dict[str, str],
    device: torch.device,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Read only Qwen3.5 language-model tensors from indexed checkpoint shards."""
    weights_by_shard: dict[str, list[str]] = {}
    for checkpoint_name, shard_name in weight_map.items():
        if checkpoint_name.startswith(_LANGUAGE_MODEL_PREFIX):
            weights_by_shard.setdefault(shard_name, []).append(checkpoint_name)

    shard_paths = {
        shard_name: model_path / shard_name for shard_name in weights_by_shard
    }
    missing_shards = [str(path) for path in shard_paths.values() if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(
            "Missing Qwen3.5 language-model weight shards: "
            + ", ".join(sorted(missing_shards))
        )

    for shard_name in sorted(weights_by_shard):
        with safe_open(
            str(shard_paths[shard_name]),
            framework="pt",
            device=str(device),
        ) as shard:
            for checkpoint_name in sorted(weights_by_shard[shard_name]):
                yield checkpoint_name, shard.get_tensor(checkpoint_name)


def strip_empty(obj):
    if isinstance(obj, dict):
        return {
            key: value
            for key, value in ((key, strip_empty(value)) for key, value in obj.items())
            if value is not None and value != [] and value != {}
        }
    if isinstance(obj, list):
        return [strip_empty(value) for value in obj if value is not None]
    return obj


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _pick(values: dict, keys: Iterable[str]) -> list[tuple[str, Any]]:
    return [
        (key, values[key])
        for key in keys
        if key in values and values[key] not in (None, [], {})
    ]


def json_to_compact_markdown(raw_json: str | dict[str, Any]) -> str:
    """Flatten a structured MAGI-2 prompt while preserving semantic sections."""
    obj = json.loads(raw_json.strip()) if isinstance(raw_json, str) else raw_json

    obj = strip_empty(obj)
    if not isinstance(obj, dict) or "global_layer" not in obj:
        return (
            raw_json
            if isinstance(raw_json, str)
            else json.dumps(raw_json, ensure_ascii=False)
        )

    global_layer = _as_dict(obj.get("global_layer"))
    dynamic_layer = _as_dict(obj.get("dynamic_layer"))
    reference_layer = obj.get("reference_layer", [])

    lines: list[str] = []
    context = _one_line(global_layer.get("context"))
    description = _one_line(global_layer.get("description"))
    if context or description:
        lines.append(f"context: {context}" if context else "context")
        if description:
            lines.append(description)

    aesthetics = _as_dict(global_layer.get("aesthetics"))
    aesthetic_parts = []
    for label, key in (
        ("style", "style"),
        ("mood", "mood_atmosphere"),
        ("color", "color_scheme"),
    ):
        value = _one_line(aesthetics.get(key))
        if value:
            aesthetic_parts.append(f"{label}={value}")
    if aesthetic_parts:
        lines.append("aesthetics: " + "; ".join(aesthetic_parts))

    audio = _as_dict(global_layer.get("audio_baseline"))
    dialogue = _as_dict(audio.get("dialogue"))
    audio_parts = []
    language = _one_line(dialogue.get("language"))
    speakers = _as_list(dialogue.get("speaker_tags"))
    ambience = _one_line(audio.get("ambience"))
    if language:
        audio_parts.append(f"language={language}")
    if speakers:
        audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
    if ambience:
        audio_parts.append(f"ambience={ambience}")
    if audio_parts:
        lines.append("audio: " + "; ".join(audio_parts))

    subjects = global_layer.get("alive_subjects_static")
    if isinstance(subjects, list) and subjects:
        lines.append("subjects:")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            sid = _one_line(subject.get("subject_id"))
            desc = _one_line(subject.get("description"))
            position = _one_line(subject.get("position"))
            orientation = _one_line(subject.get("orientation"))
            attrs = []
            for key, value in _pick(
                _as_dict(subject.get("visual_attributes")),
                ["gender", "age_group", "ethnicity", "clothing", "appearance_details"],
            ):
                value = _one_line(value)
                if value:
                    attrs.append(f"{key}={value}")
            row = " - " + " | ".join([part for part in [sid, desc] if part])
            extra = "; ".join([part for part in [position, orientation] if part])
            if extra:
                row += f" ({extra})"
            if attrs:
                row += " :: " + "; ".join(attrs)
            lines.append(row)

    objects = global_layer.get("objects_static")
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for obj_item in objects:
            if not isinstance(obj_item, dict):
                continue
            oid = _one_line(obj_item.get("object_id"))
            desc = _one_line(obj_item.get("description"))
            shape = _one_line(obj_item.get("shape_and_color"))
            position = _one_line(obj_item.get("position"))
            row = " - " + " | ".join([part for part in [oid, desc] if part])
            details = "; ".join([part for part in [shape, position] if part])
            if details:
                row += " :: " + details
            lines.append(row)

    segments = dynamic_layer.get("timeline_segments")
    if isinstance(segments, list) and segments:
        lines.append("timeline:")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            basic = _as_dict(segment.get("segment_basic_info"))
            timestamp = _one_line(basic.get("timestamp_range"))
            desc = _one_line(basic.get("segment_description"))
            head = f" - {timestamp}" if timestamp else " -"
            if desc:
                head += f" {desc}"
            lines.append(head.rstrip())

            segment_audio = _as_dict(segment.get("audio"))
            for dialogue_line in segment_audio.get("dialogue_lines", []) or []:
                if not isinstance(dialogue_line, dict):
                    continue
                speaker = _one_line(dialogue_line.get("speaker"))
                text = _one_line(dialogue_line.get("text"))
                timestamp = _one_line(dialogue_line.get("timestamp"))
                if text:
                    prefix = (
                        f"   - dialogue {speaker}: " if speaker else "   - dialogue: "
                    )
                    line = prefix + text
                    if timestamp:
                        line += f" ({timestamp})"
                    lines.append(line)

            for alive in segment.get("alive_subjects", []) or []:
                if not isinstance(alive, dict):
                    continue
                sid = _one_line(alive.get("subject_id"))
                action = _as_dict(alive.get("action"))
                parts = [
                    _one_line(action.get("primary_action")),
                    _one_line(action.get("interaction")),
                    _one_line(action.get("facial_expression")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - action {sid}: " + "; ".join(parts))

            for obj_item in segment.get("objects", []) or []:
                if not isinstance(obj_item, dict):
                    continue
                oid = _one_line(obj_item.get("object_id"))
                state = _as_dict(obj_item.get("dynamic_state"))
                parts = [
                    _one_line(state.get("state_change")),
                    _one_line(state.get("motion_detail")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - objects {oid}: " + "; ".join(parts))

    if isinstance(reference_layer, list):
        lines.extend(str(line) for line in reference_layer if str(line).strip())

    return "\n".join(line for line in lines if line.strip())


def _l2norm(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize query or key vectors with the Qwen3.5 epsilon convention."""
    inverse_norm = torch.rsqrt((tensor * tensor).sum(dim=dim, keepdim=True) + eps)
    return tensor * inverse_norm


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Evaluate the Qwen3.5 chunked gated-delta recurrence in FP32."""
    output_dtype = query.dtype
    query = _l2norm(query)
    key = _l2norm(key)
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, decay)
    ]

    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    decay = F.pad(decay, (0, pad_size))
    padded_length = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, value, key_beta, value_beta = [
        tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            -1,
            chunk_size,
            tensor.shape[-1],
        )
        for tensor in (query, key, value, key_beta, value_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    diagonal_mask = torch.triu(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=0,
    )

    decay = decay.cumsum(dim=-1)
    decay_mask = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    attention = -(
        (key_beta @ key.transpose(-1, -2)) * decay_mask
    ).masked_fill(diagonal_mask, 0)
    for row_index in range(1, chunk_size):
        row = attention[..., row_index, :row_index].clone()
        preceding_rows = attention[..., :row_index, :row_index].clone()
        attention[..., row_index, :row_index] = row + (
            row.unsqueeze(-1) * preceding_rows
        ).sum(-2)
    attention = attention + torch.eye(
        chunk_size,
        dtype=attention.dtype,
        device=attention.device,
    )
    value = attention @ value_beta
    cumulative_key = attention @ (key_beta * decay.exp().unsqueeze(-1))
    recurrent_state = torch.zeros(
        batch_size,
        num_heads,
        key_head_dim,
        value_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    recurrence_output = torch.zeros_like(value)
    causal_mask = torch.triu(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=1,
    )

    for chunk_index in range(padded_length // chunk_size):
        chunk_query = query[:, :, chunk_index]
        chunk_key = key[:, :, chunk_index]
        chunk_value = value[:, :, chunk_index]
        chunk_attention = (
            chunk_query
            @ chunk_key.transpose(-1, -2)
            * decay_mask[:, :, chunk_index]
        ).masked_fill_(causal_mask, 0)
        recurrent_value = cumulative_key[:, :, chunk_index] @ recurrent_state
        value_delta = chunk_value - recurrent_value
        inter_chunk_attention = (
            chunk_query * decay[:, :, chunk_index, :, None].exp()
        ) @ recurrent_state
        recurrence_output[:, :, chunk_index] = (
            inter_chunk_attention + chunk_attention @ value_delta
        )
        recurrent_state = (
            recurrent_state
            * decay[:, :, chunk_index, -1, None, None].exp()
            + (
                chunk_key
                * (
                    decay[:, :, chunk_index, -1, None]
                    - decay[:, :, chunk_index]
                ).exp()[..., None]
            ).transpose(-1, -2)
            @ value_delta
        )

    recurrence_output = recurrence_output.reshape(
        recurrence_output.shape[0],
        recurrence_output.shape[1],
        -1,
        recurrence_output.shape[-1],
    )
    recurrence_output = recurrence_output[:, :, :sequence_length]
    return recurrence_output.transpose(1, 2).contiguous().to(output_dtype)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of each rotary-embedding vector."""
    first_half = tensor[..., : tensor.shape[-1] // 2]
    second_half = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary embeddings to Qwen3.5 query and key heads."""
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotary_dim = cosine.shape[-1]
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rotary = query_rotary * cosine + _rotate_half(query_rotary) * sine
    key_rotary = key_rotary * cosine + _rotate_half(key_rotary) * sine
    return (
        torch.cat((query_rotary, query_pass), dim=-1),
        torch.cat((key_rotary, key_pass), dim=-1),
    )


class Qwen35RMSNormGated(nn.Module):
    """Apply FP32 root-mean-square normalization before a SiLU gate."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize value heads and apply their learned input gates."""
        input_dtype = hidden_states.dtype
        normalized_states = hidden_states.to(torch.float32)
        variance = normalized_states.pow(2).mean(-1, keepdim=True)
        normalized_states = normalized_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        normalized_states = self.weight * normalized_states.to(input_dtype)
        normalized_states = normalized_states * F.silu(gate.to(torch.float32))
        return normalized_states.to(input_dtype)


class Qwen35GatedDeltaNet(nn.Module):
    """Mix tokens with Qwen3.5 causal convolution and gated delta recurrence."""

    def __init__(self, config: Magi2Qwen35Config, layer_index: int) -> None:
        """Create the projections and recurrence parameters for one linear layer."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_value_heads = config.linear_num_value_heads
        self.num_key_heads = config.linear_num_key_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.key_head_dim * self.num_key_heads
        self.value_dim = self.value_head_dim * self.num_value_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_index = layer_index
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        transition = torch.empty(self.num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(transition))
        self.norm = Qwen35RMSNormGated(
            self.value_head_dim,
            eps=config.rms_norm_eps,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(
            self.hidden_size,
            self.num_value_heads,
            bias=False,
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size,
            self.num_value_heads,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform one complete prompt without recurrent cache state."""
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        gate = self.in_proj_z(hidden_states).reshape(
            batch_size,
            sequence_length,
            -1,
            self.value_head_dim,
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay_input = self.in_proj_a(hidden_states)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :sequence_length])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.reshape(
            batch_size,
            sequence_length,
            -1,
            self.key_head_dim,
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            -1,
            self.key_head_dim,
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            -1,
            self.value_head_dim,
        )
        decay = -self.A_log.float().exp() * F.softplus(
            decay_input.float() + self.dt_bias
        )
        head_repeat = self.num_value_heads // self.num_key_heads
        query = query.repeat_interleave(head_repeat, dim=2)
        key = key.repeat_interleave(head_repeat, dim=2)
        mixed_output = _torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
        )
        mixed_output = mixed_output.reshape(-1, self.value_head_dim)
        gate = gate.reshape(-1, self.value_head_dim)
        mixed_output = self.norm(mixed_output, gate)
        mixed_output = mixed_output.reshape(batch_size, sequence_length, -1)
        return self.out_proj(mixed_output)


class Qwen35RotaryEmbedding(nn.Module):
    """Construct Qwen3.5 partial rotary embeddings in FP32."""

    def __init__(self, config: Magi2Qwen35Config) -> None:
        """Create inverse frequencies for the checkpoint's partial rotary span."""
        super().__init__()
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inverse_frequency = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(
                    dtype=torch.float32
                )
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)
        self.mrope_section = config.mrope_section

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine tensors for three identical text coordinates."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                3,
                position_ids.shape[0],
                -1,
            )
        inverse_frequency = self.inv_freq[None, None, :, None].float().expand(
            3,
            position_ids.shape[1],
            -1,
            1,
        )
        expanded_positions = position_ids[:, :, None, :].float()
        device_type = (
            hidden_states.device.type
            if hidden_states.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (
                inverse_frequency.float() @ expanded_positions.float()
            ).transpose(2, 3)
            text_frequencies = frequencies[0]
            for dimension, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dimension] * 3
                indices = slice(offset, length, 3)
                text_frequencies[..., indices] = frequencies[
                    dimension,
                    ...,
                    indices,
                ]
            embedding = torch.cat((text_frequencies, text_frequencies), dim=-1)
            cosine = embedding.cos()
            sine = embedding.sin()
        return (
            cosine.to(dtype=hidden_states.dtype),
            sine.to(dtype=hidden_states.dtype),
        )


class Qwen35RMSNorm(nn.Module):
    """Apply Qwen3.5 one-centered RMS normalization."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and cast the scaled output to the input dtype."""
        normalized_states = hidden_states.float()
        normalized_states = normalized_states * torch.rsqrt(
            normalized_states.pow(2).mean(-1, keepdim=True) + self.eps
        )
        normalized_states = normalized_states * (1.0 + self.weight.float())
        return normalized_states.type_as(hidden_states)


class Qwen35Attention(nn.Module):
    """Apply grouped-query causal self-attention with an output gate."""

    def __init__(self, config: Magi2Qwen35Config, layer_index: int) -> None:
        """Create the query, key, value, gate, and output projections."""
        super().__init__()
        self.layer_index = layer_index
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen35RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run the complete-prompt SDPA path selected by Transformers 5.5.0."""
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_value_shape = (
            *input_shape,
            self.num_key_value_heads,
            self.head_dim,
        )
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape,
                self.num_attention_heads,
                self.head_dim * 2,
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(key_value_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)
        query_states, key_states = _apply_rotary_pos_emb(
            query_states,
            key_states,
            *position_embeddings,
        )
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=query_states.shape[2] > 1,
            scale=self.scaling,
            enable_gqa=True,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        attention_output = attention_output * torch.sigmoid(gate)
        return self.o_proj(attention_output)


class Qwen35MLP(nn.Module):
    """Apply the Qwen3.5 SwiGLU feed-forward network."""

    def __init__(self, config: Magi2Qwen35Config) -> None:
        """Create separate gate, value, and output projections for SwiGLU."""
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the gated SiLU activation."""
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen35DecoderLayer(nn.Module):
    """Combine one Qwen3.5 token mixer and one feed-forward residual block."""

    def __init__(self, config: Magi2Qwen35Config, layer_index: int) -> None:
        """Select the token mixer and create the layer's residual modules."""
        super().__init__()
        self.layer_type = (
            "full_attention"
            if (layer_index + 1) % config.full_attention_interval == 0
            else "linear_attention"
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_index)
        else:
            self.self_attn = Qwen35Attention(config, layer_index)
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply pre-normalized token mixing and feed-forward residuals."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Magi2Qwen35TextEncoder(TextEncoder):
    """Encode MAGI-2 prompts with the text-only Qwen3.5-27B architecture."""

    supports_hf_from_pretrained = True

    def __init__(self, config: Magi2Qwen35Config) -> None:
        """Create the prompt-processing state for the release Qwen3.5 model."""
        super().__init__(config)
        self.text_model: nn.Module | None = None
        self.max_length = config.text_len
        self.skip_layer = config.hidden_state_skip_layer
        self.tokenizer = None

    @classmethod
    def from_pretrained_local(
        cls,
        model_path: str,
        model_config: Magi2Qwen35Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Magi2Qwen35TextEncoder:
        """Load the same Transformers Qwen3.5 text path as the official release."""
        from transformers import Qwen3_5TextModel

        with set_default_torch_dtype(dtype):
            encoder = cls(model_config)
        encoder.text_model = Qwen3_5TextModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        encoder.to(device)
        encoder.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            local_files_only=True,
        )
        cjk_split = pre_tokenizers.Split(
            pattern=Regex(
                r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
            ),
            behavior="isolated",
        )
        original_pre_tokenizer = encoder.tokenizer.backend_tokenizer.pre_tokenizer
        encoder.tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [cjk_split, original_pre_tokenizer]
        )
        encoder.requires_grad_(False)
        return encoder.eval()

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load only the text-model tensors from the composite Qwen3.5 checkpoint."""
        parameter_by_name = dict(self.named_parameters())
        loaded_parameters: set[str] = set()
        for checkpoint_name, checkpoint_tensor in weights:
            if not checkpoint_name.startswith(_LANGUAGE_MODEL_PREFIX):
                continue
            parameter_name = checkpoint_name[len(_LANGUAGE_MODEL_PREFIX) :]
            if parameter_name not in parameter_by_name:
                raise KeyError(
                    f"Unexpected Qwen3.5 text parameter: {checkpoint_name}"
                )
            default_weight_loader(
                parameter_by_name[parameter_name],
                checkpoint_tensor,
            )
            loaded_parameters.add(parameter_name)
        return loaded_parameters

    def _normalize_prompt(self, prompt: str) -> str:
        """Convert structured JSON prompts into the compact MAGI-2 text form."""
        try:
            parsed_json = json.loads(prompt)
            if isinstance(parsed_json, (dict, list)):
                return json_to_compact_markdown(prompt)
        except (json.JSONDecodeError, TypeError):
            pass
        return prompt

    def _require_tokenizer(self):
        """Return the tokenizer initialized by the local checkpoint loader."""
        if self.tokenizer is None:
            raise RuntimeError(
                "Load Magi2Qwen35TextEncoder through from_pretrained_local()."
            )
        return self.tokenizer

    def get_target_token_indices(
        self,
        prompt: str,
        target_str: str | None,
    ) -> list[int] | None:
        """Map one figure marker's character span to tokenizer indices."""
        if not target_str:
            return None
        prompt = self._normalize_prompt(prompt)
        tokenizer = self._require_tokenizer()
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )
        offsets = inputs["offset_mapping"][0]
        start_char_index = prompt.find(target_str)
        if start_char_index == -1:
            return None
        end_char_index = start_char_index + len(target_str)
        token_indices: list[int] = []
        for token_index, (start, end) in enumerate(offsets):
            if start == 0 and end == 0 and token_index != 0:
                continue
            if max(start, start_char_index) < min(end, end_char_index):
                token_indices.append(token_index)
        return token_indices

    def get_special_token(
        self,
        prompt: str,
        target_strs: list[str],
        text_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Pool prompt embeddings over requested figure-marker token spans."""
        embeddings = []
        for target_str in target_strs:
            token_indices = self.get_target_token_indices(prompt, target_str)
            if token_indices:
                embeddings.append(
                    text_feature[0, token_indices, :].mean(dim=0).clone()
                )
            else:
                embeddings.append(
                    torch.zeros(
                        text_feature.shape[-1],
                        device=text_feature.device,
                        dtype=text_feature.dtype,
                    )
                )
        return torch.stack(embeddings, dim=0)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        """Tokenize one prompt and return the hidden state before the final two layers."""
        normalized_prompt = self._normalize_prompt(prompt)
        tokenizer = self._require_tokenizer()
        if self.text_model is None:
            raise RuntimeError("The Qwen3.5 text model has not been loaded")
        device = next(self.text_model.parameters()).device
        inputs = tokenizer(
            [normalized_prompt],
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        ).to(device)
        outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        if self.skip_layer == 0:
            return outputs.last_hidden_state
        assert outputs.hidden_states is not None
        return outputs.hidden_states[-(self.skip_layer + 1)]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        """Delegate one prompt to the Transformers model used by MAGI-2."""
        if self.text_model is None:
            raise RuntimeError("The Qwen3.5 text model has not been loaded")
        outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return BaseEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attention_mask=attention_mask,
        )


EntryClass = Magi2Qwen35TextEncoder
