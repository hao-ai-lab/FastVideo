# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import torch
from torch import nn
from transformers import Gemma3ForConditionalGeneration

from fastvideo.configs.models.encoders import BaseEncoderOutput, TextEncoderConfig
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.dits.ltx2 import (
    FeedForward,
    LTXRopeType,
    apply_ltx_rotary_emb,
    generate_ltx_freq_grid_np,
    generate_ltx_freq_grid_pytorch,
    precompute_ltx_freqs_cis,
)
from fastvideo.models.loader.weight_utils import default_weight_loader
from fastvideo.platforms import AttentionBackendEnum


def _debug_log_line(message: str) -> None:
    if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") != "1":
        return
    log_path = os.getenv("LTX2_PIPELINE_DEBUG_PATH", "")
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _debug_gemma_log_line(message: str) -> None:
    log_path = os.getenv("LTX2_FASTVIDEO_GEMMA_LOG", "")
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


@dataclass(frozen=True)
class GemmaConnectorConfig:
    num_attention_heads: int
    attention_head_dim: int
    num_layers: int
    positional_embedding_theta: float
    positional_embedding_max_pos: list[int]
    rope_type: LTXRopeType
    double_precision_rope: bool
    num_learnable_registers: int | None


class GemmaFeaturesExtractorProjLinear(nn.Module):
    """Linear projection that aggregates stacked Gemma hidden states."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.aggregate_embed = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate_embed(x)


class _BasicTransformerBlock1D(nn.Module):
    """1D transformer block for connector processing."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn1 = _GemmaAttention(
            query_dim=dim,
            context_dim=None,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
        )
        self.ff = FeedForward(dim, dim_out=dim)
        self.norm_eps = norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = torch.nn.functional.rms_norm(
            hidden_states, (hidden_states.shape[-1],), eps=self.norm_eps
        )
        if norm_hidden_states.ndim == 4:
            norm_hidden_states = norm_hidden_states.squeeze(1)

        attn_output = self.attn1(
            norm_hidden_states,
            mask=attention_mask,
            pe=pe,
        )
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = torch.nn.functional.rms_norm(
            hidden_states, (hidden_states.shape[-1],), eps=self.norm_eps
        )
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class _GemmaAttention(nn.Module):
    """Attention implementation aligned with LTX-2 text encoder."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None,
        heads: int,
        dim_head: int,
        norm_eps: float,
        rope_type: LTXRopeType,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.rope_type = rope_type

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, bias=True), nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_ltx_rotary_emb(q, pe, self.rope_type)
            k = apply_ltx_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        b, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.view(b, q_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, k_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, k_len, self.heads, self.dim_head).transpose(1, 2)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, q_len, -1)
        return self.to_out(out)


class Embeddings1DConnector(nn.Module):
    """Transformer connector that refines Gemma embeddings for LTX-2."""

    _supports_gradient_checkpointing = True

    def __init__(self, config: GemmaConnectorConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.positional_embedding_theta = config.positional_embedding_theta
        self.positional_embedding_max_pos = config.positional_embedding_max_pos
        self.rope_type = config.rope_type
        self.double_precision_rope = config.double_precision_rope
        self.transformer_1d_blocks = nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=config.num_attention_heads,
                    dim_head=config.attention_head_dim,
                    rope_type=config.rope_type,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.num_learnable_registers = config.num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = nn.Parameter(
                torch.rand(
                    self.num_learnable_registers,
                    self.inner_dim,
                    dtype=torch.bfloat16,
                )
                * 2.0
                - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[1] % self.num_learnable_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by "
            f"num_learnable_registers {self.num_learnable_registers}."
        )

        num_registers_duplications = (
            hidden_states.shape[1] // self.num_learnable_registers
        )
        learnable_registers = torch.tile(
            self.learnable_registers, (num_registers_duplications, 1)
        )
        attention_mask_binary = (
            attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0
        ).int()

        non_zero_hidden_states = hidden_states[
            :, attention_mask_binary.squeeze().bool(), :
        ]
        non_zero_nums = non_zero_hidden_states.shape[1]
        pad_length = hidden_states.shape[1] - non_zero_nums
        adjusted_hidden_states = torch.nn.functional.pad(
            non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0
        )
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        hidden_states = flipped_mask * adjusted_hidden_states + (
            1 - flipped_mask
        ) * learnable_registers

        attention_mask = torch.full_like(
            attention_mask,
            0.0,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_learnable_registers:
            hidden_states, attention_mask = (
                self._replace_padded_with_learnable_registers(
                    hidden_states, attention_mask
                )
            )

        indices_grid = torch.arange(
            hidden_states.shape[1],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        indices_grid = indices_grid[None, None, :]
        freq_grid_generator = (
            generate_ltx_freq_grid_np
            if self.double_precision_rope
            else generate_ltx_freq_grid_pytorch
        )
        freqs_cis = precompute_ltx_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, pe=freqs_cis
            )

        hidden_states = torch.nn.functional.rms_norm(
            hidden_states, (hidden_states.shape[-1],), eps=1e-6
        )

        return hidden_states, attention_mask


class LTX2GemmaTextEncoderModel(TextEncoder):
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        arch = config.arch_config

        self.feature_extractor_linear = GemmaFeaturesExtractorProjLinear(
            in_features=arch.feature_extractor_in_features,
            out_features=arch.feature_extractor_out_features,
        )

        connector_config = GemmaConnectorConfig(
            num_attention_heads=arch.connector_num_attention_heads,
            attention_head_dim=arch.connector_attention_head_dim,
            num_layers=arch.connector_num_layers,
            positional_embedding_theta=arch.connector_positional_embedding_theta,
            positional_embedding_max_pos=arch.connector_positional_embedding_max_pos,
            rope_type=LTXRopeType(arch.connector_rope_type),
            double_precision_rope=arch.connector_double_precision_rope,
            num_learnable_registers=arch.connector_num_learnable_registers,
        )
        self.embeddings_connector = Embeddings1DConnector(connector_config)
        self.audio_embeddings_connector = Embeddings1DConnector(connector_config)

        self.gemma_model_path = arch.gemma_model_path
        self.gemma_dtype = arch.gemma_dtype
        self.padding_side = arch.padding_side
        self._gemma_model: Gemma3ForConditionalGeneration | None = None

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, param in super().named_parameters(
            prefix=prefix, recurse=recurse
        ):
            if name.startswith("gemma_model."):
                continue
            yield name, param

    @property
    def gemma_model(self) -> Gemma3ForConditionalGeneration:
        if self._gemma_model is None:
            gemma_path = self.gemma_model_path
            if not gemma_path:
                raise ValueError(
                    "gemma_model_path must be set (expected text_encoder/gemma)."
                )
            dtype = getattr(torch, self.gemma_dtype, torch.bfloat16)
            self._gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                gemma_path,
                local_files_only=True,
                torch_dtype=dtype,
            )
            # Configure model-level attention implementation when using TORCH_SDPA.
            # Note: torch.backends.cuda.enable_*_sdp() settings should be configured
            # at application/pipeline initialization level, not here, to avoid
            # unexpected side effects across the application.
            if os.getenv("FASTVIDEO_ATTENTION_BACKEND") == "TORCH_SDPA":
                if hasattr(self._gemma_model.config, "attn_implementation"):
                    self._gemma_model.config.attn_implementation = "sdpa"
                if hasattr(self._gemma_model.config, "_attn_implementation"):
                    self._gemma_model.config._attn_implementation = "sdpa"
            device = next(self.feature_extractor_linear.parameters()).device
            self._gemma_model.to(device=device)
            self._gemma_model.eval()
        return self._gemma_model

    def _run_feature_extractor(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str,
    ) -> torch.Tensor:
        encoded_text_features = torch.stack(hidden_states, dim=-1)
        if os.getenv("LTX2_FASTVIDEO_GEMMA_LOG", ""):
            for idx, layer in enumerate(hidden_states):
                _debug_gemma_log_line(
                    f"fastvideo:gemma_hidden_state_{idx}"
                    f":sum={layer.float().sum().item():.6f}"
                )
            _debug_gemma_log_line(
                "fastvideo:gemma_hidden_states_stack"
                f":sum={encoded_text_features.float().sum().item():.6f}"
            )
        encoded_text_features_dtype = encoded_text_features.dtype
        sequence_lengths = attention_mask.sum(dim=-1)
        normed_text_features = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )
        return self.feature_extractor_linear(
            normed_text_features.to(encoded_text_features_dtype)
        )

    def _convert_to_additive_mask(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        return (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(dtype).max

    def _run_connectors(
        self,
        encoded_input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(
            attention_mask, encoded_input.dtype
        )
        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input, connector_attention_mask
        )

        attention_mask = (encoded_connector_attention_mask < 0.000001).to(
            torch.int64
        )
        attention_mask = attention_mask.reshape(
            [encoded.shape[0], encoded.shape[1], 1]
        )
        encoded = encoded * attention_mask

        encoded_for_audio, _ = self.audio_embeddings_connector(
            encoded_input, connector_attention_mask
        )

        return encoded, encoded_for_audio, attention_mask.squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        if input_ids is None:
            raise ValueError("input_ids is required for Gemma text encoding.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model = self.gemma_model
        input_ids = input_ids.to(device=model.device)
        attention_mask = attention_mask.to(device=model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        encoded_inputs = self._run_feature_extractor(
            outputs.hidden_states,
            attention_mask,
            padding_side=self.padding_side,
        )
        if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") == "1":
            _debug_log_line(
                "fastvideo:gemma_feature"
                f":sum={encoded_inputs.float().sum().item():.6f} "
                f"shape={tuple(encoded_inputs.shape)}"
            )
        video_encoding, audio_encoding, attention_mask = self._run_connectors(
            encoded_inputs, attention_mask
        )
        if os.getenv("LTX2_PIPELINE_DEBUG_LOG", "0") == "1":
            _debug_log_line(
                "fastvideo:gemma_video_encoding"
                f":sum={video_encoding.float().sum().item():.6f} "
                f"shape={tuple(video_encoding.shape)}"
            )
            _debug_log_line(
                "fastvideo:gemma_audio_encoding"
                f":sum={audio_encoding.float().sum().item():.6f} "
                f"shape={tuple(audio_encoding.shape)}"
            )

        hidden_states = (audio_encoding, ) if output_hidden_states else None
        return BaseEncoderOutput(
            last_hidden_state=video_encoding,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name == "aggregate_embed.weight":
                name = "feature_extractor_linear.aggregate_embed.weight"
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    b, t, d, l = encoded_text.shape
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(
            f"padding_side must be 'left' or 'right', got {padding_side}"
        )

    mask = mask.reshape(b, t, 1, 1)
    eps = 1e-6

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(
        dim=(1, 2), keepdim=True
    )
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(
        dim=(1, 2), keepdim=True
    )
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flattened = mask.reshape(b, t, 1).expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)
    return normed
