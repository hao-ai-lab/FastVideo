# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Qwen-Image-2.1 single-stream diffusion transformer.

The interleaved text/image block-causal stream has no matching FastVideo
sequence-parallel attention primitive. This implementation runs segmented
PyTorch SDPA on one device by default; tensor and sequence parallelism are not
implemented. Explicit FlashAttention or SageAttention requests replace only
mask-free dense calls, retaining SDPA for causal text and padding masks.
Prefix K/V can reside on the host, so decoding restores only one layer's cache
at a time instead of keeping every layer's reference images on the GPU.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.qwen_image21 import QwenImage21Config
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT

if TYPE_CHECKING:
    from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)

_IMG_TOKENS_PER_SLOT = 4


class QwenImage21KVCacheAllocationError(RuntimeError):
    """CPU prefix-cache allocation failed; an inference caller may retry uncached."""


class QwenImage21KVLayerCache:
    """Post-RoPE prefix K/V in ``(batch, tokens, heads, head_dim)`` format."""

    def __init__(self, storage_device: str | torch.device | None = None):
        self.storage_device = torch.device(storage_device) if storage_device is not None else None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape or k.ndim != 4:
            raise ValueError("Prefix K/V must have matching (batch, tokens, heads, head_dim) shapes")
        # A fresh allocation releases the larger prefill tensor after the layer finishes.
        device = self.storage_device or k.device
        try:
            self.k = k.detach().to(device=device, copy=True)
            self.v = v.detach().to(device=device, copy=True)
        except (MemoryError, RuntimeError) as exc:
            self.clear()
            message = str(exc).lower()
            allocation_error = isinstance(exc, MemoryError) or any(
                marker in message for marker in (
                    "defaultcpuallocator", "cannot allocate memory", "can't allocate memory",
                    "not enough memory", "out of memory", "std::bad_alloc"
                ))
            if device.type == "cpu" and allocation_error and "cuda" not in message and "hip" not in message:
                raise QwenImage21KVCacheAllocationError(
                    "CPU memory exhausted while allocating the prefix KV cache") from exc
            raise

    def get(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if self.k is None or self.v is None:
            raise RuntimeError("Prefix KV cache has not been populated; run an extract step first")
        return self.k.to(device=device, dtype=dtype), self.v.to(device=device, dtype=dtype)

    def clear(self) -> None:
        self.k = self.v = None


class QwenImage21KVCache:
    """Per-request cache container; never share it between prompts or CFG branches."""

    def __init__(self, num_layers: int, storage_device: str | torch.device | None = None):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layer_caches = [QwenImage21KVLayerCache(storage_device) for _ in range(num_layers)]
        self.prefix_len: int | None = None

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layer_caches[layer_idx]

    def clear(self) -> None:
        for layer in self.layer_caches:
            layer.clear()
        self.prefix_len = None


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(rotated * freqs_cis.unsqueeze(1)).flatten(3).to(x.dtype)


class QwenImage21TemporalTimesteps(nn.Module):

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device="cpu") / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        args = (self.time_factor * timestep.float())[:, None] * self.freqs.float()[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear_1 = ReplicatedLinear(256, embedding_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(embedding_dim, embedding_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(hidden_states)
        return self.linear_2(self.act(hidden_states))[0]


class QwenImage21TimestepProjEmbeddings(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps()
        self.timestep_embedder = QwenImage21TimestepEmbedding(embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(timestep).to(hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class QwenImage21TextProjection(nn.Module):

    def __init__(self, context_in_dim: int, hidden_size: int, eps: float):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps)
        self.in_layer = ReplicatedLinear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = ReplicatedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.in_layer(self.text_norm(hidden_states))
        return self.out_layer(self.act(hidden_states))[0]


class QwenImage21SwiGLUFeedForward(nn.Module):

    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = ReplicatedLinear(hidden_size, mlp_hidden_size, bias=False)
        self.out = ReplicatedLinear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = ReplicatedLinear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.activation_fn(self.gate_layer(hidden_states)[0])
        return self.out(gate * self.proj(hidden_states)[0])[0]


def _select_modulation_rows(params: torch.Tensor, target_token_mask: torch.Tensor | None) -> torch.Tensor:
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


class QwenImage21AdaLayerNormContinuous(nn.Module):

    def __init__(self, embedding_dim: int, eps: float):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states, conditioning_embedding, target_token_mask=None):
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))[0]
        return self.norm(hidden_states) * (1 + _select_modulation_rows(scale, target_token_mask))


def _qwenimage21_prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    prefix_ids = image_ids[:prefix_len].tolist()
    segments = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments


def _attention(query, key, value, mask=None):
    # SDPA expects (batch, heads, tokens, dim); projected QKV use tokens before heads.
    output = F.scaled_dot_product_attention(query.transpose(1, 2),
                                            key.transpose(1, 2),
                                            value.transpose(1, 2),
                                            attn_mask=mask,
                                            dropout_p=0.0)
    if mask is not None:
        # Older CPU SDPA returns NaNs for fully masked padding queries.
        output = output.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
    return output.transpose(1, 2)


def _make_dense_attention(heads: int, dim_head: int,
                          supported_attention_backends: tuple[AttentionBackendEnum, ...],
                          requested_backend: AttentionBackendEnum) -> nn.Module | None:
    # Keep optional attention kernels and GPU dispatch imports off the SDPA path.
    from fastvideo.attention.layer import LocalAttention
    from fastvideo.attention.selector import backend_name_to_enum, get_attn_backend
    from fastvideo.platforms import AttentionBackendEnum
    from fastvideo.utils import get_compute_dtype

    dtype = get_compute_dtype()
    backend_cls = get_attn_backend(dim_head, dtype,
                                   supported_attention_backends=supported_attention_backends,
                                   default_backend=AttentionBackendEnum.TORCH_SDPA,
                                   requested=requested_backend)
    backend = backend_name_to_enum(backend_cls.get_name())
    if backend is AttentionBackendEnum.TORCH_SDPA:
        return None

    class DenseLocalAttention(LocalAttention):
        """Qwen owns segment lengths and has already applied its complex RoPE.

        LocalAttention's constructor has not yet threaded explicit component
        requests to the selector. Resolve through the public selector above,
        bypassing ambient construction scopes. Its ordinary forward also reads
        unrelated global metadata, which must not reach our unequal Q/KV calls.
        """

        def __init__(self):
            nn.Module.__init__(self)
            self.softmax_scale = dim_head**-0.5
            self.attn_impl = backend_cls.get_impl_cls()(num_heads=heads, head_size=dim_head,
                                                        softmax_scale=self.softmax_scale,
                                                        num_kv_heads=heads, causal=False)
            self.num_heads = self.num_kv_heads = heads
            self.head_size, self.backend, self.dtype = dim_head, backend, dtype

        def forward(self, q, k, v):
            return self.attn_impl.forward(q, k, v, None)

    return DenseLocalAttention()


class QwenImage21Attention(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float, *,
                 requested_backend: AttentionBackendEnum | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_k = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_v = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([ReplicatedLinear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = QwenImage21RMSNorm(dim_head, eps)
        self.norm_k = QwenImage21RMSNorm(dim_head, eps)
        self.dense_attn = None
        if requested_backend is not None and requested_backend.name != "TORCH_SDPA":
            supported = supported_attention_backends or _DEFAULT_CONFIG.arch_config._supported_attention_backends
            self.dense_attn = _make_dense_attention(heads, dim_head, supported, requested_backend)

    def _attend(self, query, key, value, mask=None):
        if mask is not None or self.dense_attn is None:
            return _attention(query, key, value, mask)
        return self.dense_attn(query, key, value)

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_emb: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                layer_cache: QwenImage21KVLayerCache | None = None,
                kv_cache_mode: str | None = None,
                cache_write_slice: slice | None = None,
                segments: list[tuple[int, int, bool]] | None = None,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(hidden_states)[0].unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states)[0].unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states)[0].unflatten(-1, (self.heads, -1))
        query, key = self.norm_q(query).to(value.dtype), self.norm_k(key).to(value.dtype)
        if rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, rotary_emb)
            key = apply_rotary_emb_qwen(key, rotary_emb)
        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(key[:, cache_write_slice], value[:, cache_write_slice])
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get(key.device, key.dtype)
                if cached_k.shape[0] != key.shape[0] or cached_k.shape[2:] != key.shape[2:]:
                    raise ValueError("Cached prefix batch/head shape differs from the current request")
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)
                del cached_k, cached_v
        if segments is None:
            output = self._attend(query, key, value, attention_mask)
        else:
            outputs = []
            for start, end, is_text in segments:
                mask = None
                if is_text:
                    length = end - start
                    mask = torch.cat([torch.ones(length, start, dtype=torch.bool, device=query.device),
                                      torch.tril(torch.ones(length, length, dtype=torch.bool, device=query.device))],
                                     dim=1)[None, None]
                if key_valid is not None:
                    valid = key_valid[:, None, None, :end]
                    mask = valid if mask is None else mask & valid
                outputs.append(self._attend(query[:, start:end], key[:, :end], value[:, :end], mask))
            prefix_len = segments[-1][1] if segments else 0
            mask = None if key_valid is None else key_valid[:, None, None, :]
            outputs.append(self._attend(query[:, prefix_len:], key, value, mask))
            output = torch.cat(outputs, dim=1)
        output = output.flatten(2, 3).to(query.dtype)
        return self.to_out[1](self.to_out[0](output)[0])


class QwenImage21TransformerBlock(nn.Module):

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: int, eps: float, *,
                 requested_backend: AttentionBackendEnum | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim, num_attention_heads, attention_head_dim, eps,
                                       requested_backend=requested_backend,
                                       supported_attention_backends=supported_attention_backends)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(dim, dim * mlp_ratio)

    def _modulate(self, hidden_states, mod_params, target_token_mask):
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(self, hidden_states, modulation, target_token_mask=None, **attention_kwargs):
        mod1, mod2 = modulation.chunk(2, dim=-1)
        normalized, gate = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        hidden_states = hidden_states + gate.tanh() * self.attn(normalized, **attention_kwargs)
        normalized, gate = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + gate.tanh() * self.img_mlp(normalized)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class QwenImage21Rope(nn.Module):

    def __init__(self, theta: int, axes_dim: tuple[int, int, int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        # Frequency tensors stay fp32/complex64 even when model weights are cast.
        self.freqs = self._make_freqs(torch.device("cpu"))

    def _make_freqs(self, device):
        positive = torch.arange(8192, device=device)
        negative = torch.arange(1024, device=device).flip(0) * -1 - 1
        return [torch.cat([self.rope_params(positive, dim), self.rope_params(negative, dim)])
                for dim in self.axes_dim]

    def rope_params(self, index: torch.Tensor, dim: int) -> torch.Tensor:
        powers = torch.arange(0, dim, 2, dtype=torch.float32, device=index.device).div(dim)
        freqs = torch.outer(index.float(), 1.0 / torch.pow(self.theta, powers))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, img_shapes, image_pad_mask, device):
        self.freqs = [freq.to(device=device) for freq in self.freqs]
        frame_index, image_height_index, image_width_index = [], [], []
        cursor, position = 0, 0
        is_image = image_pad_mask.tolist()
        for _, height, width in img_shapes:
            block_start = is_image.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len
            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)
            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])
        if cursor < len(is_image):
            frame_index.extend(range(position, position + len(is_image) - cursor))
        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index, width_index = frame_index.clone(), frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)
        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)


_DEFAULT_CONFIG = QwenImage21Config()


class QwenImage21Transformer2DModel(BaseDiT):
    _fsdp_shard_conditions = _DEFAULT_CONFIG.arch_config._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_CONFIG.arch_config._compile_conditions
    _supported_attention_backends = _DEFAULT_CONFIG.arch_config._supported_attention_backends
    param_names_mapping = _DEFAULT_CONFIG.arch_config.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_CONFIG.arch_config.reverse_param_names_mapping

    def __init__(self, config: QwenImage21Config, hf_config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config, hf_config or {}, **kwargs)
        if getattr(config, "quant_config", None) is not None:
            raise NotImplementedError("Qwen-Image-2.1 transformer quantization has not been implemented")
        self.inner_dim = self.hidden_size = config.hidden_size
        self.out_channels = config.out_channels
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.pos_embed = QwenImage21Rope(10000, config.axes_dims_rope)
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = QwenImage21TextProjection(config.context_in_dim, self.inner_dim, config.eps)
        self.img_in = ReplicatedLinear(config.in_channels * config.patch_size**2, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), ReplicatedLinear(self.inner_dim, 4 * self.inner_dim, bias=False))
        requested_backend = getattr(config, "_resolved_attention_backend", None)
        self.transformer_blocks = nn.ModuleList([
            QwenImage21TransformerBlock(self.inner_dim, config.num_attention_heads, config.attention_head_dim,
                                        config.mlp_ratio, config.eps, requested_backend=requested_backend,
                                        supported_attention_backends=self._supported_attention_backends)
            for _ in range(config.num_layers)
        ])
        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, config.eps)
        self.proj_out = ReplicatedLinear(self.inner_dim, config.patch_size**2 * config.out_channels, bias=False)

    @staticmethod
    def build_token_metadata(image_pad_mask, img_shapes):
        positions = image_pad_mask.nonzero(as_tuple=True)[0]
        lengths = [math.prod(shape) for shape in img_shapes]
        if not lengths or any(length <= 0 for length in lengths) or sum(lengths) != positions.numel():
            raise ValueError("img_shapes token counts must match image_pad_mask")
        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        ids = torch.repeat_interleave(torch.arange(len(lengths), device=image_pad_mask.device),
                                     torch.tensor(lengths, device=image_pad_mask.device))
        image_ids[positions] = ids
        target_mask = torch.zeros_like(image_pad_mask)
        target_mask[positions[-lengths[-1]:]] = True
        return image_ids, target_mask

    def _validate_inputs(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask, text_mask):
        batch_size = hidden_states.shape[0]
        if hidden_states.ndim != 3 or encoder_hidden_states.ndim != 3:
            raise ValueError("Latents and text embeddings must have (batch, tokens, channels) shapes")
        if encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("Latent and text batch sizes must match")
        if len(img_shapes) != batch_size or not img_shapes or not img_shapes[0]:
            raise ValueError("img_shapes must provide a nonempty image layout for each sample")
        if any(shapes != img_shapes[0] for shapes in img_shapes):
            raise ValueError("All samples must share the same image layout")
        if any(frame != 1 or height <= 0 or width <= 0 for frame, height, width in img_shapes[0]):
            raise ValueError("Qwen-Image-2.1 expects positive single-frame image shapes")
        target_tokens = math.prod(img_shapes[0][-1])
        if target_tokens % _IMG_TOKENS_PER_SLOT:
            raise ValueError("The target latent token count must be divisible by four")
        if hidden_states.shape[1] != sum(math.prod(shape) for shape in img_shapes[0]):
            raise ValueError("Packed latents must include all references followed by the target image")
        expected_slots = encoder_hidden_states.shape[1] + target_tokens // _IMG_TOKENS_PER_SLOT
        if img_mask.shape != (batch_size, expected_slots):
            raise ValueError("img_mask must include the VLM sequence and appended target-image slots")
        if not torch.equal(img_mask.bool(), img_mask[0:1].bool().expand_as(img_mask)):
            raise ValueError("All samples must share the same image-slot layout")
        if not bool(img_mask[:, -target_tokens // _IMG_TOKENS_PER_SLOT:].all()):
            raise ValueError("Target-image slots must be appended at the end of img_mask")
        if int(img_mask[0].count_nonzero()) * _IMG_TOKENS_PER_SLOT != hidden_states.shape[1]:
            raise ValueError("Image-slot count must match the reference and target latent token count")
        if timestep.numel() != batch_size:
            raise ValueError("timestep must contain one value per sample")
        if text_mask is not None and text_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_hidden_states_mask must match the VLM embedding sequence")

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                img_shapes: list[list[tuple[int, int, int]]],
                img_mask: torch.Tensor,
                encoder_hidden_states_mask: torch.Tensor | None = None,
                attention_kwargs: dict[str, Any] | None = None,
                kv_cache: QwenImage21KVCache | None = None,
                kv_cache_mode: str | None = None,
                return_dict: bool | None = None,
                **kwargs) -> torch.Tensor | tuple[torch.Tensor]:
        """Return joint-sequence predictions; the pipeline selects the target tail.

        ``kv_cache_mode='extract'`` stores timestep-independent text/reference
        K/V. ``'cached'`` returns only the target image tokens. By default the
        return value is a tensor; ``return_dict=False`` selects a one-item tuple.
        """
        self._validate_inputs(hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                              encoder_hidden_states_mask)
        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError("Prefix KV caching requires causal_condition=True")
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError("kv_cache_mode must be extract or cached when a cache is provided")
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError("kv_cache_mode requires a cache container")
        if kv_cache is not None and len(kv_cache.layer_caches) != len(self.transformer_blocks):
            raise ValueError("KV cache layer count differs from the transformer")
        if attention_kwargs:
            raise ValueError("Custom attention processors and LoRA scaling are not implemented for Qwen-Image-2.1")
        batch_size = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)[0]
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        img_mask = img_mask.bool().to(hidden_states.device)
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_tokens = math.prod(img_shapes[0][-1])
        joint = torch.cat([encoder_hidden_states,
                           encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, self.inner_dim)], dim=1)
        joint = joint.repeat_interleave(repeats, dim=1)
        joint[:, image_pad_mask] = hidden_states
        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, hidden_states.device)
        image_ids, target_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])
        timestep = timestep.reshape(batch_size).to(device=hidden_states.device, dtype=hidden_states.dtype)
        if self.config.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)])
            modulation_mask = target_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation[1](self.modulation[0](temb))[0]
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(batch_size, image_pad_mask.numel(), dtype=torch.bool,
                                         device=hidden_states.device)
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0, :encoder_hidden_states.shape[1]]
            valid_text = encoder_hidden_states_mask.bool().to(hidden_states.device)[:, vlm_text_positions]
            joint_key_valid[:, text_positions] = valid_text
        prefix_len = image_pad_mask.numel() - target_tokens
        if kv_cache_mode == "cached":
            if kv_cache.prefix_len != prefix_len:
                raise ValueError("Cached prefix layout differs from the current request or has not been extracted")
            joint, rotary_emb = joint[:, prefix_len:], rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            segments, block_key_valid, cache_slice = None, None, None
        else:
            segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
            attention_mask, block_key_valid = None, joint_key_valid
            cache_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            if kv_cache is not None:
                kv_cache.prefix_len = prefix_len
        for index, block in enumerate(self.transformer_blocks):
            joint = block(joint, modulation,
                          target_token_mask=modulation_mask,
                          rotary_emb=rotary_emb,
                          attention_mask=attention_mask,
                          layer_cache=kv_cache.get_layer(index) if kv_cache is not None else None,
                          kv_cache_mode=kv_cache_mode,
                          cache_write_slice=cache_slice,
                          segments=segments,
                          key_valid=block_key_valid)
        output = self.proj_out(self.norm_out(joint, temb, modulation_mask))[0]
        return (output,) if return_dict is False else output


EntryClass = QwenImage21Transformer2DModel
