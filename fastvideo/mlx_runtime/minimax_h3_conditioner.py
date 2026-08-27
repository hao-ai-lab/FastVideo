# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Streamed Qwen3-VL prompt conditioner for MiniMax-H3 on Apple Silicon MLX.

Produces exactly the conditioning the H3 DiT consumes: the language-model
hidden states after the first 50 language-model layers plus the per-token
modality tags for text prompts.

Memory contract for the 36 GiB tier: the released conditioner is ~66 GB of
BF16 and never becomes resident. Tensors are memory-mapped per-key from the
safetensors shards and only the pieces a given forward needs are materialized:

- token embedding table row-gathered per batch (full table never copied);
- one decoder layer (~1 GB BF16) resident at a time, computed in FP32,
  released before the next layer loads;
The forward pass uses MLX. Tokenization uses the Transformers tokenizer API.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mlx.core as mx

from fastvideo.logger import init_logger

logger = init_logger(__name__)

TEXT_ENCODER_LAYER = 50


@dataclass
class ConditionerConfig:
    hidden_size: int = 5120
    num_layers: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 25600
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)
    vocab_size: int = 151936

    @classmethod
    def from_config_json(cls, path: str | Path) -> ConditionerConfig:
        raw = json.loads(Path(path).read_text())
        text = raw.get("text_config", raw)
        scaling = text.get("rope_scaling", {}) or {}
        return cls(
            hidden_size=int(text["hidden_size"]),
            num_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])),
            intermediate_size=int(text["intermediate_size"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta=float(text["rope_theta"]),
            mrope_section=tuple(int(v) for v in scaling.get("mrope_section", (24, 20, 20))),
            vocab_size=int(text.get("vocab_size", 151936)),
        )


class _ShardIndex:
    """Per-key memory-mapped access across the diffusers safetensors shards.

    Reads tensors directly through the safetensors header so BF16 weights
    stream from disk without loading a shard (and without torch).
    """

    def __init__(self, component_dir: Path):
        component_dir = Path(component_dir)
        index_path = component_dir / "model.safetensors.index.json"
        self.key_to_shard: dict[str, str] = {}
        self._header_cache: dict[str, tuple[dict, int]] = {}
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            self.key_to_shard = {k: str(component_dir / s) for k, s in weight_map.items()}
        else:
            single = component_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"No conditioner weights under {component_dir}")
            import struct
            with open(single, "rb") as handle:
                (header_len, ) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len))
            self.key_to_shard = {k: str(single) for k in header if k != "__metadata__"}
        # Pre-cache all shard headers
        for shard_path in set(self.key_to_shard.values()):
            self._cache_header(shard_path)

    def _cache_header(self, path: str) -> tuple[dict, int]:
        """Parse and cache a shard's header, returning (header_dict, data_start_offset)."""
        if path not in self._header_cache:
            import struct
            with open(path, "rb") as handle:
                (header_len, ) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            self._header_cache[path] = (header, data_start)
        return self._header_cache[path]

    def get(self, key: str) -> np.ndarray:
        shard = self.key_to_shard[key]
        header, data_start = self._header_cache[shard]
        return _read_safetensors_bf16(shard, key, header, data_start)

    def get_row(self, key: str, row: int) -> np.ndarray:
        shard = self.key_to_shard[key]
        header, data_start = self._header_cache[shard]
        return _read_safetensors_row(shard, key, row, header, data_start)

    def close(self) -> None:
        gc.collect()


_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def _read_safetensors_bf16(path: str, key: str, header: dict, data_start: int) -> np.ndarray:
    """Read one tensor (any dtype incl. BF16) without loading the shard."""
    meta = header[key]
    begin, end = meta["data_offsets"]
    count = end - begin
    dtype = meta["dtype"]
    if dtype == "BF16":
        raw = np.memmap(path, dtype=np.uint16, mode="r", offset=data_start + begin, shape=(count // 2, ))
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported safetensors dtype {dtype} in {path}:{key}")
    return np.asarray(
        np.memmap(path,
                  dtype=_DTYPES[dtype],
                  mode="r",
                  offset=data_start + begin,
                  shape=(count // np.dtype(_DTYPES[dtype]).itemsize, )).reshape(meta["shape"]))


def _read_safetensors_row(path: str, key: str, row: int, header: dict, data_start: int) -> np.ndarray:
    """Read one leading-dimension row without materializing the full tensor."""
    meta = header[key]
    shape = tuple(int(value) for value in meta["shape"])
    if len(shape) < 2:
        raise ValueError(f"Row access requires a rank >= 2 tensor, got {shape} for {key}")
    if not 0 <= row < shape[0]:
        raise IndexError(f"Row {row} is outside leading dimension {shape[0]} for {key}")

    dtype = meta["dtype"]
    if dtype != "BF16" and dtype not in _DTYPES:
        raise ValueError(f"Unsupported safetensors dtype {dtype} in {path}:{key}")
    item_size = 2 if dtype in {"BF16", "F16"} else np.dtype(_DTYPES[dtype]).itemsize
    row_count = int(np.prod(shape[1:], dtype=np.int64))
    begin = int(meta["data_offsets"][0]) + row * row_count * item_size
    if dtype == "BF16":
        raw = np.memmap(path, dtype=np.uint16, mode="r", offset=data_start + begin, shape=(row_count, ))
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape[1:])
    return np.array(
        np.memmap(path, dtype=_DTYPES[dtype], mode="r", offset=data_start + begin, shape=(row_count, )),
        copy=True,
    ).reshape(shape[1:])


def _rms_norm(x, weight, eps: float):
    return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _linear(x, weight, bias=None):
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def _mrope_cos_sin(positions: np.ndarray, cfg: ConditionerConfig) -> tuple[mx.array, mx.array]:
    """(3, S) axis positions -> cos/sin each (S, 1, head_dim), fp32.

    Interleaved MRoPE assembled in NumPy exactly like the reference (strided
    slice overwrite on top of the temporal-axis frequencies); MLX lacks
    strided slice assignment.
    """
    half = cfg.head_dim // 2
    inv_freq = (1.0 / (cfg.rope_theta**(np.arange(0, half, dtype=np.float32) / half))).astype(np.float32)
    frequencies = positions.astype(np.float32)[:, :, None] * inv_freq[None, None, :]  # (3, S, half)
    sections = cfg.mrope_section
    interleaved = frequencies[0].copy()
    stop = sections[1] * 3
    interleaved[:, 1:stop:3] = frequencies[1][:, 1:stop:3]
    stop = sections[2] * 3
    interleaved[:, 2:stop:3] = frequencies[2][:, 2:stop:3]
    embedding = np.concatenate([interleaved, interleaved], axis=-1)[:, None, :]
    return mx.array(np.cos(embedding)), mx.array(np.sin(embedding))


class StreamedMiniMaxH3TextConditioner:
    """Layer-streaming Qwen3-VL text stack -> H3 conditioning hidden states."""

    def __init__(self, component_dir: str | Path, tokenizer_dir: str | Path | None = None):
        self.component_dir = Path(component_dir)
        self.config = ConditionerConfig.from_config_json(self.component_dir / "config.json")
        self.index = _ShardIndex(self.component_dir)
        self.tokenizer = self._load_tokenizer(tokenizer_dir)

    def _load_tokenizer(self, tokenizer_dir: str | Path | None):
        from transformers import AutoTokenizer

        candidates = [Path(tokenizer_dir)] if tokenizer_dir else [
            self.component_dir.parent / "tokenizer",
            self.component_dir,
        ]
        last_error = None
        for candidate in candidates:
            try:
                return AutoTokenizer.from_pretrained(str(candidate))
            except Exception as error:  # noqa: BLE001 - fall through to next candidate
                last_error = error
        raise RuntimeError(f"Could not load an H3 tokenizer from {candidates}: {last_error}")

    # -- public API ------------------------------------------------------

    def tokenize(self, prompt: str) -> list[int]:
        encoded = self.tokenizer(prompt, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise ValueError("H3 conditioning expects exactly one sequence.")
            input_ids = input_ids[0]
        return [int(t) for t in input_ids]

    def encode_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """prompt -> (hidden states (S, hidden), token tags (S,)) both fp32."""
        token_ids = self.tokenize(prompt)
        return self.encode_tokens(token_ids)

    def encode_tokens(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        import mlx.core as mx

        cfg = self.config
        seq_len = len(token_ids)
        positions = np.stack([
            np.arange(seq_len, dtype=np.float64),
            np.arange(seq_len, dtype=np.float64),
            np.arange(seq_len, dtype=np.float64),
        ])
        cos, sin = _mrope_cos_sin(positions, cfg)

        # Embedding rows gathered individually; the (151936, 5120) table is
        # never fully materialized.
        rows = []
        for token in token_ids:
            key = "model.language_model.embed_tokens.weight"
            rows.append(self.index.get_row(key, token))
        hidden = mx.array(np.stack(rows).astype(np.float32))
        del rows
        gc.collect()

        if cfg.num_layers <= TEXT_ENCODER_LAYER:
            raise ValueError(f"Conditioner needs > {TEXT_ENCODER_LAYER} layers, has {cfg.num_layers}.")
        for layer in range(TEXT_ENCODER_LAYER):
            hidden = self._decoder_layer(layer, hidden, cos, sin)
            # Per-layer sync: without this the whole 50-layer graph accumulates
            # and the machine runs out of memory (same failure mode as the DiT).
            mx.eval(hidden)
            gc.collect()

        tags = np.full((seq_len, ), 1, dtype=np.int64)  # MINIMAX_H3_TEXT_TAG
        return np.asarray(hidden).astype(np.float32), tags

    # -- layers ----------------------------------------------------------

    def _decoder_layer(self, index: int, hidden, cos, sin):
        cfg = self.config
        prefix = f"model.language_model.layers.{index}."

        def w(name):
            return mx.array(np.asarray(self.index.get(prefix + name)).astype(np.float32))

        # Self-attention block.
        residual = hidden
        normed = _rms_norm(hidden, w("input_layernorm.weight"), cfg.rms_norm_eps)
        query = _linear(normed, w("self_attn.q_proj.weight"))
        key = _linear(normed, w("self_attn.k_proj.weight"))
        value = _linear(normed, w("self_attn.v_proj.weight"))

        heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        seq_len = hidden.shape[0]

        def split_heads(t, count):
            return t.reshape(seq_len, count, head_dim)

        query = split_heads(query, heads)
        key = split_heads(key, kv_heads)
        value = split_heads(value, kv_heads)
        # Per-head QK RMSNorm (Qwen3), weights only.
        query = _rms_norm(query, w("self_attn.q_norm.weight"), cfg.rms_norm_eps)
        key = _rms_norm(key, w("self_attn.k_norm.weight"), cfg.rms_norm_eps)

        # Interleaved MRoPE.
        query = _apply_mrope(query, cos, sin)
        key = _apply_mrope(key, cos, sin)

        # GQA: repeat KV heads.
        repeats = heads // kv_heads
        key = mx.repeat(key, repeats, axis=1)
        value = mx.repeat(value, repeats, axis=1)

        scores = (query.transpose(1, 0, 2) @ key.transpose(1, 2, 0)) * head_dim**-0.5
        mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
        scores = mx.where(mask[None], mx.array(-np.inf, dtype=scores.dtype), scores)
        attended = mx.softmax(scores, axis=-1) @ value.transpose(1, 0, 2)
        # Attention is head-major here: (heads, sequence, head_dim).  Restore
        # sequence-major ordering before concatenating heads for o_proj.
        attended = attended.transpose(1, 0, 2)
        attn_out = _linear(attended.reshape(seq_len, -1), w("self_attn.o_proj.weight"))
        hidden = residual + attn_out

        # MLP block (SwiGLU).
        residual = hidden
        normed = _rms_norm(hidden, w("post_attention_layernorm.weight"), cfg.rms_norm_eps)
        gate = _linear(normed, w("mlp.gate_proj.weight"))
        up = _linear(normed, w("mlp.up_proj.weight"))
        down = _linear(gate * mx.sigmoid(gate) * up, w("mlp.down_proj.weight"))
        return residual + down

    def close(self) -> None:
        self.index.close()


def _apply_mrope(q_or_k, cos, sin):
    """q_or_k: (S, H, D); cos/sin: (S, 1, D)."""
    half = q_or_k.shape[-1] // 2
    first, second = q_or_k[..., :half], q_or_k[..., half:]
    rotated = mx.concatenate([-second, first], axis=-1)
    return q_or_k * cos + rotated * sin
