# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Streamed Qwen3-VL prompt conditioner for MiniMax-H3 on Apple Silicon MLX.

Produces exactly the conditioning the H3 DiT consumes: the language-model
hidden states after the first 50 language-model layers plus the per-token
modality tags for text prompts.

Memory contract for the 36 GiB tier: the released conditioner is ~66 GB of
BF16 and never becomes resident. Tensors are read per-key from the safetensors
shards and only the pieces a given forward needs are materialized:

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
from typing import Any

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
    """Per-key access across the diffusers safetensors shards.

    Reads tensors directly through the safetensors header so BF16 weights
    stream from disk without loading a shard (and without torch).
    """

    def __init__(self, component_dir: Path):
        component_dir = Path(component_dir)
        index_path = component_dir / "model.safetensors.index.json"
        self.key_to_shard: dict[str, str] = {}
        self._header_cache: dict[str, tuple[dict, int]] = {}

        def needed(key: str) -> bool:
            if key == "model.language_model.embed_tokens.weight":
                return True
            prefix = "model.language_model.layers."
            if not key.startswith(prefix):
                return False
            layer = key[len(prefix):].split(".", 1)[0]
            return layer.isdigit() and int(layer) < TEXT_ENCODER_LAYER

        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            self.key_to_shard = {k: str(component_dir / s) for k, s in weight_map.items() if needed(k)}
        else:
            single = component_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"No conditioner weights under {component_dir}")
            import struct
            with open(single, "rb") as handle:
                (header_len, ) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len))
            self.key_to_shard = {k: str(single) for k in header if needed(k)}
        # Pre-cache only existing shard headers
        for shard_path in set(self.key_to_shard.values()):
            if Path(shard_path).exists():
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

    def _header(self, key: str) -> tuple[str, dict, int]:
        shard = self.key_to_shard[key]
        if not Path(shard).is_file():
            raise FileNotFoundError(f"Conditioner shard listed in the index is missing: {shard}")
        header, data_start = self._cache_header(shard)
        return shard, header, data_start

    def get(self, key: str) -> np.ndarray:
        shard, header, data_start = self._header(key)
        return _read_safetensors_bf16(shard, key, header, data_start)

    def get_mlx(self, key: str) -> mx.array:
        """Read one weight in FP32 without expanding BF16 on the CPU."""
        shard, header, data_start = self._header(key)
        if header[key]["dtype"] != "BF16":
            return mx.array(np.asarray(self.get(key), dtype=np.float32))
        raw = _read_bf16_words(shard, key, header, data_start)
        weight = mx.array(raw).view(mx.bfloat16).astype(mx.float32)
        # Finish each cast before constructing the layer graph, releasing its
        # BF16 input instead of retaining a second copy of every layer weight.
        mx.eval(weight)
        return weight

    def get_row(self, key: str, row: int) -> np.ndarray:
        shard, header, data_start = self._header(key)
        return _read_safetensors_row(shard, key, row, header, data_start)

    def close(self) -> None:
        gc.collect()


_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "U8": np.uint8}


def _read_bf16_words(path: str, key: str, header: dict, data_start: int) -> np.ndarray:
    """Bulk-read one BF16 tensor; avoid faulting in a large mmap during conversion."""
    meta = header[key]
    begin, end = meta["data_offsets"]
    if begin < 0 or end < begin or (end - begin) % 2:
        raise ValueError(f"Invalid BF16 data offsets {begin}:{end} in {path}:{key}")
    count = (end - begin) // 2
    with open(path, "rb") as handle:
        handle.seek(data_start + begin)
        raw = np.fromfile(handle, dtype=np.uint16, count=count)
    if raw.size != count:
        raise EOFError(f"Truncated safetensors tensor {path}:{key}: expected {count} BF16 values, got {raw.size}")
    return raw.reshape(meta["shape"])


def _read_safetensors_bf16(path: str, key: str, header: dict, data_start: int) -> np.ndarray:
    """Read one tensor (any dtype incl. BF16) without loading the shard."""
    meta = header[key]
    begin, end = meta["data_offsets"]
    count = end - begin
    dtype = meta["dtype"]
    if dtype == "BF16":
        raw = _read_bf16_words(path, key, header, data_start)
        expanded = raw.astype(np.uint32)
        expanded <<= 16  # Shift the owned conversion buffer, never the read-only mapping.
        return expanded.view(np.float32).reshape(meta["shape"])
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
        expanded = raw.astype(np.uint32)
        expanded <<= 16
        return expanded.view(np.float32).reshape(shape[1:])
    return np.array(
        np.memmap(path, dtype=_DTYPES[dtype], mode="r", offset=data_start + begin, shape=(row_count, )),
        copy=True,
    ).reshape(shape[1:])


def _rms_norm(x, weight, eps: float):
    return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


@dataclass(frozen=True)
class NVFP4Matrix:
    """MLX row-major E2M1/E4M3 weights and the export's inverse global scale."""

    weight: mx.array
    scales: mx.array
    global_scale: float

    def matmul(self, x):
        return mx.quantized_matmul(x, self.weight, self.scales, mode="nvfp4") / self.global_scale


def _linear(x, weight, bias=None):
    y = weight.matmul(x) if isinstance(weight, NVFP4Matrix) else x @ weight.T
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
        self.index: Any = _ShardIndex(self.component_dir)
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

        hidden = self._embed_tokens(token_ids)

        if cfg.num_layers < TEXT_ENCODER_LAYER:
            raise ValueError(f"Conditioner needs at least {TEXT_ENCODER_LAYER} layers, has {cfg.num_layers}.")
        for layer in range(TEXT_ENCODER_LAYER):
            hidden = self._decoder_layer(layer, hidden, cos, sin)
            # Per-layer sync keeps the 50-layer activation graph bounded.
            mx.eval(hidden)
            gc.collect()

        tags = np.full((seq_len, ), 1, dtype=np.int64)  # MINIMAX_H3_TEXT_TAG
        return np.asarray(hidden).astype(np.float32), tags

    def _embed_tokens(self, token_ids: list[int]):
        # Embedding rows gathered individually; the full table is never
        # materialized by the BF16 streaming path.
        rows = []
        for token in token_ids:
            key = "model.language_model.embed_tokens.weight"
            rows.append(self.index.get_row(key, token))
        hidden = mx.array(np.stack(rows).astype(np.float32))
        del rows
        gc.collect()
        return hidden

    # -- layers ----------------------------------------------------------

    def _decoder_layer(self, index: int, hidden, cos, sin):
        cfg = self.config
        prefix = f"model.language_model.layers.{index}."

        def w(name):
            return self.index.get_mlx(prefix + name)

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


def unswizzle_nvfp4_scales(scale: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """FlashInfer 128x4 scale bytes -> MLX row-major group-16 scale bytes."""
    pad_rows, pad_cols = -(-rows // 128) * 128, -(-cols // 4) * 4
    if scale.size != pad_rows * pad_cols:
        raise ValueError(f"NVFP4 scales need {pad_rows * pad_cols} bytes, got {scale.size}.")
    tiles = scale.reshape(pad_rows // 128, pad_cols // 4, 32, 4, 4)
    return np.ascontiguousarray(tiles.transpose(0, 3, 2, 1, 4).reshape(pad_rows, pad_cols)[:rows, :cols])


MLX_NVFP4_ENCODER_MANIFEST = "mlx_h3_nvfp4_encoder.json"


def _read_nvfp4_encoder_config(component_dir: str | Path) -> dict[str, Any]:
    raw = json.loads((Path(component_dir) / "config.json").read_text())
    expected = {
        "quant_method": "nvfp4",
        "fmt": "e2m1",
        "group_size": 16,
        "scale_fmt": "e4m3",
        "scale_layout": "128x4",
        "activation_scheme": "dynamic",
    }
    quant = raw.get("quantization_config", {})
    if any(quant.get(key) != value for key, value in expected.items()):
        raise ValueError("MLX NVFP4 conditioning requires the FastVideo group-16, 128x4 encoder export.")
    return raw


def export_mlx_h3_nvfp4_encoder(component_dir: str | Path, output_dir: str | Path) -> Path:
    """Cache the released packed encoder in MLX layout, without requantization.

    Keep original packed nibbles, row-major scale bytes, global scales and
    embedding/norm values. Later loads skip CPU scale unswizzling and staging.
    """
    raw = _read_nvfp4_encoder_config(component_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Encoder cache output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _ResidentNVFP4Index(_ShardIndex(Path(component_dir)))
    try:
        arrays = {}
        matrices = {}
        dense_keys = []
        for key, value in index.weights.items():
            if isinstance(value, NVFP4Matrix):
                arrays[key] = value.weight
                arrays[key + ".scales"] = value.scales
                matrices[key] = {"global_scale": value.global_scale}
            else:
                arrays[key] = value
                dense_keys.append(key)
        mx.save_safetensors(str(output_dir / "model.safetensors"), arrays)
        (output_dir / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
        manifest = {
            "format_version": 1,
            "language_layers": TEXT_ENCODER_LAYER,
            "matrices": matrices,
            "dense_keys": dense_keys,
            "source_dir": str(Path(component_dir).resolve())
        }
        (output_dir / MLX_NVFP4_ENCODER_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        index.close()
    return output_dir


class _ResidentNVFP4Index:
    """Load the released 50-layer encoder without expanding packed matrices."""

    def __init__(self, source: _ShardIndex):
        self.weights: dict[str, mx.array | NVFP4Matrix] = {}
        for key in sorted(source.key_to_shard):
            if key.endswith(".weight_packed"):
                prefix = key.removesuffix(".weight_packed")
                packed = np.array(source.get(key), copy=True)
                if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] % 4:
                    raise ValueError(f"Invalid packed NVFP4 matrix {key}: {packed.shape}, {packed.dtype}")
                rows, cols = packed.shape[0], packed.shape[1] * 2
                if cols % 16:
                    raise ValueError(f"NVFP4 input width must be divisible by 16: {key}")
                scales = unswizzle_nvfp4_scales(source.get(prefix + ".weight_scale"), rows, cols // 16)
                global_scale = float(source.get(prefix + ".weight_global_scale").reshape(-1)[0])
                if not np.isfinite(global_scale) or global_scale <= 0:
                    raise ValueError(f"Invalid NVFP4 global scale for {prefix}: {global_scale}")
                weight = mx.array(packed).view(mx.uint32)
                scale_bytes = mx.array(scales)
                if not bool(mx.all(mx.isfinite(mx.from_fp8(scale_bytes, dtype=mx.float32)))):
                    raise ValueError(f"Non-finite NVFP4 block scales for {prefix}")
                mx.eval(weight, scale_bytes)
                self.weights[prefix + ".weight"] = NVFP4Matrix(weight, scale_bytes, global_scale)
            elif key.endswith(".weight"):
                if key == "model.language_model.embed_tokens.weight":
                    shard = source.key_to_shard[key]
                    header, data_start = source._cache_header(shard)
                    if header[key]["dtype"] == "BF16":
                        value = mx.array(_read_bf16_words(shard, key, header, data_start)).view(mx.bfloat16)
                    else:
                        value = mx.array(source.get(key))
                else:
                    value = source.get_mlx(key)
                mx.eval(value)
                self.weights[key] = value
        source.close()

    @classmethod
    def from_mlx_checkpoint(cls, component_dir: str | Path):
        component_dir = Path(component_dir)
        manifest = json.loads((component_dir / MLX_NVFP4_ENCODER_MANIFEST).read_text())
        if manifest.get("format_version") != 1 or manifest.get("language_layers") != TEXT_ENCODER_LAYER:
            raise ValueError("Unsupported native MLX NVFP4 encoder cache")
        arrays = mx.load(str(component_dir / "model.safetensors"))
        matrices = manifest["matrices"]
        expected = set(manifest["dense_keys"]) | set(matrices) | {key + ".scales" for key in matrices}
        if set(arrays) != expected:
            raise ValueError("Native MLX encoder arrays do not match the manifest")
        index = cls.__new__(cls)
        index.weights = {key: arrays[key] for key in manifest["dense_keys"]}
        for key, info in matrices.items():
            weight, scales = arrays[key], arrays[key + ".scales"]
            factor = float(info["global_scale"])
            if (weight.ndim != 2 or weight.dtype != mx.uint32 or scales.dtype != mx.uint8
                    or scales.shape != (weight.shape[0], weight.shape[1] // 2) or weight.shape[1] % 2
                    or not np.isfinite(factor) or factor <= 0):
                raise ValueError(f"Invalid native MLX NVFP4 matrix: {key}")
            index.weights[key] = NVFP4Matrix(weight, scales, factor)
        mx.eval(list(arrays.values()))
        return index

    def get_mlx(self, key: str):
        return self.weights[key]

    def close(self) -> None:
        self.weights.clear()
        gc.collect()


class ResidentNVFP4MiniMaxH3TextConditioner(StreamedMiniMaxH3TextConditioner):
    """Released NVFP4 encoder weights with floating-point MLX activations.

    The packed weights and embedding table stay resident. CUDA quantizes
    activations to FP4; this path keeps FP32 activations, so hidden states are
    not expected to be bit-exact with the CUDA encoder.
    """

    def __init__(self, component_dir: str | Path, tokenizer_dir: str | Path | None = None):
        _read_nvfp4_encoder_config(component_dir)
        # Fail on an older MLX before reading the encoder's large shards.
        try:
            packed, scales = mx.quantize(mx.ones((1, 64)), mode="nvfp4")
            mx.eval(mx.quantized_matmul(mx.ones((1, 64)), packed, scales, mode="nvfp4"))
        except (ValueError, RuntimeError) as error:
            raise RuntimeError("Native NVFP4 conditioning requires an MLX build with nvfp4 matmul support.") from error
        super().__init__(component_dir, tokenizer_dir)
        if (Path(component_dir) / MLX_NVFP4_ENCODER_MANIFEST).exists():
            self.index.close()
            self.index = _ResidentNVFP4Index.from_mlx_checkpoint(component_dir)
        else:
            self.index = _ResidentNVFP4Index(self.index)

    def _embed_tokens(self, token_ids: list[int]):
        table = self.index.get_mlx("model.language_model.embed_tokens.weight")
        return table[mx.array(token_ids, dtype=mx.int32)].astype(mx.float32)


def _apply_mrope(q_or_k, cos, sin):
    """q_or_k: (S, H, D); cos/sin: (S, 1, D)."""
    half = q_or_k.shape[-1] // 2
    first, second = q_or_k[..., :half], q_or_k[..., half:]
    rotated = mx.concatenate([-second, first], axis=-1)
    return q_or_k * cos + rotated * sin
