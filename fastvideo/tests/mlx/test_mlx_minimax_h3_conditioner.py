# SPDX-License-Identifier: Apache-2.0
"""Focused correctness gates for the MLX-only MiniMax-H3 conditioner path."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for MiniMax-H3 conditioner tests")
torch = pytest.importorskip("torch", reason="PyTorch supplies an independent attention reference")

from fastvideo.mlx_runtime.minimax_h3_conditioner import (  # noqa: E402
    ConditionerConfig,
    StreamedMiniMaxH3TextConditioner,
    _ShardIndex,
)
from fastvideo.mlx_runtime.minimax_h3_pipeline import (  # noqa: E402
    MINIMAX_H3_PROMPT_CACHE_VERSION,
    MiniMaxH3MLXPipeline,
    _audio_sample_count,
    prompt_cache_path,
)
from fastvideo.mlx_runtime.minimax_h3_video_vae import MLXMiniMaxH3VideoVAE  # noqa: E402


class _ArrayIndex:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights

    def get(self, key: str) -> np.ndarray:
        return self.weights[key]

    def get_mlx(self, key: str):
        return mx.array(np.asarray(self.get(key), dtype=np.float32))


def test_bf16_embedding_row_read_does_not_materialize_full_table(tmp_path, monkeypatch) -> None:
    from safetensors.torch import save_file
    import fastvideo.mlx_runtime.minimax_h3_conditioner as conditioner_module

    key = "model.language_model.embed_tokens.weight"
    table = torch.arange(60, dtype=torch.float32).reshape(10, 6).to(torch.bfloat16)
    save_file({key: table}, tmp_path / "model.safetensors")
    index = _ShardIndex(tmp_path)

    def reject_full_read(*_args, **_kwargs):
        raise AssertionError("embedding lookup materialized the full table")

    monkeypatch.setattr(conditioner_module, "_read_safetensors_bf16", reject_full_read)
    row = index.get_row(key, 7)
    np.testing.assert_array_equal(row, table[7].float().numpy())
    assert row.shape == (6, )
    assert row.nbytes == 6 * np.dtype(np.float32).itemsize


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


def test_decoder_layer_matches_independent_torch_attention_layout() -> None:
    """Catch head-major attention output being flattened as sequence-major."""
    cfg = ConditionerConfig(
        hidden_size=8,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        rms_norm_eps=1e-6,
        mrope_section=(1, 1, 0),
        vocab_size=16,
    )
    rng = np.random.default_rng(4112)
    prefix = "model.language_model.layers.0."

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) * 0.15).astype(np.float32)

    weights = {
        prefix + "input_layernorm.weight": rng.uniform(0.8, 1.2, 8).astype(np.float32),
        prefix + "self_attn.q_proj.weight": matrix(8, 8),
        prefix + "self_attn.k_proj.weight": matrix(4, 8),
        prefix + "self_attn.v_proj.weight": matrix(4, 8),
        prefix + "self_attn.q_norm.weight": rng.uniform(0.8, 1.2, 4).astype(np.float32),
        prefix + "self_attn.k_norm.weight": rng.uniform(0.8, 1.2, 4).astype(np.float32),
        prefix + "self_attn.o_proj.weight": matrix(8, 8),
        prefix + "post_attention_layernorm.weight": rng.uniform(0.8, 1.2, 8).astype(np.float32),
        prefix + "mlp.gate_proj.weight": matrix(12, 8),
        prefix + "mlp.up_proj.weight": matrix(12, 8),
        prefix + "mlp.down_proj.weight": matrix(8, 12),
    }
    hidden_np = rng.standard_normal((3, 8)).astype(np.float32)
    cos = mx.ones((3, 1, 4), dtype=mx.float32)
    sin = mx.zeros((3, 1, 4), dtype=mx.float32)

    conditioner = StreamedMiniMaxH3TextConditioner.__new__(StreamedMiniMaxH3TextConditioner)
    conditioner.config = cfg
    conditioner.index = _ArrayIndex(weights)
    actual = np.asarray(conditioner._decoder_layer(0, mx.array(hidden_np), cos, sin))

    w = {name[len(prefix):]: torch.from_numpy(value) for name, value in weights.items()}
    hidden = torch.from_numpy(hidden_np)
    residual = hidden
    normed = _rms_norm(hidden, w["input_layernorm.weight"], cfg.rms_norm_eps)
    query = torch.nn.functional.linear(normed, w["self_attn.q_proj.weight"]).reshape(3, 2, 4)
    key = torch.nn.functional.linear(normed, w["self_attn.k_proj.weight"]).reshape(3, 1, 4)
    value = torch.nn.functional.linear(normed, w["self_attn.v_proj.weight"]).reshape(3, 1, 4)
    query = _rms_norm(query, w["self_attn.q_norm.weight"], cfg.rms_norm_eps)
    key = _rms_norm(key, w["self_attn.k_norm.weight"], cfg.rms_norm_eps)
    key = key.repeat_interleave(2, dim=1)
    value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query.transpose(0, 1), key.permute(1, 2, 0)) * cfg.head_dim**-0.5
    scores = scores.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1), -torch.inf)
    attended = torch.matmul(scores.softmax(dim=-1), value.transpose(0, 1))
    attended = attended.transpose(0, 1).reshape(3, 8)
    hidden = residual + torch.nn.functional.linear(attended, w["self_attn.o_proj.weight"])
    residual = hidden
    normed = _rms_norm(hidden, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    gate = torch.nn.functional.linear(normed, w["mlp.gate_proj.weight"])
    up = torch.nn.functional.linear(normed, w["mlp.up_proj.weight"])
    expected = residual + torch.nn.functional.linear(torch.nn.functional.silu(gate) * up,
                                                     w["mlp.down_proj.weight"])

    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_prompt_cache_is_versioned_for_conditioner_layout() -> None:
    current = prompt_cache_path("/tmp/cache", "/tmp/model", "two people")
    legacy_digest = __import__("hashlib").sha256(b"/tmp/model::two people").hexdigest()[:24]
    assert MINIMAX_H3_PROMPT_CACHE_VERSION == "v2-attention-layout"
    assert current.name != f"prompt_embeds_{legacy_digest}.npz"


def _cache_pipeline(tmp_path, quant_method=None):
    encoder = tmp_path / "model" / "text_encoder"
    tokenizer = tmp_path / "model" / "tokenizer"
    encoder.mkdir(parents=True, exist_ok=True)
    tokenizer.mkdir(parents=True, exist_ok=True)
    config = {"quantization_config": {"quant_method": quant_method}} if quant_method else {}
    (encoder / "config.json").write_text(json.dumps(config))
    (encoder / "model.safetensors").write_bytes(b"weights")
    (tokenizer / "tokenizer.json").write_text("{}")
    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    pipeline.prompt_cache_dir = tmp_path / "cache"
    pipeline.model_root = tmp_path / "model"
    pipeline.conditioner_dir = encoder
    pipeline.tokenizer_dir = tokenizer
    return pipeline


def test_prompt_cache_key_tracks_effective_encoder_and_files(tmp_path) -> None:
    pipeline = _cache_pipeline(tmp_path, quant_method="nvfp4")
    auto_key = pipeline._prompt_cache_key("prompt")
    pipeline.conditioner_mode = "nvfp4"
    assert pipeline._prompt_cache_key("prompt") == auto_key
    (pipeline.conditioner_dir / "model.safetensors").write_bytes(b"replaced weights")
    assert pipeline._prompt_cache_key("prompt") != auto_key
    assert pipeline._prompt_cache_key("other prompt") != pipeline._prompt_cache_key("prompt")


def test_prompt_cache_write_is_atomic(tmp_path, monkeypatch) -> None:
    class FakeConditioner:
        def encode_prompt(self, _prompt):
            return np.ones((2, 3), dtype=np.float32), np.ones((2, ), dtype=np.int64)

        def close(self):
            return None

    pipeline = _cache_pipeline(tmp_path)
    monkeypatch.setattr(MiniMaxH3MLXPipeline, "_load_conditioner", lambda _self: FakeConditioner())

    def interrupted_save(handle, **_arrays):
        handle.write(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(np, "savez", interrupted_save)
    cache_path = pipeline._prompt_cache_key("prompt")
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.encode_prompt("prompt")

    assert not cache_path.exists()
    assert not cache_path.with_name(f".{cache_path.name}.tmp").exists()


def test_video_vae_decode_defaults_to_reference_spatial_tiles() -> None:
    parameters = inspect.signature(MLXMiniMaxH3VideoVAE.decode).parameters
    assert parameters["tiled"].default is True
    assert parameters["tile_sample_min_height"].default == 256
    assert parameters["tile_sample_min_width"].default == 256

    vae = MLXMiniMaxH3VideoVAE.__new__(MLXMiniMaxH3VideoVAE)
    vae.spatial_compression_ratio = 16
    assert len(vae._split_tiles(480, 256, 64)[0]) > 1
    assert len(vae._split_tiles(832, 256, 64)[0]) > 1


def test_h3_audio_duration_rounds_up_to_cover_last_video_packet() -> None:
    assert _audio_sample_count(124) == 165334


@pytest.mark.parametrize("fmt,scale_fmt,activation", [("e2m1", "e4m3", "dynamic"),
                                                       ("nvfp4", "float8_e4m3fn", "Dynamic"),
                                                       ("FLOAT4_E2M1FN", "E4M3", "dynamic")])
def test_nvfp4_encoder_config_accepts_cuda_loader_spellings(tmp_path, fmt, scale_fmt, activation) -> None:
    from fastvideo.mlx_runtime.minimax_h3_conditioner import _read_nvfp4_encoder_config

    quant = {"quant_method": "NVFP4", "fmt": fmt, "group_size": 16, "scale_fmt": scale_fmt,
             "scale_layout": "128x4", "activation_scheme": activation}
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant}))
    assert _read_nvfp4_encoder_config(tmp_path)["quantization_config"] == quant
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {**quant, "group_size": 32}}))
    with pytest.raises(ValueError, match="group-16"):
        _read_nvfp4_encoder_config(tmp_path)


def test_unswizzle_nvfp4_scales_ignores_trailing_padding() -> None:
    from fastvideo.mlx_runtime.minimax_h3_conditioner import unswizzle_nvfp4_scales

    exact = np.arange(128 * 4, dtype=np.uint8)
    padded = np.concatenate([exact, np.full(64, 255, dtype=np.uint8)])
    np.testing.assert_array_equal(unswizzle_nvfp4_scales(padded, 100, 3), unswizzle_nvfp4_scales(exact, 100, 3))
    with pytest.raises(ValueError, match="need 512 bytes"):
        unswizzle_nvfp4_scales(exact[:-1], 100, 3)


def test_encoder_cache_is_published_with_umask_permissions(tmp_path, monkeypatch) -> None:
    import os

    from fastvideo.mlx_runtime import minimax_h3_conditioner as conditioner

    class FakeIndex:
        def __init__(self, _source):
            self.weights = {"model.norm.weight": mx.ones((4, ))}

        def close(self):
            pass

    monkeypatch.setattr(conditioner, "_read_nvfp4_encoder_config", lambda _dir: {})
    monkeypatch.setattr(conditioner, "_ShardIndex", lambda path: path)
    monkeypatch.setattr(conditioner, "_ResidentNVFP4Index", FakeIndex)
    old_umask = os.umask(0o022)
    try:
        out = conditioner.export_mlx_h3_nvfp4_encoder(tmp_path / "src", tmp_path / "cache")
    finally:
        os.umask(old_umask)
    assert out.stat().st_mode & 0o777 == 0o755
