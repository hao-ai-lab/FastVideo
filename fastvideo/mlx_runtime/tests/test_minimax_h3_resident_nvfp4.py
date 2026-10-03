# SPDX-License-Identifier: Apache-2.0
"""Native MLX NVFP4 encoder storage and residency regression checks."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from fastvideo.mlx_runtime.minimax_h3_conditioner import (
    NVFP4Matrix, ResidentNVFP4MiniMaxH3TextConditioner, _ResidentNVFP4Index,
    _ShardIndex, unswizzle_nvfp4_scales,
)
from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline


def _swizzle(values):
    # Independent coordinate mapping of FlashInfer's 128-row, four-column tiles.
    rows, cols = values.shape
    padded = np.zeros((-(-rows // 128) * 128, -(-cols // 4) * 4), np.uint8)
    padded[:rows, :cols] = values
    output = np.empty(padded.size, np.uint8)
    for r in range(padded.shape[0]):
        for c in range(padded.shape[1]):
            address = ((((r // 128) * (padded.shape[1] // 4) + c // 4) * 32
                        + r % 32) * 4 + (r % 128) // 32) * 4 + c % 4
            output[address] = padded[r, c]
    return output


def _decode_e4m3(values):
    sign = np.where(values & 128, -1.0, 1.0)
    exponent = (values >> 3) & 15
    fraction = values & 7
    return sign * np.where(exponent == 0, fraction * 2.0**-9,
                           (1.0 + fraction / 8.0) * 2.0**(exponent.astype(int) - 7))


def test_padded_scale_layout_round_trip():
    rng = np.random.default_rng(11)
    values = rng.integers(0, 127, (140, 7), dtype=np.uint8)
    np.testing.assert_array_equal(unswizzle_nvfp4_scales(_swizzle(values), 140, 7), values)
    with pytest.raises(ValueError, match="bytes"):
        unswizzle_nvfp4_scales(np.zeros(1, np.uint8), 140, 7)


@pytest.mark.parametrize("global_scale", [0.5, 4.0])
def test_serialized_encoder_linear_matches_independent_fp4_reference(tmp_path, global_scale):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(3)
    packed = rng.integers(0, 256, (128, 64), dtype=np.uint8)
    scales = rng.integers(24, 96, (128, 8), dtype=np.uint8)
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    save_file({prefix + ".weight_packed": packed,
               prefix + ".weight_scale": _swizzle(scales),
               prefix + ".weight_global_scale": np.array([global_scale], np.float32)},
              tmp_path / "model.safetensors")
    index = _ResidentNVFP4Index(_ShardIndex(tmp_path))
    weight = index.get_mlx(prefix + ".weight")
    assert isinstance(weight, NVFP4Matrix)
    assert weight.weight.dtype == mx.uint32
    assert weight.scales.dtype == mx.uint8
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    dense = np.stack((lut[packed & 15], lut[packed >> 4]), axis=-1).reshape(128, 128)
    dense *= np.repeat(_decode_e4m3(scales), 16, axis=1) / global_scale
    x = rng.standard_normal((3, 128)).astype(np.float32)
    np.testing.assert_allclose(np.array(weight.matmul(mx.array(x))), x @ dense.T, rtol=3e-5, atol=1e-3)
    index.close()
    assert not index.weights


def test_resident_embedding_keeps_bf16_storage(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    key = "model.language_model.embed_tokens.weight"
    table = torch.arange(60).reshape(10, 6).to(torch.bfloat16)
    save_file({key: table}, tmp_path / "model.safetensors")
    index = _ResidentNVFP4Index(_ShardIndex(tmp_path))
    assert index.get_mlx(key).dtype == mx.bfloat16
    conditioner = ResidentNVFP4MiniMaxH3TextConditioner.__new__(ResidentNVFP4MiniMaxH3TextConditioner)
    conditioner.index = index
    np.testing.assert_array_equal(np.array(conditioner._embed_tokens([7, 1])), table[[7, 1]].float().numpy())
    conditioner.close()


def _pipeline():
    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    pipeline.resident = True
    pipeline._resident_components = {}
    pipeline.dit_checkpoint = "tiny"
    pipeline.model_root = __import__("pathlib").Path("tiny")
    pipeline.vae_dtype = "fp16"
    return pipeline


def test_resident_preload_reuses_models_and_evaluates_audio(monkeypatch):
    import fastvideo.mlx_runtime.minimax_h3_pipeline as module
    import fastvideo.mlx_runtime.minimax_h3_audio_vae as audio
    import fastvideo.mlx_runtime.minimax_h3_video_vae as video

    pipeline = _pipeline()
    conditioner = ResidentNVFP4MiniMaxH3TextConditioner.__new__(ResidentNVFP4MiniMaxH3TextConditioner)
    conditioner.index = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(pipeline, "_load_conditioner", lambda: conditioner)
    calls = []
    def load_dit(path):
        calls.append(path)
        return SimpleNamespace(weights={"x": mx.ones((1,))}, blocks=[], refiner=[], _adaln_cache=None)
    monkeypatch.setattr(module, "load_mlx_h3_checkpoint", load_dit)
    monkeypatch.setattr(video, "mlx_h3_video_vae_from_dir", lambda *a, **k: object())
    decoder = SimpleNamespace(weights={"x": mx.ones((2,))})
    monkeypatch.setattr(audio, "mlx_h3_audio_vae_from_dir", lambda *a, **k: decoder)
    pipeline.prepare_resident()
    pipeline.prepare_resident()
    assert calls == ["tiny"]
    assert set(pipeline._resident_components) == {"conditioner", "dit", "video_vae", "audio_vae"}
    pipeline.close()
    assert not pipeline._resident_components


def test_failed_preload_releases_encoder(monkeypatch):
    import fastvideo.mlx_runtime.minimax_h3_pipeline as module

    pipeline = _pipeline()
    conditioner = ResidentNVFP4MiniMaxH3TextConditioner.__new__(ResidentNVFP4MiniMaxH3TextConditioner)
    closed = []
    conditioner.index = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(pipeline, "_load_conditioner", lambda: conditioner)
    def fail(path):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(module, "load_mlx_h3_checkpoint", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        pipeline.prepare_resident()
    assert closed == [True]
    assert not pipeline._resident_components


def test_single_shard_omits_unused_language_layers_and_vision(tmp_path):
    from safetensors.numpy import save_file

    kept = "model.language_model.layers.49.input_layernorm.weight"
    dropped = "model.language_model.layers.50.input_layernorm.weight"
    save_file({kept: np.ones(8, np.float32), dropped: np.ones(8, np.float32),
               "model.visual.weight": np.ones((8, 8), np.float32)}, tmp_path / "model.safetensors")
    index = _ResidentNVFP4Index(_ShardIndex(tmp_path))
    assert set(index.weights) == {kept}
    index.close()
