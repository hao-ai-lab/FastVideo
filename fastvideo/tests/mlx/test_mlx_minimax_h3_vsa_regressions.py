# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for H3 VSA configuration, conversion, and lazy dispatch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fastvideo.mlx_runtime import minimax_h3 as h3
from fastvideo.mlx_runtime import minimax_h3_vsa as vsa
from fastvideo.mlx_runtime import minimax_h3_vsa_simd as simd
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, QuantizedMatrix, quantize_matrix


def _config():
    return dict(hidden_size=64, num_attention_heads=1, attention_head_dim=64,
                ffn_dim=64, in_channels=4, audio_in_channels=4, patch_size=[1, 2, 2],
                text_dim=64, freq_dim=64, time_embed_dim=64, rope_freq_dim=8,
                rope_theta=10000., norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                num_layers=2, num_refiner_layers=0)


def _dit(capable=True):
    blocks = [{"attn.to_q.weight": mx.zeros((64, 64))} for _ in range(2)]
    if capable:
        for block in blocks:
            block["attn.to_gate_compress.weight"] = mx.zeros((64, 64))
    return h3.MLXMiniMaxH3DiT({}, blocks, [], _config())


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("prefix", [(), (7, 5)])
def test_gather_tiling_equals_scatter_with_ragged_padding(dtype, prefix):
    geometry = vsa.build_h3_tile_geometry(prefix, (5, 3, 7), 64)
    x = mx.random.normal((geometry.total_seq_length, 2, 8)).astype(dtype)
    expected = mx.zeros((geometry.padded_length, 2, 8), dtype=dtype)
    expected = expected.at[mx.array(geometry.untile_combined_index)].add(x)
    actual = vsa._tile_hidden(x, geometry)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(vsa._untile_hidden(actual, geometry), x).item()
    assert geometry.tile_gather_index is geometry.tile_gather_index
    prefix_tiled = mx.concatenate([x[:geometry.prefix_length], mx.zeros((1, 2, 8), dtype=dtype)])[
        geometry.prefix_gather_index]
    assert mx.array_equal(prefix_tiled, expected[:geometry.num_prefix_tiles * 64]).item()


@pytest.mark.parametrize("exempt", [True, False])
def test_small_canvas_bf16_mask_and_compete_stats(monkeypatch, exempt):
    geometry = vsa.build_h3_tile_geometry((7, 5), (8, 4, 8), 64)
    mx.random.seed(13)
    q, k, value = [mx.random.normal((geometry.total_seq_length, 2, 128)).astype(mx.bfloat16)
                   for _ in range(3)]
    stats = vsa.MiniMaxH3VSAStats()
    out = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=.75, exempt=exempt,
                               impl="reference", stats=stats)
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(out)).item()
    qt, kt = vsa._tile_hidden(q, geometry), vsa._tile_hidden(k, geometry)
    scores = (vsa._pool_tiles(qt, geometry.variable_block_sizes, 64) @
              vsa._pool_tiles(kt, geometry.variable_block_sizes, 64).transpose(0, 2, 1)) / 128**.5
    mask = vsa.build_block_mask(np.asarray(scores), geometry.num_prefix_tiles,
                               geometry.num_video_tiles, .75, exempt)
    kept = mask[:, geometry.num_prefix_tiles:, geometry.num_prefix_tiles:].sum(-1).mean()
    assert stats.video_keep == pytest.approx(kept)
    assert stats.achieved_sparsity == pytest.approx(1 - kept / geometry.num_video_tiles)
    monkeypatch.setattr(vsa, "_REFERENCE_FULL_MASK_TILE_LIMIT", 0)
    gathered = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=.75, exempt=exempt, impl="reference")
    np.testing.assert_allclose(np.asarray(out.astype(mx.float32)), np.asarray(gathered.astype(mx.float32)),
                               atol=.004, rtol=.004)


def test_reference_dispatch_ignores_auto_preference(monkeypatch):
    monkeypatch.setattr(vsa, "_AUTO_PREFERS_SIMD", True)
    monkeypatch.setattr(simd, "simd_kernel_available", lambda: pytest.fail("reference must not probe SIMD"))
    assert vsa.resolve_impl("reference", 128, 64) == "reference"


@pytest.mark.parametrize("shape", [(0, 4, 4), (4, -1, 4), (4, 4, 0)])
def test_invalid_video_axes_return_a_reason(shape):
    assert "positive" in vsa.geometry_is_supported((4,), shape, 64)


@pytest.mark.parametrize("layers", ["10", (-1,), (1.5,), (True,)])
def test_invalid_dense_layers_are_rejected(layers):
    with pytest.raises(ValueError, match="non-negative integer"):
        vsa.MiniMaxH3VSAConfig(dense_layers=layers)


def test_configuration_is_transactional_and_invalidates_geometry():
    dense = _dit(capable=False)
    before = dense.vsa_config
    with pytest.raises(vsa.DenseOnlyVSACheckpointError):
        dense.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True))
    assert dense.vsa_config is before
    dit = _dit()
    dit.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True))
    layout = h3.build_packed_layout(6, 2, 8, 16, 3, patch_size=(1, 2, 2))
    first = dit.prepare_vsa_geometry(layout)
    assert dit.prepare_vsa_geometry(layout) is first
    swapped = h3.build_packed_layout(6, 2, 16, 8, 3, patch_size=(1, 2, 2))
    second = dit.prepare_vsa_geometry(swapped)
    assert first.total_seq_length == second.total_seq_length
    assert second.dit_seq_shape != first.dit_seq_shape
    assert second is not first
    with pytest.raises(ValueError, match="block count"):
        dit.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True, dense_layers=(2,)))
    assert dit._vsa_geometry is second
    dit.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True, tile_size=256))
    assert dit._vsa_geometry is None
    assert dit.prepare_vsa_geometry(layout).tile_elems == 256


@pytest.fixture
def tiny_h3_model(monkeypatch, request):
    from fastvideo.tests.mlx.tiny_h3 import build_hf_config, build_tiny_h3_config, build_torch_model
    from fastvideo.tests.mlx.tiny_h3 import mlx_dit_from_torch_model

    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29513")
    request.getfixturevalue("distributed_setup")
    return mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_h3_config()))


def test_cached_forward_refreshes_geometry_and_collects_every_block(tiny_h3_model):
    source = tiny_h3_model
    for block in source.blocks:
        block["attn.to_gate_compress.weight"] = mx.full((512, 256), .001)
    dit = h3.MLXMiniMaxH3DiT(source.weights, source.blocks, source.refiner, source.config)
    dit.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True, sparsity=.5, dense_layers=(0,)))
    first_geometry = None
    for height, width in ((4, 8), (8, 4)):
        layout = h3.build_packed_layout(6, 8, height, width, 3, patch_size=(1, 2, 2))
        timesteps, inverse = h3.build_row_timesteps(layout, .7, .4)
        if dit._adaln_cache is None:
            dit.precompute_adaln(timesteps)
        output = dit.forward_with_cache(
            mx.zeros((len(layout.video_indices), dit.patch_dim)),
            mx.zeros((len(layout.audio_indices), dit.audio_in_channels)),
            mx.zeros((6, dit.text_dim)), layout=layout,
            step_timesteps=timesteps, row_timestep_inverse=inverse)
        mx.eval(output)
        assert all(mx.all(mx.isfinite(x)).item() for x in output)
        if first_geometry is None:
            first_geometry = dit._vsa_geometry
        else:
            assert dit._vsa_geometry is not first_geometry
            assert dit._vsa_geometry.dit_seq_shape == (8, 4, 2)
    assert dit.last_vsa_stats.attention_calls == 4
    assert dit.last_vsa_stats.impl_counts == {"dense": 2, "reference": 2}


def test_dense_generate_ignores_unused_vsa_parameters(tmp_path, monkeypatch):
    from fastvideo.mlx_runtime import minimax_h3_pipeline as pipeline_mod

    pipeline = object.__new__(pipeline_mod.MiniMaxH3MLXPipeline)
    pipeline.dit_checkpoint = tmp_path
    monkeypatch.setattr(pipeline_mod, "_validate_checkpoint_step_ladder", lambda *a: None)
    def stop_at_media_preflight(**kwargs):
        raise RuntimeError("reached media preflight")
    monkeypatch.setattr(pipeline_mod, "_preflight_media_dependencies", stop_at_media_preflight)
    with pytest.raises(RuntimeError, match="reached media preflight"):
        pipeline.generate("test", output_path=tmp_path / "unused.mp4", vsa=False,
                          vsa_sparsity=1., vsa_tile_size=-1, vsa_dense_layers="10", vsa_impl="invalid")
    with pytest.raises(ValueError, match="sparsity"):
        pipeline.generate("test", output_path=tmp_path / "unused.mp4", vsa=True, vsa_sparsity=1.)


def test_forward_rejects_vsa_before_computing():
    dit = _dit()
    dit.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True))
    with pytest.raises(ValueError, match="forward_with_cache"):
        dit.forward(None, None, None, position_ids=None, token_tags=None,
                    timestep_indices=None, timesteps=None, video_indices=None,
                    audio_indices=None, text_indices=None)


@pytest.mark.parametrize("fmt", ["int8", "int6", "int4"])
def test_quantized_zero_and_constant_gate_detection(fmt):
    spec = MLXQuantizationSpec.from_name(fmt)
    zero = quantize_matrix(mx.zeros((64, 64), dtype=mx.bfloat16), spec)
    constant = quantize_matrix(mx.full((64, 64), .25, dtype=mx.bfloat16), spec)
    assert not h3._gate_weight_is_active(zero)
    assert h3._gate_weight_is_active(constant)


@pytest.mark.parametrize("fmt", ["int8", "int6", "int4"])
def test_convert_save_load_preserves_quantized_gates(tmp_path, fmt):
    source = tmp_path / "source.safetensors"
    mx.random.seed(17)
    arrays = {}
    for i in range(2):
        arrays[f"transformer_blocks.{i}.attn.to_q.weight"] = mx.random.normal((64, 64))
        arrays[vsa.vsa_gate_key(i)] = mx.random.normal((64, 64))
    mx.save_safetensors(str(source), arrays)
    converted = h3.mlx_h3_dit_from_diffusers_safetensors(
        source, _config(), dtype="bf16", quantization=fmt, include_vsa=True)
    output = tmp_path / fmt
    h3.save_mlx_h3_checkpoint(converted, output)
    loaded = h3.load_mlx_h3_checkpoint(output)
    assert loaded.vsa_capable and h3.mlx_h3_checkpoint_vsa_capable(output)
    manifest = json.loads((output / h3.H3_MANIFEST_FILENAME).read_text())
    assert manifest["vsa"]["num_gate_matrices"] == 2
    for i in range(2):
        gate = loaded.blocks[i]["attn.to_gate_compress.weight"]
        original = converted.blocks[i]["attn.to_gate_compress.weight"]
        assert isinstance(gate, QuantizedMatrix)
        assert f"blocks.{i}.attn.to_gate_compress.weight" in manifest["quantized_keys"]
        for name in ("weight", "scales", "biases"):
            assert mx.array_equal(getattr(gate, name), getattr(original, name)).item()
    del arrays[vsa.vsa_gate_key(1)]
    mx.save_safetensors(str(source), arrays)
    with pytest.raises(KeyError, match="gate projections are missing"):
        h3.mlx_h3_dit_from_diffusers_safetensors(source, _config(), include_vsa=True)


def test_stats_include_dense_overrides_and_later_fallbacks():
    dit = _dit()
    dit.configure_vsa(vsa.MiniMaxH3VSAConfig(enabled=True, sparsity=.75, dense_layers=(0,)))
    dit.prepare_vsa_geometry(h3.build_packed_layout(6, 8, 8, 16, 3, patch_size=(1, 2, 2)))
    geometry = dit._vsa_geometry
    q = mx.random.normal((geometry.total_seq_length, 1, 8))
    for step in range(2):
        for block in range(2):
            kwargs = dit._vsa_block_kwargs(block, step)
            call = kwargs["vsa_stats"]
            vsa.h3_vsa_attention(q, q, q, geometry, sparsity=kwargs["vsa_sparsity"],
                                 impl="reference", stats=call)
            if step == 1 and block == 1:
                call.dense_fallback_reason = "simulated later-block failure"
            dit.last_vsa_stats.record(call)
    stats = dit.last_vsa_stats
    assert stats.configured_sparsity == .75
    assert stats.attention_calls == 4 and stats.sparse_calls == 2
    assert stats.impl_counts == {"dense": 2, "reference": 2}
    assert stats.impl == "mixed"
    assert stats.achieved_sparsity == pytest.approx(.375)
    assert stats.fallback_reasons == ["simulated later-block failure"]
    dit.reset_vsa_stats()
    assert dit.last_vsa_stats.attention_calls == 0


def test_no_metal_probe_is_cached(monkeypatch):
    monkeypatch.setattr(simd, "_SIMD_KERNEL", None)
    monkeypatch.setattr(simd, "_SIMD_KERNEL_ERROR", None)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert not simd.simd_kernel_available()
    assert "Metal is not available" in simd.simd_kernel_error()
    monkeypatch.setattr(mx.metal, "is_available", lambda: pytest.fail("failed probe retried"))
    assert not simd.simd_kernel_available()


def _require_metal():
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("requires actual Metal execution")


def test_probe_evaluates_lazy_compile_and_caches_failure(monkeypatch):
    _require_metal()
    monkeypatch.setattr(simd, "_SIMD_KERNEL", None)
    monkeypatch.setattr(simd, "_SIMD_KERNEL_ERROR", None)
    monkeypatch.setattr(simd, "_SIMD_SOURCE", "invalid_metal_source_for_test;")
    assert not simd.simd_kernel_available()
    assert simd.simd_kernel_error()
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: pytest.fail("failed compile retried"))
    assert not simd.simd_kernel_available()


def test_lazy_execution_failure_falls_back_and_disables_simd(monkeypatch):
    _require_metal()
    monkeypatch.setattr(simd, "_SIMD_KERNEL", None)
    monkeypatch.setattr(simd, "_SIMD_KERNEL_ERROR", None)
    monkeypatch.setattr(vsa, "resolve_impl", lambda *args: "simd")
    kernel = mx.fast.metal_kernel(name="h3_test_invalid_kernel", input_names=["x"],
                                 output_names=["out"], source="invalid_metal_source_for_test;")
    def fail_lazily(q, *args):
        return kernel(inputs=[q], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                      output_shapes=[q.shape], output_dtypes=[q.dtype])[0]
    monkeypatch.setattr(simd, "simd_block_sparse", fail_lazily)
    geometry = vsa.build_h3_tile_geometry((4,), (4, 4, 4), 64)
    q = mx.ones((geometry.total_seq_length, 1, 128), dtype=mx.bfloat16)
    stats = vsa.MiniMaxH3VSAStats()
    out = vsa.h3_vsa_attention(q, q, q, geometry, sparsity=.5, impl="simd", stats=stats)
    mx.eval(out)
    assert mx.array_equal(out, q).item()
    assert stats.impl == "reference" and "simd kernel failed" in stats.dense_fallback_reason
    assert not simd.simd_kernel_available()


def test_cli_help_and_validation_without_site_packages():
    script = Path(__file__).resolve().parents[3] / "examples/inference/basic/mlx_fasth3.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--vsa-impl" in result.stdout
    for value in ("-1", "oops"):
        result = subprocess.run([sys.executable, "-S", str(script), "--mlx-checkpoint", "missing",
                                 "--prompt", "test", "--output-path", "unused.mp4",
                                 "--vsa-dense-layers", value], capture_output=True, text=True)
        assert result.returncode == 2
        assert "--vsa-dense-layers" in result.stderr and "Traceback" not in result.stderr


def test_converter_continues_past_mismatched_existing_format(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/checkpoint_conversion/convert_minimax_h3_mlx.py"
    spec = importlib.util.spec_from_file_location("h3_converter_regression", path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    existing = tmp_path / "int8"
    existing.mkdir()
    (existing / h3.H3_MANIFEST_FILENAME).write_text(json.dumps({"vsa": {"capable": False}}))
    (existing / h3.H3_WEIGHTS_FILENAME).write_bytes(b"existing")
    monkeypatch.setattr(converter, "parse_args", lambda: argparse.Namespace(
        formats="int8 int6", out=tmp_path, model_root="unused", include_vsa=True))
    monkeypatch.setattr(converter, "mlx_h3_dit_from_diffusers_safetensors", lambda *a, **k: _dit())
    saved = []
    monkeypatch.setattr(converter, "save_mlx_h3_checkpoint", lambda model, path: saved.append(path.name))
    converter.main()
    assert saved == ["int6"]
    assert (existing / h3.H3_WEIGHTS_FILENAME).read_bytes() == b"existing"
