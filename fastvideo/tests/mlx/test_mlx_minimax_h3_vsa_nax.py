# SPDX-License-Identifier: Apache-2.0
"""Independent selected-token checks for the explicitly requested M5 backend."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="Apple MLX is required")

from fastvideo.mlx_runtime import minimax_h3_vsa as vsa
from fastvideo.mlx_runtime import minimax_h3_vsa_nax as nax


@pytest.fixture
def nax_device():
    if not mx.metal.is_available():
        pytest.skip("NAX attention requires Metal on an Apple M5 GPU")
    if not str(mx.device_info().get("device_name", "")).startswith("Apple M5"):
        pytest.skip("NAX attention is validated only on Apple M5 GPUs")


def _selected_rows(indices, counts, geometry, head, query):
    tile = geometry.tile_elems
    query_tile = geometry.num_prefix_tiles + query
    query_rows = np.arange(query_tile * tile, query_tile * tile + geometry.variable_block_sizes[query_tile])
    selected = indices[head, query, :counts[head, query]]
    key_rows = (np.concatenate([np.arange(index * tile, index * tile + geometry.variable_block_sizes[index])
                                for index in selected]) if len(selected) else np.empty(0, dtype=np.int64))
    return query_rows, key_rows


def _numpy_reference(q, k, value, indices, counts, geometry, scale):
    """Select individual valid tokens independently of all MLX block masks."""
    result = np.zeros_like(q, dtype=np.float64)
    for head in range(q.shape[1]):
        for query in range(geometry.num_video_tiles):
            query_rows, key_rows = _selected_rows(indices, counts, geometry, head, query)
            if not len(key_rows):
                continue
            scores = q[query_rows, head].astype(np.float64) @ k[key_rows, head].astype(np.float64).T * scale
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            result[query_rows, head] = probabilities @ value[key_rows, head].astype(np.float64)
    return result


def _native_reference(q, k, value, indices, counts, geometry, scale):
    """Gather the same valid tokens; no custom kernel, padding, masks or GQA."""
    result = np.zeros(q.shape, dtype=np.float32)
    for head in range(q.shape[1]):
        for query in range(geometry.num_video_tiles):
            query_rows, key_rows = _selected_rows(indices, counts, geometry, head, query)
            if not len(key_rows):
                continue
            output = mx.fast.scaled_dot_product_attention(
                q[mx.array(query_rows), head][None, None],
                k[mx.array(key_rows), head][None, None],
                value[mx.array(key_rows), head][None, None], scale=scale, force_fused=True)
            mx.eval(output)
            result[query_rows, head] = np.asarray(output[0, 0].astype(mx.float32))
    return result


def _metrics(actual, expected):
    difference = actual.astype(np.float64) - expected
    return {"max_abs": float(np.abs(difference).max()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30))}


def _case(tile_size, dtype_name, *, segments=None):
    # Dense/reference/target regions are interleaved before tiling. Ragged tiles
    # cross 16/32/64-row boundaries, including inactive SIMD groups in a launch.
    geometry = vsa.build_h3_segment_geometry(segments or (7, (1, 3, 5), 13, (3, 7, 11)), tile_size)
    rng = np.random.default_rng(9241)
    shape = (geometry.padded_length, 3, 128)
    arrays = [rng.normal(size=shape).astype(np.float32) * 0.6 for _ in range(3)]
    padding = np.concatenate([np.arange(index * tile_size + size, (index + 1) * tile_size)
                              for index, size in enumerate(geometry.variable_block_sizes)])
    for index, array in enumerate(arrays):
        array[padding] = 30.0 + index
    width = min(5, geometry.num_tiles)
    # Intentionally out of bounds: inactive route slots must never be read.
    indices = np.full((3, geometry.num_video_tiles, width), np.iinfo(np.int32).max, dtype=np.int32)
    counts = np.empty((3, geometry.num_video_tiles), dtype=np.int32)
    for head in range(3):
        for query in range(geometry.num_video_tiles):
            count = 1 + (head + query) % width
            counts[head, query] = count
            indices[head, query, :count] = rng.permutation(geometry.num_tiles)[:count]
    counts[-1, -1] = 0
    indices[-1, -1] = np.iinfo(np.int32).max
    tensors = [mx.array(array).astype(getattr(mx, dtype_name)) for array in arrays]
    mx.eval(*tensors)
    return geometry, tensors, indices, counts, padding


# MLX's installed steel/attn/nax.h sets MPP relaxed_precision=true even for
# float32. The native gathered control reproduces its ~7e-4 NumPy discrepancy
# on M5. Keep the 1e-3 independent FP32 bound and record BOTH numerical controls;
# changing the selected tokens must not be excused as a precision difference.
# The probability accumulation schedule differs from the native dense kernel.
@pytest.mark.parametrize("dtype_name,tolerance", [("float16", 0.003), ("bfloat16", 0.02), ("float32", 0.001)])
@pytest.mark.parametrize("tile_size,query_rows", [(128, 16), (128, 32), (128, 64), (128, 128),
                                                  (64, 16), (64, 32), (64, 64)])
def test_nax_matches_independent_valid_token_oracle(nax_device, record_property, dtype_name,
                                                   tolerance, tile_size, query_rows):
    geometry, tensors, indices, counts, padding = _case(tile_size, dtype_name)
    scale = 128**-0.5
    rounded = [np.asarray(tensor.astype(mx.float32)) for tensor in tensors]
    expected = _numpy_reference(*rounded, indices, counts, geometry, scale)
    actual = nax.nax_block_sparse(*tensors, mx.array(indices), mx.array(counts), geometry,
                                  scale, query_rows=query_rows)
    mx.eval(actual)
    got = np.asarray(actual.astype(mx.float32))
    record_property("candidate_vs_numpy", json.dumps(_metrics(got, expected)))
    assert actual.shape == tensors[0].shape
    assert actual.dtype == tensors[0].dtype
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(got[padding], 0)
    np.testing.assert_array_equal(got[:geometry.num_prefix_tiles * tile_size], 0)
    zero_rows, _ = _selected_rows(indices, counts, geometry, 2, geometry.num_video_tiles - 1)
    np.testing.assert_array_equal(got[zero_rows, 2], 0)
    if dtype_name == "float32":
        native = _native_reference(*tensors, indices, counts, geometry, scale)
        record_property("native_vs_numpy", json.dumps(_metrics(native, expected)))
        record_property("candidate_vs_native", json.dumps(_metrics(got, native)))
        np.testing.assert_allclose(native, expected, atol=tolerance, rtol=tolerance)
        # Both paths have independently bounded NumPy error. This pairwise
        # budget is their sum, not a claim of bitwise native-kernel equivalence.
        np.testing.assert_allclose(got, native, atol=2 * tolerance, rtol=2 * tolerance)


@pytest.mark.parametrize("query_rows", [16, 32, 64, 128])
def test_nax_uniform_logits_expose_head_and_tile_selection(nax_device, query_rows):
    geometry, tensors, indices, counts, padding = _case(128, "float32")
    shape = tensors[0].shape
    q, k = mx.zeros(shape), mx.zeros(shape)
    value = np.zeros(shape, dtype=np.float32)
    for tile, size in enumerate(geometry.variable_block_sizes):
        for head in range(shape[1]):
            # Each tile and head carries a distinct basis vector. Uniform
            # logits make the answer a valid-row-weighted tile histogram.
            value[tile * 128:tile * 128 + size, head, tile] = head + 1
    value[padding] = 32.0
    expected = _numpy_reference(np.zeros(shape), np.zeros(shape), value, indices, counts, geometry, 0.071)
    actual = nax.nax_block_sparse(q, k, mx.array(value), mx.array(indices), mx.array(counts),
                                  geometry, 0.071, query_rows=query_rows)
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=2e-6)


def test_nax_without_dense_prefix_and_with_nondefault_scale(nax_device):
    geometry, tensors, indices, counts, padding = _case(64, "bfloat16", segments=((3, 7, 11),))
    assert geometry.num_prefix_tiles == 0
    expected = _numpy_reference(*[np.asarray(t.astype(mx.float32)) for t in tensors],
                                indices, counts, geometry, 0.17)
    actual = nax.nax_block_sparse(*tensors, mx.array(indices), mx.array(counts), geometry, 0.17,
                                  query_rows=32)
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, atol=0.02, rtol=0.02)
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32))[padding], 0)


def _attention_inputs():
    geometry = vsa.build_h3_segment_geometry((7, (1, 3, 5), 13, (3, 7, 11)), 128)
    rng = np.random.default_rng(874)
    shape = (geometry.total_seq_length, 2, 128)
    tensors = [mx.array(rng.normal(size=shape).astype(np.float32) * 0.1) for _ in range(4)]
    return geometry, tensors


def test_explicit_nax_preserves_router_dense_prefix_and_compression(nax_device, monkeypatch):
    geometry, (q, k, value, gate) = _attention_inputs()
    expected = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=0.5, ref_keep_rate=0.5,
                                     gate_compress=gate, impl="reference")
    calls = []
    original = nax.nax_block_sparse

    def record(*args, **kwargs):
        calls.append((args[3].shape, args[4].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(nax, "nax_block_sparse", record)
    stats = vsa.MiniMaxH3VSAStats()
    actual = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=0.5, ref_keep_rate=0.5,
                                  gate_compress=gate, impl="nax", stats=stats)
    mx.eval(actual, expected)
    assert calls == [((2, geometry.num_video_tiles, calls[0][0][-1]), (2, geometry.num_video_tiles))]
    assert stats.impl == "nax"
    assert stats.dense_fallback_reason is None
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-4)


def test_explicit_nax_propagates_kernel_failure(nax_device, monkeypatch):
    geometry, (q, k, value, _) = _attention_inputs()

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate NAX kernel failure")

    monkeypatch.setattr(nax, "nax_block_sparse", fail)
    with pytest.raises(RuntimeError, match="deliberate NAX kernel failure"):
        actual = vsa.h3_vsa_attention(q, k, value, geometry, sparsity=0.5, ref_keep_rate=0.5, impl="nax")
        mx.eval(actual)


def test_nax_requires_explicit_selection(monkeypatch):
    from fastvideo.mlx_runtime import minimax_h3_vsa_simd as simd

    monkeypatch.setattr(simd, "simd_kernel_available", lambda: True)
    assert "nax" in vsa.PUBLIC_VSA_IMPLS
    assert vsa.resolve_impl("nax", 128, 128) == "nax"
    assert vsa.resolve_impl("nax", 128, 64) == "nax"
    assert vsa.resolve_impl("auto", 128, 128) == "simd"
    assert vsa.resolve_impl("reference", 128, 128) == "reference"
    monkeypatch.setattr(simd, "simd_kernel_available", lambda: False)
    monkeypatch.setattr(simd, "simd_kernel_error", lambda: "test unavailable")
    assert vsa.resolve_impl("auto", 128, 128) == "reference"


@pytest.mark.parametrize("head_dim,tile_size", [(64, 128), (128, 256)])
def test_nax_rejects_unsupported_public_shapes(head_dim, tile_size):
    with pytest.raises(ValueError):
        vsa.resolve_impl("nax", head_dim, tile_size)


@pytest.mark.parametrize("device_name", ["Apple M4 Max", "Apple M3", "Apple M50", "AMD Radeon", ""])
def test_nax_rejects_unsupported_device_before_kernel_construction(monkeypatch, device_name):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device_name}, raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("An unsupported device must not construct a Metal kernel")

    monkeypatch.setattr(mx.fast, "metal_kernel", unexpected, raising=False)
    nax._nax_kernel.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="M5"):
            nax._nax_kernel()
    finally:
        nax._nax_kernel.cache_clear()


def test_nax_rejects_missing_metal(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal|M5"):
        nax._check_nax_device()


def test_nax_construction_failure_keeps_original_cause(monkeypatch):
    original_error = RuntimeError("deliberate compiler error")
    monkeypatch.setattr(nax, "_check_nax_device", lambda: None)
    monkeypatch.setattr(nax, "_header", lambda: ("unused", {}))

    def fail(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(mx.fast, "metal_kernel", fail, raising=False)
    nax._nax_kernel.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="kernel construction") as captured:
            nax._nax_kernel()
        assert captured.value.__cause__ is original_error
    finally:
        nax._nax_kernel.cache_clear()


def test_unvalidated_precision_override_is_not_a_public_option():
    with pytest.raises(TypeError, match="relaxed_precision"):
        nax.nax_block_sparse(None, None, None, None, None, None, 1.0,
                             relaxed_precision=False)


def _assert_nax_provenance_schema(provenance):
    assert provenance["relaxed_precision"] is True
    assert provenance["math_mode"] == "safe"
    assert provenance["key_rows"] == 32
    for name in ("source_sha256", "expanded_header_sha256"):
        assert len(provenance[name]) == 64
        int(provenance[name], 16)
    dependencies = provenance["installed_header_sha256"]
    assert dependencies
    for path, digest in dependencies.items():
        assert path.startswith("mlx/")
        assert not PurePosixPath(path).is_absolute()
        assert ".." not in PurePosixPath(path).parts
        assert len(digest) == 64
        int(digest, 16)


def test_nax_provenance_schema_without_installed_metal_headers(monkeypatch):
    # The Linux MLX CPU wheel need not ship Apple's Metal helper headers.
    # Exercise receipt formatting and hashing without consulting that package.
    header = "// synthetic expanded NAX header\n"
    dependencies = {"mlx/backend/metal/kernels/steel/attn/nax.h": hashlib.sha256(b"synthetic header").hexdigest()}
    monkeypatch.setattr(nax, "_header", lambda: (header, dependencies))
    provenance = nax.get_nax_provenance()
    _assert_nax_provenance_schema(provenance)
    assert provenance["expanded_header_sha256"] == hashlib.sha256(header.encode()).hexdigest()
    assert provenance["source_sha256"] == hashlib.sha256(nax._SOURCE.encode()).hexdigest()
    assert provenance["installed_header_sha256"] == dependencies


def test_nax_installed_header_provenance_on_supported_device(nax_device):
    # A supported M5 installation must supply the real validated headers;
    # missing or incompatible headers are a failure here, not a skip.
    provenance = nax.get_nax_provenance()
    _assert_nax_provenance_schema(provenance)
    include_root = nax._mlx_include_root()
    for relative, digest in provenance["installed_header_sha256"].items():
        assert hashlib.sha256((include_root / relative).read_bytes()).hexdigest() == digest


@pytest.fixture
def synthetic_nax_headers(monkeypatch, tmp_path):
    # Parse a tiny include tree without Metal or the installed private headers.
    # The expected API markers deliberately match the loader's known contract.
    dependency = "mlx/testing/nax_dependency.h"
    descriptor = "transpose_b,\n        true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate\n"
    header = (f'#include "{dependency}"\n'
              "struct BaseNAXFrag; struct NAXTile;\n"
              "get_coord(); row_reduce(); row_bin_op(); load_rows(); store_rows();\n"
              + descriptor * 2)
    files = {nax._NAX_HEADER: header, dependency: "// validated dependency\n"}
    hashes = {}
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(nax, "_mlx_include_root", lambda: tmp_path)
    monkeypatch.setattr(nax, "_VALIDATED_HEADER_SHA256", hashes)
    nax._header.cache_clear()
    try:
        yield tmp_path, dependency, hashes
    finally:
        nax._header.cache_clear()


def test_nax_rejects_modified_validated_header(synthetic_nax_headers):
    include_root, dependency, hashes = synthetic_nax_headers
    _, actual_hashes = nax._header()
    assert actual_hashes == hashes
    (include_root / dependency).write_text("// changed dependency with unchanged API\n")
    nax._header.cache_clear()
    with pytest.raises(RuntimeError, match="Unvalidated MLX private header") as captured:
        nax._header()
    assert dependency in str(captured.value)


def test_nax_rejects_unregistered_header_dependency(synthetic_nax_headers):
    include_root, _, hashes = synthetic_nax_headers
    unknown = "mlx/testing/unregistered_dependency.h"
    (include_root / unknown).write_text("// This dependency is outside the validated set.\n")
    header_path = include_root / nax._NAX_HEADER
    header_path.write_text(header_path.read_text() + f'#include "{unknown}"\n')
    hashes[nax._NAX_HEADER] = hashlib.sha256(header_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="Unvalidated MLX private header") as captured:
        nax._header()
    assert unknown in str(captured.value)


def test_pipeline_rejects_nax_prefix_reuse_before_loading_components():
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

    pipeline = object.__new__(MiniMaxH3MLXPipeline)
    with pytest.raises(ValueError, match="prefix|Prefix"):
        pipeline.generate("unused", output_path="unused.mp4", vsa_impl="nax", reuse_prefix_kv=True)
