# SPDX-License-Identifier: Apache-2.0
"""Storage-bit and source-ownership checks for streamed H3 weights.

These tests read tiny synthetic files. GPU conversion tests allocate only
small MLX arrays and never load a model checkpoint.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX runtime module import requires MLX")

from fastvideo.mlx_runtime.minimax_h3_conditioner import (  # noqa: E402
    _ShardIndex,
    _read_safetensors_bf16,
    _read_safetensors_row,
)


def _write_tensor(tmp_path, bits: np.ndarray, dtype: str):
    prefix = b"prefix!!"
    header = {
        "prefix": {"dtype": "U8", "shape": [len(prefix)], "data_offsets": [0, len(prefix)]},
        "weight": {
            "dtype": dtype,
            "shape": list(bits.shape),
            "data_offsets": [len(prefix), len(prefix) + bits.nbytes],
        },
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + prefix + bits.tobytes())
    return str(path), header, 8 + len(encoded)


def test_bf16_reader_preserves_every_storage_bit_pattern(tmp_path) -> None:
    """Include signed zero, subnormals, infinity, and all NaN payloads."""
    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    path, header, offset = _write_tensor(tmp_path, bits, "BF16")
    actual = _read_safetensors_bf16(path, "weight", header, offset)
    assert actual.dtype == np.float32
    assert actual.shape == bits.shape
    words = actual.view(np.uint32)
    np.testing.assert_array_equal(words >> 16, bits)
    assert not np.any(words & np.uint32(0xFFFF))


@pytest.mark.parametrize("row", [0, 127, 255])
def test_bf16_row_reader_matches_full_read(tmp_path, row: int) -> None:
    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    path, header, offset = _write_tensor(tmp_path, bits, "BF16")
    full = _read_safetensors_bf16(path, "weight", header, offset)
    actual = _read_safetensors_row(path, "weight", row, header, offset)
    assert actual.shape == (256, )
    assert actual.nbytes == 256 * np.dtype(np.float32).itemsize
    np.testing.assert_array_equal(actual.view(np.uint32), full[row].view(np.uint32))


@pytest.mark.parametrize("row_only", [False, True])
def test_bf16_conversion_owns_writable_storage_without_changing_source(tmp_path, row_only: bool) -> None:
    bits = np.array([[0x3F80, 0x8000, 0x7FC1], [0x0001, 0xFF80, 0xFFFF]], dtype=np.uint16)
    path, header, offset = _write_tensor(tmp_path, bits, "BF16")
    before = (tmp_path / "weights.safetensors").read_bytes()
    if row_only:
        actual = _read_safetensors_row(path, "weight", 1, header, offset)
    else:
        actual = _read_safetensors_bf16(path, "weight", header, offset)
    assert actual.flags.writeable
    actual[...] = 42.0
    assert (tmp_path / "weights.safetensors").read_bytes() == before


@pytest.mark.parametrize("row", [-1, 2])
def test_bf16_row_reader_rejects_out_of_bounds(tmp_path, row: int) -> None:
    path, header, offset = _write_tensor(tmp_path, np.zeros((2, 3), dtype=np.uint16), "BF16")
    with pytest.raises(IndexError, match="outside leading dimension"):
        _read_safetensors_row(path, "weight", row, header, offset)


@pytest.mark.parametrize("dtype,code", [(np.float32, "F32"), (np.float16, "F16")])
def test_native_float_reads_keep_values_and_mapping_immutable(tmp_path, dtype, code: str) -> None:
    values = np.array([[1.5, -0.0, 0.25], [-2.0, 8.0, 16.0]], dtype=dtype)
    path, header, offset = _write_tensor(tmp_path, values, code)
    full = _read_safetensors_bf16(path, "weight", header, offset)
    assert not full.flags.writeable
    converted = np.asarray(full, dtype=np.float32)
    np.testing.assert_array_equal(converted, values.astype(np.float32))
    if dtype == np.float32:
        assert np.shares_memory(converted, full)
    else:
        assert not np.shares_memory(converted, full)
    row = _read_safetensors_row(path, "weight", 1, header, offset)
    np.testing.assert_array_equal(row, values[1])
    row[...] = 7.0
    np.testing.assert_array_equal(full, values)


def test_mlx_bf16_conversion_preserves_every_storage_bit_pattern(tmp_path) -> None:
    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    path, header, offset = _write_tensor(tmp_path, bits, "BF16")
    index = _ShardIndex.__new__(_ShardIndex)
    index.key_to_shard = {"weight": path}
    index._header_cache = {path: (header, offset)}
    before = (tmp_path / "weights.safetensors").read_bytes()
    actual = index.get_mlx("weight")
    assert actual.dtype == mx.float32
    assert actual.shape == bits.shape
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint32)), bits.astype(np.uint32) << 16)
    assert (tmp_path / "weights.safetensors").read_bytes() == before


@pytest.mark.parametrize("dtype,code", [(np.float32, "F32"), (np.float16, "F16")])
def test_mlx_native_float_fallback_preserves_values(tmp_path, dtype, code: str) -> None:
    values = np.array([[1.5, -0.0, 0.25], [-2.0, 8.0, 16.0]], dtype=dtype)
    path, header, offset = _write_tensor(tmp_path, values, code)
    index = _ShardIndex.__new__(_ShardIndex)
    index.key_to_shard = {"weight": path}
    index._header_cache = {path: (header, offset)}
    actual = index.get_mlx("weight")
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), values.astype(np.float32).view(np.uint32))


def test_bulk_bf16_read_rejects_truncated_tensor(tmp_path) -> None:
    path, header, offset = _write_tensor(tmp_path, np.ones((2, 3), dtype=np.uint16), "BF16")
    file = tmp_path / "weights.safetensors"
    file.write_bytes(file.read_bytes()[:-2])
    with pytest.raises(EOFError, match="expected 6 BF16 values, got 5"):
        _read_safetensors_bf16(path, "weight", header, offset)


@pytest.mark.parametrize("offsets", [[8, 6], [8, 11], [-2, 10]])
def test_bulk_bf16_read_rejects_invalid_offsets_before_reading(tmp_path, monkeypatch, offsets) -> None:
    path, header, offset = _write_tensor(tmp_path, np.ones((2, 3), dtype=np.uint16), "BF16")
    header["weight"]["data_offsets"] = offsets

    def reject_read(*_args, **_kwargs):
        raise AssertionError("invalid offsets reached np.fromfile")

    monkeypatch.setattr(np, "fromfile", reject_read)
    with pytest.raises(ValueError, match="Invalid BF16 data offsets"):
        _read_safetensors_bf16(path, "weight", header, offset)
