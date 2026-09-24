# SPDX-License-Identifier: Apache-2.0
"""CUDA numerical parity for the serialized int8 MiniMax-H3 transformer GEMM.

The CPU tests pin the contract on small shapes with an exact int32 reference.
These run the real ``torch._int_mm`` path on the seven H3 block-linear
geometries, which is where a transposed operand, a bad chunk boundary or a
missing rescale would show up and the CPU fallback would not.
"""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29517")

from fastvideo.layers.quantization.minimax_h3_int8 import (
    INT8_ROW_CHUNK,
    int8_linear,
    quantize_rows_int8,
)

# (name, input_size, output_size) for the FastH3 geometry: hidden 5376,
# 56 heads x 128 head_dim = 7168 inner, ffn 14336 with a packed SwiGLU fc_in.
H3_LINEARS = (
    ("attn.to_q", 5376, 7168),
    ("attn.to_k", 5376, 7168),
    ("attn.to_v", 5376, 7168),
    ("attn.to_out", 7168, 5376),
    ("attn.to_gate_compress", 5376, 7168),
    ("ff.fc_in", 5376, 28672),
    ("ff.fc_out", 14336, 5376),
)


def _require_int8_tensor_cores() -> None:
    if not torch.cuda.is_available():
        pytest.skip("serialized int8 execution needs a CUDA device")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 75:
        pytest.skip(f"int8 tensor cores need compute capability 7.5+, got {major}.{minor}")


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return ((output.float() - reference).norm() / reference.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("name, input_size, output_size", H3_LINEARS, ids=[row[0] for row in H3_LINEARS])
def test_int8_linear_matches_bf16_on_h3_geometry(name: str, input_size: int, output_size: int):
    """W8A8 noise on random rows is about 1.3%; a layout bug reads as 1.0."""
    _require_int8_tensor_cores()
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(output_size, input_size, generator=generator, device="cuda", dtype=torch.float32).bfloat16()
    x = torch.randn(512, input_size, generator=generator, device="cuda", dtype=torch.float32).bfloat16()

    weight_int8, weight_scale = quantize_rows_int8(weight)
    assert weight_int8.dtype == torch.int8
    assert tuple(weight_int8.shape) == (output_size, input_size)
    assert tuple(weight_scale.shape) == (output_size, )

    output = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.bfloat16)
    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == (512, output_size)
    assert torch.isfinite(output).all()

    reference = x.float() @ weight.float().t()
    assert _relative_error(output, reference) < 0.05, f"{name}: int8 GEMM drifted from the bf16 product"


def test_chunking_is_invisible_above_the_row_chunk():
    """A sequence longer than one chunk must give the same answer as a small chunk."""
    _require_int8_tensor_cores()
    generator = torch.Generator(device="cuda").manual_seed(1)
    weight = torch.randn(5376, 7168, generator=generator, device="cuda", dtype=torch.float32).bfloat16()
    x = torch.randn(INT8_ROW_CHUNK + 777, 7168, generator=generator, device="cuda", dtype=torch.float32).bfloat16()
    weight_int8, weight_scale = quantize_rows_int8(weight)

    whole = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    chunked = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.float32, row_chunk=1024)
    assert torch.equal(whole, chunked)
    assert _relative_error(whole, x.float() @ weight.float().t()) < 0.05


@pytest.mark.parametrize("rows", [1, 7, 31, 32, 33])
def test_small_batches_take_the_padded_path(rows: int):
    """``torch._int_mm`` refuses tiny batches; the padded rows must not leak into the output."""
    _require_int8_tensor_cores()
    generator = torch.Generator(device="cuda").manual_seed(2)
    weight = torch.randn(5376, 5376, generator=generator, device="cuda", dtype=torch.float32).bfloat16()
    x = torch.randn(rows, 5376, generator=generator, device="cuda", dtype=torch.float32).bfloat16()
    weight_int8, weight_scale = quantize_rows_int8(weight)

    output = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    assert tuple(output.shape) == (rows, 5376)
    assert _relative_error(output, x.float() @ weight.float().t()) < 0.05
    # One row at a time must agree with the batched answer, which it cannot if
    # the padding rows were read back.
    for index in range(rows):
        single = int8_linear(x[index:index + 1], weight_int8, weight_scale, out_dtype=torch.float32)
        assert _relative_error(single, output[index:index + 1].float()) < 1e-6
