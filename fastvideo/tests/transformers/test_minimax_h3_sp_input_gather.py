# SPDX-License-Identifier: Apache-2.0
"""Exact FP4 input gathering and source-ordered head packets on Blackwell."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


def _swizzle(logical):
    rows, groups = logical.shape
    return logical.reshape(rows // 128, 4, 32, groups // 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(rows, groups)


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 7430])
@pytest.mark.parametrize("groups", [4, 336])
def test_join_scale_rows_preserves_unaligned_sources_and_zeros_padding(rows, groups):
    from fastvideo.models.dits.minimax_h3_fusions.sp_input_gather import join_scale_rows
    torch.manual_seed(217)
    padded, total = (rows + 127) // 128 * 128, (2 * rows + 127) // 128 * 128
    logical = torch.randint(0, 256, (2, padded, groups), dtype=torch.uint8, device="cuda")
    source = torch.stack([_swizzle(part) for part in logical])
    expected = torch.zeros((total, groups), dtype=torch.uint8, device="cuda")
    expected[:2 * rows] = logical[:, :rows].reshape(2 * rows, groups)
    assert torch.equal(join_scale_rows(source, rows), _swizzle(expected))


@pytest.mark.parametrize("rows", [1, 127, 7430])
@pytest.mark.parametrize("projection", [0, 1, 2])
@pytest.mark.parametrize("rank", [0, 1])
def test_gathered_head_packets_match_original_exchange(rows, projection, rank):
    from fastvideo.models.dits.minimax_h3_fusions.sp_input_gather import pack_gathered
    from fastvideo.models.dits.minimax_h3_fusions.sp_projection import allocate, pack_one, pack_value
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("ordered QK packing requires Blackwell")
    torch.manual_seed(29176)
    values = torch.randn((2, rows, 56, 128), device="cuda", dtype=torch.bfloat16)
    values[:, ::127] = 0
    weight = torch.randn((128,), device="cuda", dtype=torch.bfloat16)
    weight[::17] = 0
    cosine, sine = [torch.randn((2, rows, 96), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    gathered = values[:, :, rank * 28:(rank + 1) * 28].contiguous().reshape(2 * rows, 3584)
    payload, scale = allocate(rows, 56, 2, 7, values.device)
    pack_gathered(gathered, payload, scale, projection, weight, cosine.flatten(0, 1), sine.flatten(0, 1), 1e-5)
    for source in range(2):
        expected, expected_scale = allocate(rows, 56, 2, 7, values.device)
        if projection == 2:
            pack_value(values[source], expected, expected_scale)
        else:
            pack_one(values[source], weight, cosine[source], sine[source], expected, expected_scale, 1e-5, projection)
        assert torch.equal(payload[:, source, projection].contiguous().view(torch.uint8),
                           expected[:, rank, projection].contiguous().view(torch.uint8))
        assert torch.equal(scale[:, source, projection], expected_scale[:, rank, projection])


def test_gathered_input_head_gemm_matches_sequence_projection():
    from flashinfer import SfLayout, nvfp4_quantize
    from fastvideo.layers.quantization.nvfp4_config import _mm_fp4
    from fastvideo.models.dits.minimax_h3_fusions.sp_input_gather import join_scale_rows
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("production route is restricted to GB10")
    torch.manual_seed(62071)
    rows, inner, columns = 7430, 5376, 7168
    unit = torch.ones((), device="cuda", dtype=torch.float32)
    x = torch.randn((2, rows, inner), device="cuda", dtype=torch.bfloat16)
    x[1] *= 8
    x[:, ::127] = 0
    w = torch.randn((columns, inner), device="cuda", dtype=torch.bfloat16) / inner**.5
    global_scale = torch.full((), 13721.5, device="cuda", dtype=torch.float32)
    b, sb = nvfp4_quantize(w, global_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    inputs = [nvfp4_quantize(part, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False) for part in x]
    full_a = torch.cat([p[0] for p in inputs])
    full_sa = join_scale_rows(torch.stack([p[1] for p in inputs]), rows)
    expected = [_mm_fp4(a, b.T, sa, sb.T, 1.0 / global_scale, torch.bfloat16, None, backend="cudnn")
                for a, sa in inputs]
    for rank in (0, 1):
        start, stop = rank * 3584, (rank + 1) * 3584
        output = _mm_fp4(full_a, b[start:stop].T, full_sa, sb[start:stop].T,
                         1.0 / global_scale, torch.bfloat16, None, backend="cudnn")
        target = torch.cat([part[:, start:stop] for part in expected])
        assert torch.equal(output.view(torch.int16), target.view(torch.int16))


@pytest.mark.parametrize("rank", [0, 1])
def test_completed_packets_order_other_stream_and_retain_source(rank):
    from fastvideo.distributed.projection_exchange import completed_projection_packets
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        payload = torch.empty((7, 2, 3, 127, 4, 128), device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty(payload.shape[:-1], device="cuda", dtype=torch.float32)
        torch.cuda._sleep(5_000_000)
        payload.view(torch.uint8).fill_(37)
        scale.fill_(2.5)
        packets = completed_projection_packets(payload, scale, rank)
    with torch.cuda.stream(consumer):
        received = [(p.wait().materialize(), s.wait().clone()) for p, s in packets]
    consumer.synchronize()
    del payload, scale
    for payload, scale in received:
        assert torch.all(payload.view(torch.uint8) == 37)
        assert torch.all(scale == 2.5)


def test_input_gather_guard_rejects_other_shapes_and_backend(monkeypatch):
    from fastvideo.models.dits.minimax_h3_fusions import sp_input_gather
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("production route is restricted to GB10")
    hidden = torch.empty((1, 7430, 5376), device="cuda", dtype=torch.bfloat16)
    linear = SimpleNamespace(_nvfp4_weight=torch.empty((7168, 2688), dtype=torch.uint8, device="cuda"),
                             _nvfp4_weight_scale=torch.empty((7168, 336), dtype=torch.uint8, device="cuda"),
                             bias=None)
    attn = SimpleNamespace(num_attention_heads=56, to_q=linear, to_k=linear, to_v=linear)
    monkeypatch.setattr(sp_input_gather, "_mm_fp4_backend", lambda: "cudnn")
    assert sp_input_gather.supports_input_gather(attn, hidden)
    assert not sp_input_gather.supports_input_gather(attn, hidden[:, :-1])
    monkeypatch.setattr(sp_input_gather, "_mm_fp4_backend", lambda: "cutlass")
    assert not sp_input_gather.supports_input_gather(attn, hidden)
    monkeypatch.setattr(sp_input_gather, "_mm_fp4_backend", lambda: "cudnn")
    linear.bias = torch.zeros((7168,), device="cuda")
    assert not sp_input_gather.supports_input_gather(attn, hidden)
