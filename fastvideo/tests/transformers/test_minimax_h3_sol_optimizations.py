# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Sol-Engine-aligned MiniMax-H3 optimizations."""

from types import SimpleNamespace

import pytest
import torch


def test_minimax_h3_relayout_is_bit_exact_and_compile_safe() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")

    from fastvideo.attention.minimax_h3_relayout import (
        merge_heads,
        pack_qkv_destination_major,
    )

    rows, heads, head_dim, world = 17, 8, 16, 4
    fused = torch.randn(rows, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, v = fused[:, 0], fused[:, 1], fused[:, 2]
    heads_local = heads // world

    expected_pack = torch.empty(
        world,
        rows,
        heads_local,
        3 * head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    for index, tensor in enumerate((q, k, v)):
        shard = tensor.reshape(rows, world, heads_local, head_dim).permute(1, 0, 2, 3)
        expected_pack[..., index * head_dim:(index + 1) * head_dim].copy_(shard)

    compiled_pack = torch.compile(pack_qkv_destination_major, fullgraph=True, dynamic=False)
    actual_pack = compiled_pack(q, k, v, world)
    assert torch.equal(actual_pack, expected_pack)

    packed_output = torch.randn_like(expected_pack[..., :head_dim])
    expected_merge = packed_output.permute(1, 0, 2, 3).contiguous().reshape(rows, heads, head_dim)
    compiled_merge = torch.compile(merge_heads, fullgraph=True, dynamic=False)
    actual_merge = compiled_merge(packed_output)
    assert torch.equal(actual_merge, expected_merge)


def test_adaln_precompute_rejects_a_different_trajectory() -> None:
    from fastvideo.models.dits.minimax_h3 import (
        MiniMaxH3Transformer3DModel,
        _MiniMaxH3StepCursor,
    )

    cursor = _MiniMaxH3StepCursor(torch.device("cpu"), ((1.0,),))
    transformer = SimpleNamespace(
        _h3_adaln_cursor=cursor,
        transformer_blocks=[object()],
    )
    same_plan = [(torch.tensor([1.0]), torch.tensor([0]))]
    stats = MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, same_plan)
    assert stats["installed"] == 0

    different_plan = [(torch.tensor([0.5]), torch.tensor([0]))]
    with pytest.raises(RuntimeError, match="same denoising schedule"):
        MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, different_plan)


def test_minimax_h3_fusions_capture_in_fullgraph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")

    from fastvideo.models.dits.minimax_h3_fusions import (
        fused_qknorm_rope,
        fused_residual_gate_rmsnorm_modulate,
        fused_rmsnorm_modulate,
        minimax_h3_swiglu,
    )

    device, dtype = torch.device("cuda"), torch.bfloat16
    batch, rows, hidden = 1, 12, 128
    x = torch.randn(batch, rows, hidden, device=device, dtype=dtype)
    branch = torch.randn_like(x)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    table = torch.randn(18, hidden, device=device, dtype=dtype)
    index = torch.arange(rows, device=device).remainder(table.shape[0])
    q = torch.randn(batch, rows, 1, hidden, device=device, dtype=dtype)
    cos = torch.randn(rows, 96, device=device, dtype=dtype)
    sin = torch.randn_like(cos)

    cases = (
        (fused_rmsnorm_modulate, (x, weight, table, table, index, 1e-5)),
        (fused_residual_gate_rmsnorm_modulate, (x, branch, table, weight, table, table, index, 1e-5)),
        (fused_qknorm_rope, (q, weight, cos, sin, 1e-5)),
        (minimax_h3_swiglu, (torch.randn(batch, rows, 512, device=device, dtype=dtype),)),
    )
    with torch.inference_mode():
        for function, args in cases:
            expected = function(*args)
            actual = torch.compile(function, fullgraph=True, dynamic=False)(*args)
            expected_items = expected if isinstance(expected, tuple) else (expected,)
            actual_items = actual if isinstance(actual, tuple) else (actual,)
            for expected_item, actual_item in zip(expected_items, actual_items, strict=True):
                torch.testing.assert_close(actual_item, expected_item, atol=0, rtol=0)
