# SPDX-License-Identifier: Apache-2.0
"""Fullgraph trace contract for the MiniMax-H3 inference fusions."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from fastvideo.models.dits.minimax_h3_fusions import (
    fused_qknorm_rope,
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
    minimax_h3_swiglu,
)


def _all_fusions(
    x: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
    index: torch.Tensor,
    q: torch.Tensor,
    q_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    packed_mlp: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    norm1 = fused_rmsnorm_modulate(x, weight, scale, shift, index, 1e-5)
    hidden, norm2 = fused_residual_gate_rmsnorm_modulate(
        x,
        branch,
        gate,
        weight,
        scale,
        shift,
        index,
        1e-5,
    )
    q_out = fused_qknorm_rope(q, q_weight, cos, sin, 1e-5)
    mlp_out = minimax_h3_swiglu(packed_mlp)
    return norm1, hidden, norm2, q_out, mlp_out


def test_minimax_h3_fusions_are_dynamic_fullgraph_traceable() -> None:
    """Dynamo sees four opaque ops without requiring a physical CUDA GPU."""
    graphs: list[torch.fx.GraphModule] = []

    def capture_graph(graph: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        del example_inputs
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(_all_fusions, backend=capture_graph, fullgraph=True, dynamic=True)

    with torch.no_grad(), FakeTensorMode():
        x = torch.empty((2, 3, 8), device="cuda", dtype=torch.bfloat16)
        weight = torch.empty((8,), device="cuda", dtype=torch.bfloat16)
        table = torch.empty((6, 8), device="cuda", dtype=torch.bfloat16)
        index = torch.empty((3,), device="cuda", dtype=torch.int64)
        q = torch.empty((2, 3, 4, 8), device="cuda", dtype=torch.bfloat16)
        rotary = torch.empty((3, 8), device="cuda", dtype=torch.bfloat16)
        packed_mlp = torch.empty((2, 3, 16), device="cuda", dtype=torch.bfloat16)

        outputs = compiled(
            x,
            x,
            weight,
            table,
            table,
            table,
            index,
            q,
            weight,
            rotary,
            rotary,
            packed_mlp,
        )

    assert tuple(output.shape for output in outputs) == (
        (2, 3, 8),
        (2, 3, 8),
        (2, 3, 8),
        (2, 3, 4, 8),
        (2, 3, 8),
    )
    assert all(output.is_contiguous() for output in outputs)
    assert len(graphs) == 1
    call_targets = {node.target for node in graphs[0].graph.nodes if node.op == "call_function"}
    assert {
        torch.ops.fastvideo._minimax_h3_rmsnorm_modulate,
        torch.ops.fastvideo._minimax_h3_residual_gate_rmsnorm_modulate,
        torch.ops.fastvideo._minimax_h3_qknorm_rope,
        torch.ops.fastvideo._minimax_h3_swiglu,
    } <= call_targets


@pytest.mark.gpu
def test_minimax_h3_fusion_custom_ops_match_real_fullgraph_cuda() -> None:
    """Cross-check fake metadata and execute the real CUDA dispatcher path."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA is required")
    pytest.importorskip("triton")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(2026)
    x = torch.randn((2, 3, 128), generator=generator, device=device, dtype=torch.bfloat16)
    branch = torch.randn_like(x)
    weight = torch.randn((128,), generator=generator, device=device, dtype=torch.bfloat16)
    scale = torch.randn((6, 128), generator=generator, device=device, dtype=torch.bfloat16)
    shift = torch.randn((6, 128), generator=generator, device=device, dtype=torch.bfloat16)
    gate = torch.randn((6, 128), generator=generator, device=device, dtype=torch.bfloat16)
    index = torch.tensor((0, 3, 5), device=device, dtype=torch.int64)
    q = torch.randn((2, 3, 4, 128), generator=generator, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((128,), generator=generator, device=device, dtype=torch.bfloat16).contiguous()
    rotary = torch.randn((3, 96), generator=generator, device=device, dtype=torch.bfloat16).contiguous()
    packed_mlp = torch.randn((2, 3, 256), generator=generator, device=device, dtype=torch.bfloat16)

    opchecks = (
        (
            torch.ops.fastvideo._minimax_h3_rmsnorm_modulate,
            (x, weight, scale, shift, index, 1e-5),
        ),
        (
            torch.ops.fastvideo._minimax_h3_residual_gate_rmsnorm_modulate,
            (x, branch, gate, weight, scale, shift, index, 1e-5),
        ),
        (
            torch.ops.fastvideo._minimax_h3_qknorm_rope,
            (q, q_weight, rotary, rotary.clone(), 1e-5),
        ),
        (torch.ops.fastvideo._minimax_h3_swiglu, (packed_mlp,)),
    )
    for op, args in opchecks:
        torch.library.opcheck(op, args)

    compiled = torch.compile(_all_fusions, fullgraph=True, dynamic=True)
    inputs = (
        x,
        branch,
        weight,
        scale,
        shift,
        gate,
        index,
        q,
        q_weight,
        rotary,
        rotary.clone(),
        packed_mlp,
    )
    with torch.no_grad():
        expected = _all_fusions(*inputs)
        actual = compiled(*inputs)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=0.0, rtol=0.0)
