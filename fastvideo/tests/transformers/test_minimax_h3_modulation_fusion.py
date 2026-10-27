# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3_fusions.modulation import (
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
)


EPS = 1e-6
SOL_ENGINE_BF16_TOLERANCE = 3e-2


def _chunk_tables(rows: int, hidden_size: int, *, device: torch.device | str = "cpu", dtype=torch.float32):
    wide = torch.randn(rows, 6 * hidden_size, device=device, dtype=dtype)
    tables = wide.chunk(6, dim=-1)
    assert all(table.stride() == (6 * hidden_size, 1) for table in tables)
    assert all(not table.is_contiguous() for table in tables)
    return tables


def _eager_rmsnorm_modulate(x, weight, scale, shift, index):
    normed = F.rms_norm(x, (x.shape[-1], ), weight, EPS)
    return normed * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)


def _eager_residual_gate_rmsnorm_modulate(residual, branch, gate, weight, scale, shift, index):
    hidden = residual + gate.index_select(0, index) * branch
    normed = F.rms_norm(hidden, (hidden.shape[-1], ), weight, EPS)
    modulated = normed * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)
    return hidden, modulated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_bf16_sol_engine_fusions_match_production_eager_within_tolerance():
    """Sol-Engine keeps fused intermediates in FP32 until its output stores."""
    pytest.importorskip("triton")
    torch.manual_seed(2)
    device = torch.device("cuda")
    batch, sequence_length, hidden_size, table_rows = 2, 9, 5376, 6
    residual = torch.randn(batch, sequence_length, hidden_size, device=device, dtype=torch.bfloat16)
    branch = torch.randn_like(residual)
    weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    shift, scale, gate, shift_mlp, scale_mlp, _ = _chunk_tables(
        table_rows,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    index = torch.tensor([5, 0, 4, 1, 3, 2, 5, 1, 0], device=device, dtype=torch.int64)

    expected_norm1 = _eager_rmsnorm_modulate(residual, weight, scale, shift, index)
    actual_norm1 = fused_rmsnorm_modulate(residual, weight, scale, shift, index, EPS)
    expected_hidden, expected_norm2 = _eager_residual_gate_rmsnorm_modulate(
        residual,
        branch,
        gate,
        weight,
        scale_mlp,
        shift_mlp,
        index,
    )
    actual_hidden, actual_norm2 = fused_residual_gate_rmsnorm_modulate(
        residual,
        branch,
        gate,
        weight,
        scale_mlp,
        shift_mlp,
        index,
        EPS,
    )

    # Production eager materializes BF16 after each PyTorch operator. The
    # single-kernel Sol-Engine path deliberately removes those round points.
    torch.testing.assert_close(
        actual_norm1,
        expected_norm1,
        rtol=SOL_ENGINE_BF16_TOLERANCE,
        atol=SOL_ENGINE_BF16_TOLERANCE,
    )
    torch.testing.assert_close(
        actual_hidden,
        expected_hidden,
        rtol=SOL_ENGINE_BF16_TOLERANCE,
        atol=SOL_ENGINE_BF16_TOLERANCE,
    )
    torch.testing.assert_close(
        actual_norm2,
        expected_norm2,
        rtol=SOL_ENGINE_BF16_TOLERANCE,
        atol=SOL_ENGINE_BF16_TOLERANCE,
    )


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda args: args | {"x": args["x"][0, 0]}, ValueError, "shape"),
        (lambda args: args | {"weight": args["weight"][:-1]}, ValueError, "weight"),
        (lambda args: args | {"index": args["index"][:-1]}, ValueError, "index"),
        (lambda args: args | {"index": args["index"].float()}, TypeError, "index"),
        (lambda args: args | {"eps": 0.0}, ValueError, "eps"),
    ],
)
def test_fusion_rejects_invalid_contracts(mutate, error, match):
    args = {
        "x": torch.randn(2, 3, 8),
        "weight": torch.randn(8),
        "scale": torch.randn(4, 8),
        "shift": torch.randn(4, 8),
        "index": torch.tensor([0, 3, 1]),
        "eps": EPS,
    }
    with pytest.raises(error, match=match):
        fused_rmsnorm_modulate(**mutate(args))


def test_residual_fusion_rejects_mismatched_branch():
    residual = torch.randn(2, 3, 8)
    with pytest.raises(ValueError, match="branch"):
        fused_residual_gate_rmsnorm_modulate(
            residual,
            torch.randn(2, 2, 8),
            torch.randn(4, 8),
            torch.randn(8),
            torch.randn(4, 8),
            torch.randn(4, 8),
            torch.tensor([0, 1, 2]),
            EPS,
        )


def test_triton_wrappers_fail_explicitly_on_cpu():
    x = torch.randn(1, 2, 8)
    branch = torch.randn_like(x)
    weight = torch.randn(8)
    gate = torch.randn(3, 8)
    scale = torch.randn(3, 8)
    shift = torch.randn(3, 8)
    index = torch.tensor([0, 2])

    with pytest.raises(RuntimeError, match="Triton|CUDA"):
        fused_rmsnorm_modulate(x, weight, scale, shift, index, EPS)
    with pytest.raises(RuntimeError, match="Triton|CUDA"):
        fused_residual_gate_rmsnorm_modulate(x, branch, gate, weight, scale, shift, index, EPS)


def test_triton_wrappers_reject_autograd_before_backend_check():
    x = torch.randn(1, 2, 8)
    branch = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(8, requires_grad=True)
    gate = torch.randn(3, 8)
    scale = torch.randn(3, 8)
    shift = torch.randn(3, 8)
    index = torch.tensor([0, 2])

    with pytest.raises(RuntimeError, match="forward-only"):
        fused_rmsnorm_modulate(x, weight, scale, shift, index, EPS)
    with pytest.raises(RuntimeError, match="forward-only"):
        fused_residual_gate_rmsnorm_modulate(
            x,
            branch,
            gate,
            weight.detach(),
            scale,
            shift,
            index,
            EPS,
        )
