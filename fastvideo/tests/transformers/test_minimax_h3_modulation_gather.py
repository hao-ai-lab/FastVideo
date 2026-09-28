# SPDX-License-Identifier: Apache-2.0
"""Training-path AdaLN row gather of the MiniMax-H3 transformer."""

import pytest
import torch

from fastvideo.models.dits import minimax_h3
from fastvideo.models.dits.minimax_h3 import _gather_modulation_rows

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def _layout_indices(sequence_length: int, num_rows: int, *, shuffled: bool) -> torch.Tensor:
    """Contiguous modality/timestep segments, or the same rows in arbitrary order."""
    bounds = torch.linspace(0, sequence_length, num_rows + 1).long()
    indices = torch.repeat_interleave(torch.arange(num_rows), bounds.diff())
    if shuffled:
        indices = indices[torch.randperm(sequence_length)]
    return indices


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def test_gather_forward_matches_index_select() -> None:
    torch.manual_seed(0)
    table = torch.randn(6, 16, dtype=torch.bfloat16, requires_grad=True)
    indices = _layout_indices(37, 6, shuffled=True)

    rows = _gather_modulation_rows(table, indices)

    assert type(rows.grad_fn).__name__ == "_ModulationRowGatherBackward"
    torch.testing.assert_close(rows, table.detach().index_select(0, indices), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("num_rows", [3, 6])
@pytest.mark.parametrize("shuffled", [False, True])
def test_gather_gradient_matches_index_select_in_fp64(device: str, num_rows: int, shuffled: bool) -> None:
    """Any token order and number of timesteps gets the exact per-row gradient sums."""
    torch.manual_seed(0)
    table = torch.randn(num_rows, 16, device=device, dtype=torch.float64, requires_grad=True)
    indices = _layout_indices(37, num_rows, shuffled=shuffled).to(device)
    hidden = torch.randn(2, 37, 16, device=device, dtype=torch.float64)

    (hidden * _gather_modulation_rows(table, indices)).square().sum().backward()
    expected_table = table.detach().clone().requires_grad_(True)
    (hidden * expected_table.index_select(0, indices)).square().sum().backward()

    torch.testing.assert_close(table.grad, expected_table.grad, rtol=1e-12, atol=1e-12)


@requires_cuda
@pytest.mark.parametrize(("grad_mean", "min_index_select_error"), [(0.0, 0.05), (1.0, 0.5)])
def test_bf16_gather_gradient_matches_fp64_reference(grad_mean: float, min_index_select_error: float) -> None:
    """15k BF16 token gradients summed into 3 rows stay at BF16 output precision.

    ``index_select``'s backward adds each token into the BF16 row with an
    atomic add, rounding after every add; coherent gradients stall once the
    row sum outgrows the addends. The assertion on its error documents that.
    """
    torch.manual_seed(1)
    device = torch.device("cuda")
    sequence_length, hidden_size = 15360, 1024
    indices = _layout_indices(sequence_length, 3, shuffled=False).to(device)
    grad_rows = (grad_mean + torch.randn(sequence_length, hidden_size, device=device)).to(torch.bfloat16)
    expected = torch.zeros(3, hidden_size, device=device, dtype=torch.float64)
    expected.index_add_(0, indices, grad_rows.double())

    table = torch.zeros(3, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    _gather_modulation_rows(table, indices).backward(grad_rows)
    index_select_table = torch.zeros_like(table, requires_grad=True)
    index_select_table.index_select(0, indices).backward(grad_rows)

    assert table.grad.dtype == torch.bfloat16
    # BF16 epsilon; the measured error is ~3e-3 against ~2e-3 for rounding the exact sums.
    assert _relative_error(table.grad, expected) < 2**-7
    assert _relative_error(index_select_table.grad, expected) > min_index_select_error


def test_gather_without_grad_is_plain_index_select(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference and frozen tables keep the plain index_select graph."""

    def _fail(*args):
        raise AssertionError("custom gather ran without a table gradient")

    monkeypatch.setattr(minimax_h3._ModulationRowGather, "apply", _fail)
    table = torch.randn(3, 8, requires_grad=True)
    indices = torch.tensor([0, 2, 1, 1, 2, 0])
    expected = table.detach().index_select(0, indices)

    with torch.no_grad():
        rows = _gather_modulation_rows(table, indices)
    assert rows.grad_fn is None
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)

    frozen_rows = _gather_modulation_rows(table.detach(), indices)
    assert frozen_rows.grad_fn is None
    torch.testing.assert_close(frozen_rows, expected, rtol=0, atol=0)
