"""VSA-128 CuTe/Triton forward and backward parity on Blackwell."""

from __future__ import annotations

import math

import pytest
import torch

from fastvideo_kernel import video_sparse_attn, video_sparse_attn_bshd
from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_128

from .test_vsa256_triton import _metrics, _torch_vsa256_reference

_BLOCK = 128
_BLOCK_SIZE_3D = (2, 8, 8)


def _select_backend(monkeypatch, backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if backend == "cute":
        pytest.importorskip(
            "flash_attn.cute.block_sparsity",
            reason="optional FA4 CuTe build (flash_attn.cute) not installed",
        )
        monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
        monkeypatch.delenv("FASTVIDEO_VSA_TRITON", raising=False)
        monkeypatch.delenv("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", raising=False)
    else:
        monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
        monkeypatch.delenv("FASTVIDEO_VSA_CUTEDSL", raising=False)


def _dense_sparse_reference(q, k, v, block_map, variable_block_sizes):
    token_mask = block_map.repeat_interleave(_BLOCK, dim=2).repeat_interleave(_BLOCK, dim=3)
    kv_valid = torch.arange(_BLOCK, device=k.device) < variable_block_sizes[:, None]
    token_mask = token_mask & kv_valid.reshape(1, 1, 1, -1)
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    probabilities = torch.softmax(logits.masked_fill(~token_mask, float("-inf")), dim=-1)
    return torch.matmul(probabilities, v.float()).to(q.dtype)


def _check(tag: str, expected: torch.Tensor, actual: torch.Tensor, avg_tol: float, rel_tol: float) -> None:
    assert torch.isfinite(actual).all().item(), f"{tag}: non-finite values"
    avg_abs, max_rel = _metrics(expected, actual)
    print(f"  {tag}: avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < avg_tol
    assert max_rel < rel_tol


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["cute", "triton"])
def test_vsa128_explicit_routes_forward_backward(backend: str, monkeypatch) -> None:
    """Adjacent Q128 blocks must keep independent routes instead of merging."""
    _select_backend(monkeypatch, backend)
    torch.manual_seed(53)
    shape = (1, 1, 3 * _BLOCK, 128)
    base = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    grad_output = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    variable_block_sizes = torch.tensor([128, 91, 37], device="cuda", dtype=torch.int32)
    block_map = torch.eye(3, device="cuda", dtype=torch.bool).view(1, 1, 3, 3)

    actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in base]
    actual, _ = block_sparse_attn_128(*actual_inputs, block_map, variable_block_sizes)
    (actual * grad_output).sum().backward()

    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in base]
    expected = _dense_sparse_reference(*reference_inputs, block_map, variable_block_sizes)
    (expected * grad_output).sum().backward()

    print(f"[vsa128-explicit-{backend}]")
    _check("out", expected, actual, 1e-3, 0.2)
    for name, reference, candidate in zip(("dq", "dk", "dv"), reference_inputs, actual_inputs, strict=True):
        _check(name, reference.grad, candidate.grad, 2e-2, 0.5)


def _zero_kv_tail(x: torch.Tensor, variable_block_sizes: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(_BLOCK, device=x.device) < variable_block_sizes[:, None]
    valid = valid.view(1, 1, -1, _BLOCK, 1).expand_as(x.view(1, x.shape[1], -1, _BLOCK, x.shape[-1]))
    return x * valid.reshape_as(x).to(x.dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["cute", "triton"])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa128_wrapper_forward_backward(backend: str, layout: str, monkeypatch) -> None:
    _select_backend(monkeypatch, backend)
    torch.manual_seed(59)
    batch, heads, dim = 1, 2, 128
    q_blocks, kv_blocks, topk = 3, 4, 2
    q_shape = (batch, heads, q_blocks * _BLOCK, dim)
    kv_shape = (batch, heads, kv_blocks * _BLOCK, dim)
    q_base = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    kv_sizes = torch.tensor([128, 91, 37, 128], device="cuda", dtype=torch.int32)
    q_sizes = torch.full((q_blocks, ), _BLOCK, device="cuda", dtype=torch.int32)
    k_base = _zero_kv_tail(torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16), kv_sizes)
    v_base = _zero_kv_tail(torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16), kv_sizes)
    gate_base = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16) * 0.1
    grad_output = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)

    if layout == "bhsd":
        actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in (q_base, k_base, v_base)]
        actual_gate = gate_base.detach().clone().requires_grad_()
        actual = video_sparse_attn(
            *actual_inputs,
            kv_sizes,
            q_sizes,
            topk,
            block_size=_BLOCK_SIZE_3D,
            compress_attn_weight=actual_gate,
        )
        actual_grads = actual_inputs
    else:
        bshd_inputs = [tensor.transpose(1, 2).contiguous().detach().requires_grad_()
                       for tensor in (q_base, k_base, v_base)]
        bshd_gate = gate_base.transpose(1, 2).contiguous().detach().requires_grad_()
        actual = video_sparse_attn_bshd(
            *bshd_inputs,
            kv_sizes,
            q_sizes,
            topk,
            block_size=_BLOCK_SIZE_3D,
            compress_attn_weight=bshd_gate,
        ).transpose(1, 2)
        actual_grads = bshd_inputs
    (actual * grad_output).sum().backward()

    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in (q_base, k_base, v_base)]
    reference_gate = gate_base.detach().clone().requires_grad_()
    expected = _torch_vsa256_reference(
        *reference_inputs,
        q_sizes,
        kv_sizes,
        topk,
        compress_attn_weight=reference_gate,
    )
    (expected * grad_output).sum().backward()

    print(f"[vsa128-wrapper-{backend}-{layout}]")
    _check("out", expected, actual, 1e-3, 0.2)
    for name, reference, candidate in zip(("dq", "dk", "dv"), reference_inputs, actual_grads, strict=True):
        candidate_grad = candidate.grad if layout == "bhsd" else candidate.grad.transpose(1, 2)
        _check(name, reference.grad, candidate_grad, 2e-2, 0.5)
    actual_gate_grad = actual_gate.grad if layout == "bhsd" else bshd_gate.grad.transpose(1, 2)
    _check("dgate", reference_gate.grad, actual_gate_grad, 1e-3, 0.2)
