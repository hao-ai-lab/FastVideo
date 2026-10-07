"""VSA-256 FA4 CuTe forward/backward parity for BHSD and BSHD APIs.

Covers the shapes the CuTe backward actually sees in production: the gated
compression branch (`compress_attn_weight`), partially filled Q tiles,
and q_len != kv_len. Also pins the inference fast path, which must skip the
KV-owned backward metadata without changing the forward result.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from fastvideo_kernel import video_sparse_attn, video_sparse_attn_bshd

from .test_vsa256_triton import _metrics, _torch_vsa256_reference

_BLOCK = 256
_BLOCK_SIZE_3D = (4, 8, 8)  # prod == 256

# Measured on GB200 (sm_100) with bf16 inputs: grads land around 1e-4 avg_abs
# and <=0.11 max_rel across every case below, so these leave ~10x headroom
# without being loose enough to hide a real regression.
_OUT_TOL = (1e-3, 0.2)
_GRAD_TOL = (1e-3, 0.25)


@pytest.fixture(autouse=True)
def _require_cute_backend(monkeypatch):
    pytest.importorskip(
        "flash_attn.cute.block_sparsity",
        reason="optional FA4 CuTe build (flash_attn.cute) not installed",
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON", raising=False)
    monkeypatch.delenv("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", raising=False)


def _zero_pad_tail(x: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Zero the padded tail of every 256-token tile of a [B, H, S, D] tensor.

    VSA callers scatter into a zeroed tile buffer, so padded slots are zero;
    both the kernel and the reference rely on that.
    """
    bsz, heads, _, dim = x.shape
    blocks = var.numel()
    token_idx = torch.arange(_BLOCK, device=x.device, dtype=torch.int32)
    valid = (token_idx.view(1, -1) < var.view(-1, 1)).view(1, 1, blocks, _BLOCK, 1)
    valid = valid.expand(bsz, heads, blocks, _BLOCK, dim).reshape_as(x)
    return x * valid.to(x.dtype)


def _make_inputs(
    q_blocks: int,
    kv_blocks: int,
    kv_var: torch.Tensor,
    q_var: torch.Tensor,
    heads: int = 2,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sq, skv = q_blocks * _BLOCK, kv_blocks * _BLOCK
    q = torch.randn(1, heads, sq, dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, skv, dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, skv, dim, device=device, dtype=dtype)
    grad_out = torch.randn_like(q)
    return _zero_pad_tail(q, q_var), _zero_pad_tail(k, kv_var), _zero_pad_tail(v, kv_var), grad_out


def _check(tag: str, ref: torch.Tensor, got: torch.Tensor, tol: Tuple[float, float]) -> None:
    assert torch.isfinite(got).all().item(), f"{tag}: non-finite values"
    avg_abs, max_rel = _metrics(ref, got)
    print(f"  {tag}: avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < tol[0], f"{tag}: avg_abs {avg_abs:.3e} >= {tol[0]:.3e}"
    assert max_rel < tol[1], f"{tag}: max_rel {max_rel:.3e} >= {tol[1]:.3e}"


def _run_bhsd(q, k, v, kv_var, q_var, topk, gate=None):
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = video_sparse_attn(qg, kg, vg, kv_var, q_var, topk, block_size=_BLOCK_SIZE_3D, compress_attn_weight=gate)
    return out, (qg, kg, vg)


def _run_bshd(q, k, v, kv_var, q_var, topk, gate=None):
    qg, kg, vg = (t.transpose(1, 2).contiguous().requires_grad_(True) for t in (q, k, v))
    gate_bshd = None if gate is None else gate.transpose(1, 2).contiguous()
    out = video_sparse_attn_bshd(qg,
                                 kg,
                                 vg,
                                 kv_var,
                                 q_var,
                                 topk,
                                 block_size=_BLOCK_SIZE_3D,
                                 compress_attn_weight=gate_bshd)
    return out.transpose(1, 2), (qg, kg, vg)


def _reference(q, k, v, q_var, kv_var, topk, gate=None):
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = _torch_vsa256_reference(qr, kr, vr, q_var, kv_var, topk, compress_attn_weight=gate)
    return out, (qr, kr, vr)


def _compare(tag, layout, q, k, v, kv_var, q_var, topk, grad_out, gate=None):
    runner = _run_bhsd if layout == "bhsd" else _run_bshd
    out, (qg, kg, vg) = runner(q, k, v, kv_var, q_var, topk, gate=gate)
    (out * grad_out).sum().backward()
    grads = [g.grad if g.grad.dim() == 4 and layout == "bhsd" else g.grad for g in (qg, kg, vg)]
    if layout == "bshd":
        grads = [g.transpose(1, 2) for g in grads]

    out_ref, refs = _reference(q, k, v, q_var, kv_var, topk, gate=gate)
    (out_ref * grad_out).sum().backward()

    print(f"[{tag}-{layout}]")
    _check("out", out_ref, out, _OUT_TOL)
    for name, ref, got in zip(("dq", "dk", "dv"), refs, grads):
        _check(name, ref.grad, got, _GRAD_TOL)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_forward_backward_vs_torch_ref(layout: str) -> None:
    kv_var = torch.tensor([256, 173, 79, 256], dtype=torch.int32, device="cuda")
    q_var = torch.full((3, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(3, 4, kv_var, q_var)
    _compare("vsa256-cute", layout, q, k, v, kv_var, q_var, 2, grad_out)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_backward_with_compress_gate(layout: str) -> None:
    """The gated compression branch is what Wan and MiniMax-H3 actually run.

    It is also the branch that composes the sparse output with the compression
    output, so it is the one that breaks if that composition mutates FA4's
    saved output in place.
    """
    kv_var = torch.tensor([256, 200, 256, 91], dtype=torch.int32, device="cuda")
    q_var = torch.full((3, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(3, 4, kv_var, q_var, seed=7)
    gate = torch.randn_like(q) * 0.1
    _compare("vsa256-cute-gated", layout, q, k, v, kv_var, q_var, 2, grad_out, gate=gate)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_backward_partial_q_blocks(layout: str) -> None:
    """Q tiles that are not full: only the compression divisor depends on it,
    but it is the one axis the existing coverage held constant."""
    kv_var = torch.tensor([256, 128, 256], dtype=torch.int32, device="cuda")
    q_var = torch.tensor([256, 61, 199], dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(3, 3, kv_var, q_var, seed=11)
    _compare("vsa256-cute-partial-q", layout, q, k, v, kv_var, q_var, 2, grad_out)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_backward_cross_q_kv(layout: str) -> None:
    """q_len != kv_len: forward has coverage, backward did not."""
    kv_var = torch.tensor([256, 143, 256, 256, 88], dtype=torch.int32, device="cuda")
    q_var = torch.full((2, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(2, 5, kv_var, q_var, seed=13)
    _compare("vsa256-cute-cross", layout, q, k, v, kv_var, q_var, 3, grad_out)


@pytest.mark.cuda
def test_vsa256_cute_inference_matches_training_forward() -> None:
    """The KV-owned backward metadata is only built when something requires
    grad. Skipping it must not perturb the forward result."""
    kv_var = torch.tensor([256, 173, 79, 256], dtype=torch.int32, device="cuda")
    q_var = torch.full((3, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, _ = _make_inputs(3, 4, kv_var, q_var, seed=5)

    with torch.no_grad():
        out_infer = video_sparse_attn_bshd(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            kv_var,
            q_var,
            2,
            block_size=_BLOCK_SIZE_3D,
            compress_attn_weight=None,
        )

    out_train, _ = _run_bshd(q, k, v, kv_var, q_var, 2)
    torch.testing.assert_close(out_infer, out_train.transpose(1, 2).detach(), rtol=0, atol=0)


@pytest.mark.cuda
def test_vsa256_cute_lse_is_bhs() -> None:
    """The aux return is [B, H, S] on both entrypoints, matching the Triton
    path's contract."""
    from fastvideo_kernel.block_sparse_attn_256 import (block_sparse_attn_256, block_sparse_attn_256_bshd)

    device = torch.device("cuda")
    heads, dim, q_blocks, kv_blocks = 2, 128, 3, 4
    sq, skv = q_blocks * _BLOCK, kv_blocks * _BLOCK
    q = torch.randn(1, heads, sq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, heads, skv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, heads, skv, dim, device=device, dtype=torch.bfloat16)
    vbs = torch.full((kv_blocks, ), _BLOCK, dtype=torch.int32, device=device)
    mask = torch.zeros(1, heads, q_blocks, kv_blocks, dtype=torch.bool, device=device)
    mask[..., :2] = True

    _, lse_bhsd = block_sparse_attn_256(q, k, v, mask, vbs)
    assert lse_bhsd.shape == (1, heads, sq), lse_bhsd.shape

    _, lse_bshd = block_sparse_attn_256_bshd(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        mask,
        vbs,
    )
    assert lse_bshd.shape == (1, heads, sq), lse_bshd.shape
