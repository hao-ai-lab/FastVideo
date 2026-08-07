"""VSA-256 FA4 CuTe forward/backward parity for BHSD and BSHD APIs."""

from __future__ import annotations

import pytest
import torch

from fastvideo_kernel import video_sparse_attn, video_sparse_attn_bshd

from .test_vsa256_triton import _metrics, _torch_vsa256_reference


@pytest.fixture(autouse=True)
def _require_cute_backend(monkeypatch):
    pytest.importorskip(
        "flash_attn.cute.block_sparsity",
        reason="optional FA4 CuTe build (flash_attn.cute) not installed",
    )
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON", raising=False)
    monkeypatch.delenv("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", raising=False)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_forward_backward_vs_torch_ref(layout: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    bsz, heads, dim = 1, 2, 128
    q_blocks_256, kv_blocks_256 = 3, 4
    q_block = 256
    kv_block = 256
    topk_logical = 2
    sq = q_blocks_256 * q_block
    skv = kv_blocks_256 * kv_block

    q_base = torch.randn(bsz, heads, sq, dim, device=device, dtype=dtype)
    k_base = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    v_base = torch.randn(bsz, heads, skv, dim, device=device, dtype=dtype)
    grad_out = torch.randn_like(q_base)

    q_var = torch.full(
        (q_blocks_256,), q_block, dtype=torch.int32, device=device
    )
    kv_var = torch.tensor(
        [256, 173, 79, 256], dtype=torch.int32, device=device
    )
    token_idx = torch.arange(kv_block, device=device, dtype=torch.int32)
    kv_valid = token_idx.view(1, -1) < kv_var.view(-1, 1)
    kv_valid = kv_valid.view(1, 1, kv_blocks_256, kv_block, 1)
    kv_valid = kv_valid.expand(
        bsz, heads, kv_blocks_256, kv_block, dim
    ).reshape(bsz, heads, skv, dim)
    k_base = k_base * kv_valid.to(k_base.dtype)
    v_base = v_base * kv_valid.to(v_base.dtype)

    if layout == "bhsd":
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        out = video_sparse_attn(
            q,
            k,
            v,
            kv_var,
            q_var,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=None,
        )
        (out * grad_out).sum().backward()
        out_bhsd = out
        dq, dk, dv = q.grad, k.grad, v.grad
    else:
        q = q_base.transpose(1, 2).contiguous().requires_grad_(True)
        k = k_base.transpose(1, 2).contiguous().requires_grad_(True)
        v = v_base.transpose(1, 2).contiguous().requires_grad_(True)
        grad_out_bshd = grad_out.transpose(1, 2).contiguous()
        out = video_sparse_attn_bshd(
            q,
            k,
            v,
            kv_var,
            q_var,
            topk_logical,
            block_size=(4, 8, 8),
            compress_attn_weight=None,
        )
        (out * grad_out_bshd).sum().backward()
        out_bhsd = out.transpose(1, 2)
        dq = q.grad.transpose(1, 2)
        dk = k.grad.transpose(1, 2)
        dv = v.grad.transpose(1, 2)

    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)
    out_ref = _torch_vsa256_reference(
        q_ref, k_ref, v_ref, q_var, kv_var, topk_logical
    )
    (out_ref * grad_out).sum().backward()

    tensors = (
        out_bhsd,
        dq,
        dk,
        dv,
        q_ref.grad,
        k_ref.grad,
        v_ref.grad,
    )
    assert all(torch.isfinite(t).all().item() for t in tensors)

    m_out = _metrics(out_ref, out_bhsd)
    m_dq = _metrics(q_ref.grad, dq)
    m_dk = _metrics(k_ref.grad, dk)
    m_dv = _metrics(v_ref.grad, dv)
    print(
        f"[vsa256-cute-{layout}] "
        f"out(avg_abs={m_out[0]:.6e}, max_rel={m_out[1]:.6e}), "
        f"dq(avg_abs={m_dq[0]:.6e}, max_rel={m_dq[1]:.6e}), "
        f"dk(avg_abs={m_dk[0]:.6e}, max_rel={m_dk[1]:.6e}), "
        f"dv(avg_abs={m_dv[0]:.6e}, max_rel={m_dv[1]:.6e})"
    )

    assert m_out[0] < 1e-3 and m_out[1] < 0.2
    assert m_dq[0] < 2e-2 and m_dq[1] < 0.5
    assert m_dk[0] < 2e-2 and m_dk[1] < 0.5
    assert m_dv[0] < 2e-2 and m_dv[1] < 0.5
