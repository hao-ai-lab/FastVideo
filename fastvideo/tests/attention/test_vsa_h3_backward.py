# SPDX-License-Identifier: Apache-2.0
"""GPU backward checks for the VSA-H3 backend.

The CuTe backend returns FA4's own output tensor, which FA4's autograd node
saved for its backward. Composing the compression branch onto it in place
therefore poisons the graph, and the failure only appears once the VSA-256
CuTe path has a backward at all. These tests pin the composition.
"""

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder)

_SPEC = dict(raw_latent_shape=(16, 16, 24), patch_size=(1, 2, 2), prefix_segments=(64, 32, 16))
_HEADS = 2
_DIM = 128


def _build_meta(device, sparsity=0.5):
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=_SPEC["raw_latent_shape"],
        patch_size=_SPEC["patch_size"],
        VSA_sparsity=sparsity,
        prefix_segments=_SPEC["prefix_segments"],
        device=device,
    )


def _select_backend(monkeypatch, backend):
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


def _forward_backward(impl, meta, gate_compress, device):
    seq = meta.total_seq_length
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, seq, _HEADS, _DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))

    gate = None
    if gate_compress:
        gate = torch.randn(1, tq.shape[1], _HEADS, _DIM, device=device, dtype=torch.bfloat16) * 0.1

    out = impl.forward(tq, tk, tv, gate, meta)
    out = impl.postprocess_output(out, meta)
    out.float().pow(2).sum().backward()
    return out, (q, k, v)


@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("gate_compress", [False, True])
def test_h3_vsa_backward_runs(monkeypatch, backend: str, gate_compress: bool) -> None:
    """Regression: with the CuTe backend and a non-zero gate this used to die
    with "one of the variables needed for gradient computation has been
    modified by an inplace operation ... output 0 of FlashAttnFuncBackward".
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _select_backend(monkeypatch, backend)

    device = torch.device("cuda")
    meta = _build_meta(device)
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)

    out, leaves = _forward_backward(impl, meta, gate_compress, device)

    assert torch.isfinite(out).all().item()
    for name, leaf in zip(("q", "k", "v"), leaves):
        assert leaf.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(leaf.grad).all().item(), f"{name}.grad has non-finite values"
        assert leaf.grad.abs().sum().item() > 0, f"{name}.grad is all zero"


@pytest.mark.parametrize("gate_compress", [False, True])
def test_h3_vsa_backward_cute_matches_triton(monkeypatch, gate_compress: bool) -> None:
    """CuTe and Triton take different routes to the same math; their gradients
    should agree to bf16 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)

    grads = {}
    for backend in ("triton", "cute"):
        with monkeypatch.context() as m:
            _select_backend(m, backend)
            meta = _build_meta(device)
            _, leaves = _forward_backward(impl, meta, gate_compress, device)
            grads[backend] = [leaf.grad.detach().float() for leaf in leaves]

    for name, ref, got in zip(("dq", "dk", "dv"), grads["triton"], grads["cute"]):
        diff = (ref - got).abs()
        avg_abs = diff.mean().item()
        max_rel = (diff.max() / (ref.abs().mean() + 1e-6)).item()
        print(f"[h3-vsa gate={gate_compress}] {name}: avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
        assert avg_abs < 1e-2, f"{name}: avg_abs {avg_abs:.3e}"
        assert max_rel < 0.5, f"{name}: max_rel {max_rel:.3e}"
