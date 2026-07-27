# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the per-arch ATTN_QAT_INFER kernel resolution.

The capability sets route:
  * sm_120a/sm_121a -> fastvideo-kernel CUTLASS extension,
  * sm_100a/sm_103a -> FP4 FA4 (flash-attention-fp4, #1221 plumbing),
  * anything else  -> unavailable (honest FlashAttention fallback upstream).

All device/import probes are monkeypatched; no GPU or kernel install needed.
"""
from __future__ import annotations

import pytest

import fastvideo.attention.backends.attn_qat_infer as aqi


def _patch(monkeypatch, *, cap, cutlass, fa4) -> None:
    monkeypatch.setattr(aqi, "_active_capability", lambda: cap)
    monkeypatch.setattr(aqi, "_get_attn_qat_infer", lambda: (lambda *a, **k: None) if cutlass else None)
    monkeypatch.setattr(aqi, "_fa4_fp4_available", lambda: fa4)


@pytest.mark.parametrize(
    "cap,cutlass,fa4,expected_kernel",
    [
        ((12, 0), True, False, "cutlass_sm12x"),
        ((12, 1), True, False, "cutlass_sm12x"),
        ((12, 0), False, False, None),  # ext not built
        ((10, 0), False, True, "fa4_fp4"),  # GB200
        ((10, 3), False, True, "fa4_fp4"),  # GB300
        ((10, 0), False, False, None),  # fork not installed
        ((9, 0), False, True, None),  # H100: FA4-FP4 set excludes it
        ((8, 9), True, True, None),  # Ada: neither set
        (None, True, True, None),  # no CUDA
    ],
)
def test_arch_resolution(monkeypatch, cap, cutlass, fa4, expected_kernel) -> None:
    _patch(monkeypatch, cap=cap, cutlass=cutlass, fa4=fa4)
    assert aqi._resolved_kernel() == expected_kernel
    assert aqi.is_attn_qat_infer_available() == (expected_kernel is not None)


def test_receipt_records_fa4_quant_knobs(monkeypatch) -> None:
    _patch(monkeypatch, cap=(10, 0), cutlass=False, fa4=True)
    monkeypatch.setattr(aqi, "_configured_fa4_pv_mode", "bf16")
    receipt = aqi.attn_qat_infer_receipt()
    assert "arch=sm_100" in receipt
    assert "qk_mode=nvfp4(per-16-e4m3-sf)" in receipt
    assert "pv_mode=bf16" in receipt
    assert "train_sim_mismatch=measured" in receipt


def test_receipt_pv_mode_is_derived_from_configured_knob(monkeypatch) -> None:
    """The receipt's pv_mode field derives from the fa4_pv_mode knob recorded
    at impl construction, not a literal."""
    _patch(monkeypatch, cap=(10, 0), cutlass=False, fa4=True)
    monkeypatch.setattr(aqi, "_configured_fa4_pv_mode", "bf16")
    monkeypatch.setattr(aqi, "_receipt_logged", False)

    aqi.AttnQatInferImpl(num_heads=1, head_size=128, causal=False, softmax_scale=128**-0.5, fa4_pv_mode="fp8")
    assert "pv_mode=fp8" in aqi.attn_qat_infer_receipt()


@pytest.mark.parametrize("impl_module", ["attn_qat_infer", "flash_attn"])
def test_fa4_pv_mode_typo_fails_at_construction(monkeypatch, impl_module) -> None:
    """Both knob consumers validate fa4_pv_mode eagerly: a bad value raises at
    impl construction (CPU-only), never surviving to a GPU forward."""
    monkeypatch.setattr(aqi, "_configured_fa4_pv_mode", aqi._configured_fa4_pv_mode)
    if impl_module == "attn_qat_infer":
        impl_cls = aqi.AttnQatInferImpl
    else:
        from fastvideo.attention.backends.flash_attn import FlashAttentionImpl
        impl_cls = FlashAttentionImpl

    with pytest.raises(ValueError, match="fa4_pv_mode"):
        impl_cls(num_heads=1, head_size=128, causal=False, softmax_scale=128**-0.5, fa4_pv_mode="fp16")


def test_receipt_records_cutlass_scheme(monkeypatch) -> None:
    _patch(monkeypatch, cap=(12, 0), cutlass=True, fa4=False)
    receipt = aqi.attn_qat_infer_receipt()
    assert "arch=sm_120" in receipt
    assert "cutlass" in receipt


def test_receipt_names_install_hint_on_uninstalled_fa4_arch(monkeypatch) -> None:
    _patch(monkeypatch, cap=(10, 3), cutlass=False, fa4=False)
    receipt = aqi.attn_qat_infer_receipt()
    assert "arch=sm_103" in receipt
    assert "flash-attention-fp4" in receipt


def test_unsupported_arch_receipt_lists_support_matrix(monkeypatch) -> None:
    _patch(monkeypatch, cap=(9, 0), cutlass=True, fa4=True)
    receipt = aqi.attn_qat_infer_receipt()
    assert "kernel=none" in receipt
    assert "sm_100a/sm_103a" in receipt


def test_unsupported_arch_forward_never_calls_bundled_extension(monkeypatch) -> None:
    """An unsupported GPU (e.g. sm_90) carrying an importable sm_12x wheel
    must fail cleanly at forward — never dispatch into the wrong binary."""
    calls = []
    monkeypatch.setattr(aqi, "_active_capability", lambda: (9, 0))
    monkeypatch.setattr(aqi, "_get_attn_qat_infer", lambda: (lambda *a, **k: calls.append(1)))
    monkeypatch.setattr(aqi, "_fa4_fp4_available", lambda: False)

    import torch

    impl = aqi.AttnQatInferImpl(num_heads=1, head_size=128, causal=False, softmax_scale=128**-0.5)
    q = torch.zeros(1, 4, 1, 128)
    with pytest.raises(ImportError, match="not available"):
        impl.forward(q, q, q, attn_metadata=None)
    assert not calls, "unsupported arch dispatched into the bundled CUTLASS extension"


def test_fa4_route_resolution_runs_once_across_forwards(monkeypatch) -> None:
    """The FA4 kernel/quantizer resolution is lazy but memoized: across N
    forward calls the slow resolution path (imports + toolchain probe)
    executes exactly once — per-forward re-resolution graph-breaks dynamo
    every step and blocks fullgraph compilation."""
    import torch

    _patch(monkeypatch, cap=(10, 0), cutlass=False, fa4=True)

    resolves = []

    def fake_quant(t):
        return t, torch.zeros(1)

    def fake_cast(t):
        return t

    def fake_kernel(q, k, v, sfq, sfk, softmax_scale=None, causal=False):
        return v

    monkeypatch.setattr(aqi, "_import_fa4_route_ops",
                        lambda: (resolves.append(1) or (fake_quant, fake_cast, fake_kernel)))
    monkeypatch.setattr(aqi, "_FA4_ROUTE_OPS", None)

    impl = aqi.AttnQatInferImpl(num_heads=1, head_size=128, causal=False, softmax_scale=128**-0.5)
    q = torch.zeros(1, 4, 1, 128)
    for _ in range(3):
        out = impl.forward(q, q, q, attn_metadata=None)
    assert out.shape == q.shape
    assert len(resolves) == 1, f"resolution ran {len(resolves)}x across 3 forwards"


def _register_fa4_quantize_cpu_kernel():
    """Register a CPU kernel for fastvideo::nvfp4_quantize_fa4 that honors the
    production output contract — including STRIDES: the real sf output is a
    permuted view of a contiguous (batch, nheads, rest_m, rest_k, 32, 4, 4)
    buffer, and torch.compile bakes output strides into generated code."""
    import torch

    def _cpu_kernel(tensor_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, nheads, headdim = tensor_4d.shape
        seqlen_padded = (seqlen + 127) // 128 * 128
        fp4 = torch.zeros(batch, seqlen_padded, nheads, headdim // 2,
                          dtype=torch.int8).view(torch.float4_e2m1fn_x2)
        rest_m, rest_k = seqlen_padded // 128, (headdim // 16) // 4
        sf = torch.zeros(batch, nheads, rest_m, rest_k, 32, 4, 4,
                         dtype=torch.uint8).permute(4, 5, 2, 6, 3, 1, 0)
        return fp4, sf

    try:
        torch.library.register_kernel("fastvideo::nvfp4_quantize_fa4", "cpu")(_cpu_kernel)
    except RuntimeError:
        pass  # already registered by a previous test/run


def test_fa4_quantize_path_is_fullgraph_traceable() -> None:
    """The FP4 quantize step must be a single opaque graph node: tracing its
    python body descends into flashinfer's toolchain probe (a subprocess),
    which torch.compile(fullgraph=True) rejects — a runtime memo cannot fix
    that because cache hits are invisible at trace time. fullgraph=True over
    the op-backed path must compile and run without a graph break."""
    import torch

    from fastvideo.attention.backends import flash_attn as fa

    _register_fa4_quantize_cpu_kernel()

    def path(x: torch.Tensor) -> torch.Tensor:
        fp4, sf = fa._nvfp4_quantize_for_fa4(x)
        return fp4[:, :x.shape[1]].view(torch.int8).float() + float(sf.shape[0])

    compiled = torch.compile(path, fullgraph=True, backend="eager")
    out = compiled(torch.randn(1, 64, 2, 128, dtype=torch.bfloat16))
    assert out.shape == (1, 64, 2, 64)


def test_fa4_quantize_op_fake_matches_real() -> None:
    """torch.library.opcheck cross-checks the registered fake against a real
    kernel run — shapes, dtypes, AND strides — so a fake whose output layout
    drifts from the impl (e.g. contiguous fake vs permuted-view real sf, which
    torch.compile turns into a runtime stride assertion) fails here on CPU.

    The op is forward-only (no autograd registration), so restrict opcheck to
    the non-autograd suites."""
    import torch

    import fastvideo.attention.backends.flash_attn  # noqa: F401  registers the op + fake

    _register_fa4_quantize_cpu_kernel()

    # batch>1 and seqlen>128 so every sf dim (incl. batch, rest_m) is
    # non-trivial and its stride actually participates in the check.
    x = torch.randn(2, 200, 2, 128, dtype=torch.bfloat16)
    torch.library.opcheck(
        torch.ops.fastvideo.nvfp4_quantize_fa4,
        (x,),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def _register_fa4_v_to_fp8_cpu_kernel():
    """Register a CPU kernel for fastvideo::fa4_v_to_fp8. Unlike the quantize
    op this needs no mock: the production impl is a plain dtype cast, which
    CPU torch executes natively, so the CPU kernel IS the production body."""
    import torch

    def _cpu_kernel(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float8_e4m3fn)

    try:
        torch.library.register_kernel("fastvideo::fa4_v_to_fp8", "cpu")(_cpu_kernel)
    except RuntimeError:
        pass  # already registered by a previous test/run


def test_fa4_v_to_fp8_op_fake_matches_real() -> None:
    """torch.library.opcheck for the fp8 V-cast op: the fake must reproduce
    the impl's shape, dtype, AND strides (`.to` preserves the input layout for
    dense tensors; empty_like's preserve_format mirrors that). Forward-only op
    (no autograd registration), so restrict to the non-autograd suites."""
    import torch

    import fastvideo.attention.backends.flash_attn  # noqa: F401  registers the op + fake

    _register_fa4_v_to_fp8_cpu_kernel()

    for v in (
            torch.randn(2, 200, 2, 128, dtype=torch.bfloat16),
            # Non-contiguous dense input: strides must carry through the cast.
            torch.randn(2, 2, 200, 128, dtype=torch.bfloat16).transpose(1, 2),
    ):
        torch.library.opcheck(
            torch.ops.fastvideo.fa4_v_to_fp8,
            (v,),
            test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        )


def test_fa4_fp8_pv_path_is_fullgraph_traceable(monkeypatch) -> None:
    """With fa4_pv_mode="fp8" the production _forward_fa4_fp4 body (quantize
    op + V-cast op + kernel call) must compile fullgraph with no graph break:
    the naive in-forward `.to(torch.float8_e4m3fn)` cast breaks the graph,
    which is exactly why the cast lives behind a custom op."""
    import torch

    from fastvideo.attention.backends import flash_attn as fa

    _patch(monkeypatch, cap=(10, 0), cutlass=False, fa4=True)
    monkeypatch.setattr(aqi, "_configured_fa4_pv_mode", aqi._configured_fa4_pv_mode)
    _register_fa4_quantize_cpu_kernel()
    _register_fa4_v_to_fp8_cpu_kernel()

    def fake_kernel(q, k, v, sfq, sfk, softmax_scale=None, causal=False):
        # The real kernel emits BF16 with full headdim regardless of V dtype.
        return v.to(torch.bfloat16)

    # Pre-resolve the route with the real op-backed helpers and a mocked
    # kernel entry point (importing the real one needs a CUDA install).
    monkeypatch.setattr(aqi, "_FA4_ROUTE_OPS", (fa._nvfp4_quantize_for_fa4, fa._fa4_v_to_fp8, fake_kernel))

    impl = aqi.AttnQatInferImpl(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5, fa4_pv_mode="fp8")
    compiled = torch.compile(impl._forward_fa4_fp4, fullgraph=True, backend="eager")
    x = torch.randn(1, 64, 2, 128, dtype=torch.bfloat16)
    out = compiled(x, x, x)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


def test_fa4_pv_mode_env_bridge(monkeypatch) -> None:
    """The FASTVIDEO_FA4_PV_MODE env bridge is the user-reachable path to the
    knob (model code constructs attention with fixed literals); explicit
    kwargs still win over the environment."""
    import torch  # noqa: F401

    _patch(monkeypatch, cap=(10, 0), cutlass=False, fa4=True)
    monkeypatch.setenv("FASTVIDEO_FA4_PV_MODE", "fp8")

    impl = aqi.AttnQatInferImpl(num_heads=1, head_size=128, causal=False, softmax_scale=128**-0.5)
    assert impl.fa4_pv_mode == "fp8"

    explicit = aqi.AttnQatInferImpl(num_heads=1, head_size=128, causal=False, softmax_scale=128**-0.5,
                                    fa4_pv_mode="bf16")
    assert explicit.fa4_pv_mode == "bf16"
