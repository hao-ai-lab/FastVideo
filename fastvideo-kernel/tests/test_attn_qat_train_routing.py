# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from fastvideo_kernel.triton_kernels import attn_qat_train as kernel


def _production_route_kwargs():
    return {
        "device": torch.device("cuda"),
        "head_dim": 128,
        "causal": False,
        "is_qat": True,
        "fake_quant_p": True,
        "two_level_quant_p": False,
        "use_global_sf_p": False,
    }


def test_sm100_production_configuration_uses_optimized_route(monkeypatch):
    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: True)
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED", raising=False)

    assert kernel._use_sm100_optimized_qat(**_production_route_kwargs())


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("head_dim", 64),
        ("causal", True),
        ("is_qat", False),
        ("fake_quant_p", False),
        ("two_level_quant_p", True),
        ("use_global_sf_p", True),
    ],
)
def test_unsupported_configuration_keeps_legacy_route(monkeypatch, override, value):
    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: True)
    kwargs = _production_route_kwargs()
    kwargs[override] = value

    assert not kernel._use_sm100_optimized_qat(**kwargs)


def test_non_sm100_and_debug_switch_keep_legacy_route(monkeypatch):
    kwargs = _production_route_kwargs()
    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: False)
    assert not kernel._use_sm100_optimized_qat(**kwargs)

    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: True)
    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED", "0")
    assert not kernel._use_sm100_optimized_qat(**kwargs)


def test_exact_m_is_opt_in(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", raising=False)
    assert not kernel._sm100_exact_m_enabled()

    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", "1")
    assert kernel._sm100_exact_m_enabled()


def test_sm120_joined_pv_is_enabled_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", raising=False)
    assert kernel._consumer_blackwell_join_qat_pv_enabled()

    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", "0")
    assert not kernel._consumer_blackwell_join_qat_pv_enabled()


@pytest.mark.parametrize(
    ("n_ctx", "mode", "expected"),
    [
        (2_048, "fast", (32, 32, 4, 5)),
        (4_096, "fast", (128, 128, 8, 3)),
        (4_096, "balanced", (64, 32, 4, 4)),
        (4_096, "reference", (32, 32, 4, 5)),
        (31_200, "reference", (32, 32, 4, 4)),
    ],
)
def test_sm100_forward_config_selection(n_ctx, mode, expected):
    assert kernel._select_sm100_forward_config(n_ctx, n_ctx, mode) == expected


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 parity test",
)
@pytest.mark.parametrize(("q_length", "kv_length"), [(2_112, 2_112), (2_112, 2_080)])
def test_sm100_optimized_forward_backward_matches_legacy(monkeypatch, q_length, kv_length):
    torch.manual_seed(7)
    q_shape = (1, 1, q_length, 128)
    kv_shape = (1, 1, kv_length, 128)
    inputs = [
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
    ]
    grad_out = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    flags = (
        True,  # use_qat_qkv_backward
        False,  # smooth_k
        True,  # warp_specialize (disabled internally on Blackwell)
        True,  # IS_QAT
        False,  # two_level_quant_P
        True,  # fake_quant_P
        True,  # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_P
        False,  # use_global_sf_QKV
    )

    def run(optimized: bool):
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED", "1" if optimized else "0")
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_MODE", "fast")
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", "1")
        q, k, v = [tensor.clone().requires_grad_(True) for tensor in inputs]
        output = kernel.attention(
            q,
            k,
            v,
            False,
            1.0 / math.sqrt(q_shape[-1]),
            *flags,
        )
        output.backward(grad_out)
        return output.detach(), q.grad, k.grad, v.grad

    legacy = run(False)
    optimized = run(True)

    assert (optimized[0].float() - legacy[0].float()).abs().max().item() <= 1e-2
    assert (optimized[1].float() - legacy[1].float()).abs().max().item() <= 4e-3
    assert (optimized[2].float() - legacy[2].float()).abs().max().item() <= 4e-3
    assert torch.equal(optimized[3], legacy[3])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 parity test",
)
@pytest.mark.parametrize(("q_length", "kv_length"), [(2_112, 2_112), (2_112, 2_080)])
def test_sm120_joined_pv_forward_backward_matches_split_path(monkeypatch, q_length, kv_length):
    torch.manual_seed(11)
    q_shape = (1, 1, q_length, 128)
    kv_shape = (1, 1, kv_length, 128)
    inputs = [
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
    ]
    grad_out = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    flags = (
        True,  # use_qat_qkv_backward
        False,  # smooth_k
        True,  # warp_specialize (disabled internally on Blackwell)
        True,  # IS_QAT
        False,  # two_level_quant_P
        True,  # fake_quant_P
        True,  # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_P
        False,  # use_global_sf_QKV
    )

    def run(joined_pv: bool):
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", "1" if joined_pv else "0")
        q, k, v = [tensor.clone().requires_grad_(True) for tensor in inputs]
        output = kernel.attention(
            q,
            k,
            v,
            False,
            1.0 / math.sqrt(q_shape[-1]),
            *flags,
        )
        output.backward(grad_out)
        return output.detach(), q.grad, k.grad, v.grad

    split = run(False)
    joined = run(True)

    assert torch.equal(joined[0], split[0])
    assert torch.equal(joined[1], split[1])
    assert torch.equal(joined[2], split[2])
    assert torch.equal(joined[3], split[3])
