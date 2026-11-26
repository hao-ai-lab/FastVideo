# SPDX-License-Identifier: Apache-2.0
"""Deployed NVFP4 FA4 Q/K preconditioning and per-block BF16 opt-out.

The transform must be applied to Q/K right before NVFP4 quantization with the same function the QAD emulator uses,
and MiniMax-H3 blocks listed in ``FASTVIDEO_NVFP4_FA4_BF16_BLOCKS`` must build BF16 attention impls.
"""
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import fastvideo.envs as envs  # noqa: E402
from fastvideo.attention.utils.nvfp4_qk_precondition import (  # noqa: E402
    parse_block_indices, parse_qk_transform, precondition_qk,
)

flash_attn_module = pytest.importorskip(
    "fastvideo.attention.backends.flash_attn",
    reason="no usable flash-attention package installed",
    exc_type=ImportError,
)
FlashAttentionImpl = flash_attn_module.FlashAttentionImpl


def _impl(**extra) -> "FlashAttentionImpl":
    return FlashAttentionImpl(num_heads=2, head_size=64, causal=False, softmax_scale=0.125, num_kv_heads=2, **extra)


@pytest.fixture
def fa4_runtime_available():
    with (
        patch.object(flash_attn_module, "_FA4_FP4_AVAILABLE", True),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        yield


def test_block_list_parsing() -> None:
    assert parse_block_indices("") == frozenset()
    assert parse_block_indices(None) == frozenset()
    assert parse_block_indices(" 45, 49,45 ") == frozenset({45, 49})
    for bad in ("a,1", "-1", "4.5"):
        with pytest.raises(ValueError):
            parse_block_indices(bad)


def test_transform_env_default_and_validation(env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4_QK_TRANSFORM.override(None))
    assert envs.FASTVIDEO_NVFP4_FA4_QK_TRANSFORM.get() == "none"
    assert _impl().nvfp4_smooth_k is False and _impl().nvfp4_hadamard is False
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4_QK_TRANSFORM.override("Smooth_K+Hadamard"))
    impl = _impl()
    assert (impl.nvfp4_smooth_k, impl.nvfp4_hadamard) == (True, True)
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4_QK_TRANSFORM.override("bogus"))
    with pytest.raises(envs.EnvVarError):
        envs.FASTVIDEO_NVFP4_FA4_QK_TRANSFORM.get()


def test_impl_arg_wins_over_transform_env(env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4_QK_TRANSFORM.override("hadamard"))
    impl = _impl(nvfp4_qk_transform="smooth_k")
    assert (impl.nvfp4_smooth_k, impl.nvfp4_hadamard) == (True, False)
    with pytest.raises(ValueError):
        _impl(nvfp4_qk_transform="bogus")


@pytest.mark.parametrize("name", ["none", "smooth_k", "hadamard", "smooth_k+hadamard"])
def test_transform_applied_before_quantization(name, env_overrides, fa4_runtime_available) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4.override(None))
    impl = _impl(nvfp4_fa4=True, nvfp4_qk_transform=name)
    torch.manual_seed(0)
    q = torch.randn(1, 20, 2, 64, dtype=torch.bfloat16) + 3.0
    k = torch.randn(1, 20, 2, 64, dtype=torch.bfloat16) + 3.0
    v = torch.randn(1, 20, 2, 64, dtype=torch.bfloat16)
    seen = []

    def fake_quant(tensor):
        seen.append(tensor.clone())
        return tensor, tensor

    fp4 = MagicMock(side_effect=lambda *a, **kw: a[2])
    with (
        patch.object(flash_attn_module, "_nvfp4_quantize_for_fa4", side_effect=fake_quant),
        patch.object(flash_attn_module, "flash_attn_fp4_func", fp4),
    ):
        impl._forward_nvfp4(q, k, v)

    smooth_k, hadamard = parse_qk_transform(name)
    expected_q, expected_k = precondition_qk(q, k, smooth_k=smooth_k, hadamard=hadamard)
    assert torch.equal(seen[0], expected_q) and torch.equal(seen[1], expected_k)
    assert (name == "none") == (torch.equal(seen[0], q) and torch.equal(seen[1], k))


def test_smooth_k_removes_mean_and_preserves_attention() -> None:
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 32, 2, 64) for _ in range(3))
    k = k + 5.0
    qt, kt = precondition_qk(q, k, smooth_k=True, hadamard=True)
    ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
    out = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (qt, kt, v)))
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def _build_h3_attention(bf16_attention: bool):
    h3 = pytest.importorskip("fastvideo.models.dits.minimax_h3")
    backend = MagicMock()
    backend.get_name.return_value = "FLASH_ATTN"
    distributed = MagicMock()
    with (
        patch.object(h3, "get_attn_backend", return_value=backend),
        patch.object(h3, "get_compute_dtype", return_value=torch.bfloat16),
        patch.object(h3, "ReplicatedLinear", side_effect=lambda *a, **kw: torch.nn.Identity()),
        patch.object(h3, "DistributedAttention", distributed),
    ):
        h3.MiniMaxH3Attention(64, 2, 32, 1e-6, (), None, prefix="transformer_blocks.3.attn",
                              bf16_attention=bf16_attention)
    return distributed.call_args.kwargs


def test_h3_attention_bf16_opt_out_wiring() -> None:
    assert _build_h3_attention(True)["nvfp4_fa4"] is False
    assert "nvfp4_fa4" not in _build_h3_attention(False)


def test_bf16_block_impl_ignores_env_opt_in(env_overrides, fa4_runtime_available) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_NVFP4_FA4.override(True))
    assert _impl(nvfp4_fa4=_build_h3_attention(True)["nvfp4_fa4"]).nvfp4_fa4 is False
    assert _impl(**{k: v for k, v in _build_h3_attention(False).items() if k == "nvfp4_fa4"}).nvfp4_fa4 is True  # env wins
