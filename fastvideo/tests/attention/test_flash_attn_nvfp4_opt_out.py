# SPDX-License-Identifier: Apache-2.0
"""NVFP4 FA4 opt-in/opt-out precedence for ``FlashAttentionImpl``.

The process-wide ``FASTVIDEO_NVFP4_FA4=1`` env opt-in targets the DiT. An
explicit ``nvfp4_fa4`` impl arg must win over the env in both directions so
precision-sensitive layers (e.g. the FP32-pinned MiniMax-H3 VAE attention)
can force-disable FP4 Q/K quantization while the DiT keeps it.
"""

from unittest.mock import patch

import pytest

# The backend module needs a flash-attention package (FA2/FA3, or FA4 with
# FASTVIDEO_FA4=1) at import. exc_type=ImportError also covers partial
# installs (e.g. an FA4-only environment without the opt-in env set raises
# ImportError rather than ModuleNotFoundError).
flash_attn_module = pytest.importorskip(
    "fastvideo.attention.backends.flash_attn",
    reason="no usable flash-attention package installed",
    exc_type=ImportError,
)
FlashAttentionImpl = flash_attn_module.FlashAttentionImpl


def _build_impl(**extra_impl_args) -> FlashAttentionImpl:
    return FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        causal=False,
        softmax_scale=0.125,
        num_kv_heads=2,
        **extra_impl_args,
    )


@pytest.fixture
def fa4_runtime_available():
    """Satisfy the Blackwell + flash-attention-fp4 asserts without hardware."""
    with (
        patch.object(flash_attn_module, "_FA4_FP4_AVAILABLE", True),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        yield


def test_nvfp4_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_NVFP4_FA4", raising=False)
    assert _build_impl().nvfp4_fa4 is False


def test_nvfp4_env_opt_in_enables_when_arg_absent(monkeypatch, fa4_runtime_available) -> None:
    monkeypatch.setenv("FASTVIDEO_NVFP4_FA4", "1")
    assert _build_impl().nvfp4_fa4 is True


def test_explicit_disable_wins_over_env_opt_in(monkeypatch) -> None:
    """The H3 VAE constructs its impl with ``nvfp4_fa4=False``; the DiT env
    opt-in must not FP4-quantize Q/K inside the FP32-pinned VAE."""
    monkeypatch.setenv("FASTVIDEO_NVFP4_FA4", "1")
    assert _build_impl(nvfp4_fa4=False).nvfp4_fa4 is False


def test_explicit_enable_works_without_env(monkeypatch, fa4_runtime_available) -> None:
    monkeypatch.delenv("FASTVIDEO_NVFP4_FA4", raising=False)
    assert _build_impl(nvfp4_fa4=True).nvfp4_fa4 is True
