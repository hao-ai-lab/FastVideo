# SPDX-License-Identifier: Apache-2.0
"""Explicit fine attention must bypass the native consumer FP4 shortcut."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo import envs
from fastvideo.attention.backends import fp4_vsa_qat, video_sparse_attn_h3
from fastvideo.models.dits import minimax_h3, minimax_h3_vsa_fp4


def test_fine_override_bypasses_native_fp4(monkeypatch):
    class PairIdentity(torch.nn.Module):
        def forward(self, value):
            return value, None

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_impl = SimpleNamespace(fine_attention_override=object())
            self.calls = 0

        def forward(self, q, k, v, **kwargs):
            self.calls += 1
            return v, None

    attn = minimax_h3.MiniMaxH3Attention.__new__(minimax_h3.MiniMaxH3Attention)
    torch.nn.Module.__init__(attn)
    attn._vsa_fp4 = True
    attn._vsa_tile_first = False
    attn.distributed_attention = FakeAttention()
    attn.to_q = attn.to_k = attn.to_v = attn.to_out = PairIdentity()
    attn.num_attention_heads, attn.attention_head_dim = 1, 8
    attn.norm_q = attn.norm_k = torch.nn.Identity()
    attn.fuse_qknorm_rope = False
    attn.to_gate_compress = None
    monkeypatch.setattr(attn, "_rotate", lambda value, rotary: value)
    monkeypatch.setattr(minimax_h3_vsa_fp4, "_shared_input_projections", lambda layers, value: (value, value, value))

    def unexpected_native(*args):
        raise AssertionError("the native route must not run when a fine override exists")

    # The native branch reads metadata before projecting. Neither may be reached.
    monkeypatch.setattr(minimax_h3, "get_forward_context", unexpected_native)
    monkeypatch.setattr(minimax_h3, "vsa_fp4_attention", unexpected_native)
    value = torch.randn(1, 3, 8)
    with torch.no_grad():
        result = attn(value, (torch.ones(3, 8), torch.zeros(3, 8)), 3)
    torch.testing.assert_close(result, value)
    assert attn.distributed_attention.calls == 1


def test_bf16_override_keeps_tile128_without_quantization(monkeypatch, env_overrides):
    numerics = []
    monkeypatch.setattr(fp4_vsa_qat, "FP4VSAFineAttention", lambda config: numerics.append(config) or config)
    video_sparse_attn_h3._fine_attention_from_name.cache_clear()
    try:
        env_overrides.enter_context(envs.FASTVIDEO_H3_VSA_FINE_ATTENTION.override("emulator-bf16"))
        result = video_sparse_attn_h3.fine_attention_from_env()
        assert result.quantize is False
        assert result.prefix_precision is None
        assert len(numerics) == 1
    finally:
        video_sparse_attn_h3._fine_attention_from_name.cache_clear()


@pytest.mark.parametrize("name", ["", "emulator", "mix8", "emulator-", "other-mix8", "other-bf16",
                                "emulator-unknown"])
def test_invalid_fine_attention_names_raise_actionable_value_error(name):
    with pytest.raises(ValueError, match="FASTVIDEO_H3_VSA_FINE_ATTENTION") as error:
        video_sparse_attn_h3._fine_attention_from_name(name)
    for accepted in ("emulator-mix8", "emulator-mixmx", "emulator-qk", "emulator-bf16"):
        assert accepted in str(error.value)


@pytest.mark.parametrize("mode", ["mix8", "mixmx", "qk"])
def test_valid_fine_attention_names_keep_implementation_paths(monkeypatch, mode):
    monkeypatch.setattr(fp4_vsa_qat, "FP4VSAFineAttention", lambda config: ("emulator", config))
    video_sparse_attn_h3._fine_attention_from_name.cache_clear()
    try:
        kind, config = video_sparse_attn_h3._fine_attention_from_name(f"emulator-{mode}")
        assert kind == "emulator"
        assert config.prefix_precision == {"mix8": "fp8", "mixmx": "mxfp8", "qk": None}[mode]
        assert config.quantize and not config.quantize_pv and config.smooth_k
    finally:
        video_sparse_attn_h3._fine_attention_from_name.cache_clear()
