# SPDX-License-Identifier: Apache-2.0
"""Focused routing and FA4 integration checks for MiniMax-H3 fusions."""

from __future__ import annotations

import pytest
import torch

from fastvideo.platforms import AttentionBackendEnum


@pytest.mark.parametrize("enabled", [False, True])
def test_only_opted_in_packed_minimax_h3_blocks_enable_fa4_varlen(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    import fastvideo.models.dits.minimax_h3 as h3

    routed: list[bool] = []

    class RecordingAttention(torch.nn.Module):

        def __init__(self, *args, fa4_packed_varlen: bool = False, **kwargs) -> None:
            super().__init__()
            del args, kwargs
            routed.append(fa4_packed_varlen)

    monkeypatch.setattr(h3, "MiniMaxH3Attention", RecordingAttention)
    common = dict(
        hidden_size=8,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_dim=16,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, ),
        quant_config=None,
    )
    h3.MiniMaxH3TransformerBlock(
        **common,
        time_embed_dim=4,
        prefix="minimax_h3.transformer_blocks.0",
        fa4_packed_varlen=enabled,
    )
    h3.MiniMaxH3TokenRefinerBlock(
        **common,
        prefix="minimax_h3.token_refiner.refiner_blocks.0",
    )

    assert routed == [enabled, False]


@pytest.mark.parametrize("fa4_packed_varlen", [False, True])
def test_minimax_h3_varlen_flag_reaches_resolved_backend_impl(
    monkeypatch: pytest.MonkeyPatch,
    fa4_packed_varlen: bool,
) -> None:
    """Cover the H3-attention -> distributed-attention -> impl boundary."""
    import fastvideo.attention.layer as attention_layer
    import fastvideo.models.dits.minimax_h3 as h3

    init_kwargs: dict = {}

    class RecordingAttentionImpl:

        def __init__(self, **kwargs) -> None:
            init_kwargs.update(kwargs)

    class RecordingAttentionBackend:

        @staticmethod
        def get_name() -> str:
            return "FLASH_ATTN"

        @staticmethod
        def get_impl_cls() -> type[RecordingAttentionImpl]:
            return RecordingAttentionImpl

    def resolve_backend(*args, **kwargs):
        del args, kwargs
        return RecordingAttentionBackend

    monkeypatch.setattr(h3, "get_attn_backend", resolve_backend)
    monkeypatch.setattr(attention_layer, "get_attn_backend", resolve_backend)
    h3.MiniMaxH3Attention(
        hidden_size=8,
        num_attention_heads=1,
        attention_head_dim=8,
        qk_norm_eps=1e-5,
        supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, ),
        quant_config=None,
        prefix="minimax_h3.transformer_blocks.0.attn",
        fa4_packed_varlen=fa4_packed_varlen,
    )

    assert init_kwargs["fa4_packed_varlen"] is fa4_packed_varlen


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", frozenset()),
        ("0", frozenset()),
        ("none", frozenset()),
        ("all", frozenset({"modulate", "qknorm_rope", "swiglu"})),
        ("1", frozenset({"modulate", "qknorm_rope", "swiglu"})),
        ("swiglu, modulate", frozenset({"swiglu", "modulate"})),
    ],
)
def test_minimax_h3_fusion_selector(raw: str, expected: frozenset[str]) -> None:
    from fastvideo.models.dits.minimax_h3 import _enabled_minimax_h3_fusions

    assert _enabled_minimax_h3_fusions(raw) == expected


def test_minimax_h3_fusion_selector_rejects_unknown_name() -> None:
    from fastvideo.models.dits.minimax_h3 import _enabled_minimax_h3_fusions

    with pytest.raises(ValueError, match="Unknown MiniMax H3 fusion"):
        _enabled_minimax_h3_fusions("swiglu,unknown")


def test_swiglu_fusion_stays_on_eager_path_with_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.models.dits.minimax_h3 as h3

    def unexpected_kernel(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("inference-only fusion ran with grad enabled")

    monkeypatch.setattr(h3, "minimax_h3_swiglu", unexpected_kernel)
    layer = h3.MiniMaxH3FeedForward(8, 16, fuse_swiglu=True)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    layer(inputs).sum().backward()

    assert inputs.grad is not None


def test_all_minimax_h3_fusions_match_one_eager_block_under_fa4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real block wiring without loading any H3 checkpoint."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA is required")
    pytest.importorskip("triton")
    flash_attn = pytest.importorskip("flash_attn")
    if "fa4" not in getattr(flash_attn, "__version__", "").lower():
        pytest.skip("the focused integration test requires the FA4 environment")

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29573")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3RotaryPosEmbed, MiniMaxH3TransformerBlock

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        kwargs = dict(
            hidden_size=128,
            num_attention_heads=1,
            attention_head_dim=128,
            ffn_dim=256,
            time_embed_dim=64,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN, ),
            quant_config=None,
            prefix="minimax_h3.test_block",
        )
        previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            eager = MiniMaxH3TransformerBlock(**kwargs)
            fused = MiniMaxH3TransformerBlock(
                **kwargs,
                fuse_modulate=True,
                fuse_qknorm_rope=True,
                fuse_swiglu=True,
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)

        with torch.no_grad():
            for name, parameter in eager.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1.0)
                elif parameter.ndim > 1:
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
                else:
                    parameter.zero_()
        fused.load_state_dict(eager.state_dict(), strict=True)

        device = torch.device("cuda")
        eager = eager.to(device=device, dtype=torch.bfloat16).eval()
        fused = fused.to(device=device, dtype=torch.bfloat16).eval()
        generator = torch.Generator(device=device).manual_seed(2026)
        hidden_states = torch.randn(2, 12, 128, generator=generator, device=device, dtype=torch.bfloat16)
        temb = torch.randn(2, 64, generator=generator, device=device, dtype=torch.bfloat16)
        adaln_indices = torch.arange(12, device=device, dtype=torch.long).remainder(6)
        position_ids = torch.zeros(12, 3, device=device, dtype=torch.float32)
        position_ids[:, 0] = torch.arange(12, device=device)
        rotary_emb = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0).to(device)(position_ids)
        inputs = dict(
            hidden_states=hidden_states,
            temb=temb,
            adaln_indices=adaln_indices,
            rotary_emb=tuple(value.to(torch.bfloat16) for value in rotary_emb),
            original_seq_len=12,
        )

        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            eager_output = eager(**inputs)
            fused_output = fused(**inputs)

        # Sol-Engine keeps fused intermediates in FP32 registers until their
        # final BF16 stores, so the opt-in path is close but not bit-identical.
        torch.testing.assert_close(fused_output, eager_output, atol=3e-2, rtol=3e-2)
    finally:
        cleanup_dist_env_and_memory()


def test_minimax_h3_fusions_engage_on_cuda_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the positive side of the routing guard.

    The parity test above still passes if ``_can_run_minimax_h3_fusion``
    silently degrades to always-False (both blocks then run the identical
    eager path), so count the fused-kernel calls: one CUDA inference forward
    through a fully fused block must hit ``fused_rmsnorm_modulate`` once,
    ``fused_residual_gate_rmsnorm_modulate`` once, ``fused_qknorm_rope``
    twice (q and k), and ``minimax_h3_swiglu`` once -- and a grad-enabled
    forward must leave every counter unchanged.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA is required")
    pytest.importorskip("triton")

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29574")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    import fastvideo.models.dits.minimax_h3 as h3
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.forward_context import set_forward_context

    calls = dict.fromkeys(("rmsnorm_modulate", "residual_gate_rmsnorm_modulate", "qknorm_rope", "swiglu"), 0)

    def _counting(name: str, real):

        def wrapper(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(h3, "fused_rmsnorm_modulate", _counting("rmsnorm_modulate", h3.fused_rmsnorm_modulate))
    monkeypatch.setattr(h3, "fused_residual_gate_rmsnorm_modulate",
                        _counting("residual_gate_rmsnorm_modulate", h3.fused_residual_gate_rmsnorm_modulate))
    monkeypatch.setattr(h3, "fused_qknorm_rope", _counting("qknorm_rope", h3.fused_qknorm_rope))
    monkeypatch.setattr(h3, "minimax_h3_swiglu", _counting("swiglu", h3.minimax_h3_swiglu))

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        block = h3.MiniMaxH3TransformerBlock(
            hidden_size=128,
            num_attention_heads=1,
            attention_head_dim=128,
            ffn_dim=256,
            time_embed_dim=64,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
            quant_config=None,
            prefix="minimax_h3.engagement_block",
            fuse_modulate=True,
            fuse_qknorm_rope=True,
            fuse_swiglu=True,
        )
        device = torch.device("cuda")
        block = block.to(device=device, dtype=torch.bfloat16).eval()

        generator = torch.Generator(device=device).manual_seed(2026)
        hidden_states = torch.randn(2, 12, 128, generator=generator, device=device, dtype=torch.bfloat16)
        temb = torch.randn(2, 64, generator=generator, device=device, dtype=torch.bfloat16)
        adaln_indices = torch.arange(12, device=device, dtype=torch.long).remainder(6)
        position_ids = torch.zeros(12, 3, device=device, dtype=torch.float32)
        position_ids[:, 0] = torch.arange(12, device=device)
        rotary_emb = h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0).to(device)(position_ids)
        inputs = dict(
            hidden_states=hidden_states,
            temb=temb,
            adaln_indices=adaln_indices,
            rotary_emb=tuple(value.to(torch.bfloat16) for value in rotary_emb),
            original_seq_len=12,
        )

        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            block(**inputs)
        engaged = dict(calls)
        assert engaged == {
            "rmsnorm_modulate": 1,
            "residual_gate_rmsnorm_modulate": 1,
            "qknorm_rope": 2,
            "swiglu": 1,
        }, engaged

        grad_inputs = {**inputs, "hidden_states": hidden_states.clone().requires_grad_(True)}
        with set_forward_context(current_timestep=0, attn_metadata=None):
            block(**grad_inputs)
        assert dict(calls) == engaged, f"a fusion ran under grad: {calls} vs {engaged}"
    finally:
        cleanup_dist_env_and_memory()
