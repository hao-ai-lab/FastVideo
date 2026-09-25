# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 integration coverage for the FlashInfer backend."""

from __future__ import annotations

import pytest
import torch

from fastvideo.platforms import AttentionBackendEnum


def test_minimax_h3_routes_flashinfer_to_distributed_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the H3 layer forwards its backend support to the shared router."""
    import fastvideo.attention.layer as attention_layer
    import fastvideo.models.dits.minimax_h3 as h3

    resolved_backends: list[tuple[AttentionBackendEnum, ...] | None] = []

    class RecordingAttentionImpl:

        def __init__(self, **kwargs) -> None:
            del kwargs

    class RecordingAttentionBackend:

        @staticmethod
        def get_name() -> str:
            return "FLASHINFER"

        @staticmethod
        def get_impl_cls() -> type[RecordingAttentionImpl]:
            return RecordingAttentionImpl

    def resolve_backend(*args, **kwargs):
        del args
        resolved_backends.append(kwargs.get("supported_attention_backends"))
        return RecordingAttentionBackend

    monkeypatch.setattr(h3, "get_attn_backend", resolve_backend)
    monkeypatch.setattr(attention_layer, "get_attn_backend", resolve_backend)

    layer = h3.MiniMaxH3Attention(
        hidden_size=128,
        num_attention_heads=1,
        attention_head_dim=128,
        qk_norm_eps=1e-5,
        supported_attention_backends=(AttentionBackendEnum.FLASHINFER, ),
        quant_config=None,
        prefix="minimax_h3.test_flashinfer",
    )

    assert layer.distributed_attention.backend == AttentionBackendEnum.FLASHINFER
    assert resolved_backends == [
        (AttentionBackendEnum.FLASHINFER, ),
        (AttentionBackendEnum.FLASHINFER, ),
    ]


@pytest.mark.parametrize("prefill_backend", ["single", "cudnn"])
def test_minimax_h3_flashinfer_matches_torch_sdpa(
    monkeypatch: pytest.MonkeyPatch,
    prefill_backend: str,
) -> None:
    """Run the real H3 attention path and compare FlashInfer with Torch SDPA."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA is required")
    pytest.importorskip("flashinfer")

    monkeypatch.setenv("FASTVIDEO_FLASHINFER_PREFILL_BACKEND", prefill_backend)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29575")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                       maybe_init_distributed_environment_and_model_parallel)
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention, MiniMaxH3RotaryPosEmbed

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        common = dict(
            hidden_size=256,
            num_attention_heads=2,
            attention_head_dim=128,
            qk_norm_eps=1e-5,
            quant_config=None,
            prefix="minimax_h3.test_flashinfer",
        )
        previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            reference = MiniMaxH3Attention(
                **common,
                supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
            )
            flashinfer = MiniMaxH3Attention(
                **common,
                supported_attention_backends=(AttentionBackendEnum.FLASHINFER, ),
            )
        finally:
            torch.set_default_dtype(previous_default_dtype)

        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1.0)
                elif parameter.ndim > 1:
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
                else:
                    parameter.zero_()
        flashinfer.load_state_dict(reference.state_dict(), strict=True)

        device = torch.device("cuda")
        reference = reference.to(device=device, dtype=torch.bfloat16).eval()
        flashinfer = flashinfer.to(device=device, dtype=torch.bfloat16).eval()
        generator = torch.Generator(device=device).manual_seed(2026)
        sequence_length = 128
        hidden_states = torch.randn(
            1,
            sequence_length,
            256,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        position_ids = torch.zeros(sequence_length, 3, device=device, dtype=torch.float32)
        position_ids[:, 0] = torch.arange(sequence_length, device=device)
        rotary_emb = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0).to(device)(position_ids)
        rotary_emb = tuple(value.to(torch.bfloat16) for value in rotary_emb)

        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            expected = reference(hidden_states, rotary_emb, sequence_length)
            actual = flashinfer(hidden_states, rotary_emb, sequence_length)

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    finally:
        cleanup_dist_env_and_memory()
