from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from fastvideo.attention.backends.flashinfer import FlashInferImpl, FlashInferMetadata, _mask_for_sample


def test_padding_mask_is_front_padded_and_expanded() -> None:
    mask = torch.tensor([[1, 0]], dtype=torch.int64)
    actual = _mask_for_sample(mask, sample=0, query_len=3, key_len=4)
    expected = torch.tensor([[1, 1, 1, 0]] * 3, dtype=torch.bool)
    torch.testing.assert_close(actual, expected)


def test_forward_preserves_bshd_contract_and_arguments(monkeypatch) -> None:
    calls = []

    def fake_kernel(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, v.shape, kwargs))
        return q + 1

    prefill = types.ModuleType("flashinfer.prefill")
    prefill.single_prefill_with_kv_cache = fake_kernel
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.prefill = prefill
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)

    impl = FlashInferImpl(num_heads=4, head_size=64, causal=True, softmax_scale=0.125)
    query = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 2, 64, dtype=torch.bfloat16)
    value = torch.randn(2, 5, 2, 64, dtype=torch.bfloat16)
    metadata = FlashInferMetadata(current_timestep=0, attn_mask=torch.ones(2, 5, dtype=torch.bool))

    output = impl.forward(query, key, value, metadata)

    assert output.shape == query.shape
    assert len(calls) == 2
    assert calls[0][0] == (3, 4, 64)
    assert calls[0][1] == (5, 2, 64)
    assert calls[0][3]["kv_layout"] == "NHD"
    assert calls[0][3]["causal"] is False
    assert calls[0][3]["custom_mask"].shape == (3, 5)
    assert calls[0][3]["sm_scale"] == 0.125


def _require_flashinfer_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires one NVIDIA CUDA GPU")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("FlashInfer attention requires sm80 or newer")
    pytest.importorskip("flashinfer.prefill", reason="flashinfer-python is not installed")


def _sdpa_reference(query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    *,
                    scale: float,
                    causal: bool = False,
                    key_mask: torch.Tensor | None = None) -> torch.Tensor:
    attn_mask = key_mask[:, None, None, :] if key_mask is not None else None
    output = F.scaled_dot_product_attention(query.transpose(1, 2),
                                            key.transpose(1, 2),
                                            value.transpose(1, 2),
                                            attn_mask=attn_mask,
                                            is_causal=causal,
                                            scale=scale,
                                            enable_gqa=query.shape[2] != key.shape[2])
    return output.transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_flashinfer_real_cuda_kernel_matches_sdpa(dtype: torch.dtype, head_size: int) -> None:
    """Launch the real single-GPU FlashInfer kernel for every supported head size."""
    _require_flashinfer_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    scale = head_size**-0.5
    query = torch.randn(1, 128, 4, head_size, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    impl = FlashInferImpl(num_heads=4, head_size=head_size, causal=False, softmax_scale=scale)

    actual = impl.forward(query, key, value, FlashInferMetadata(current_timestep=0))
    expected = _sdpa_reference(query, key, value, scale=scale)

    torch.cuda.synchronize(device)
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("mode", ["causal", "cross_gqa", "causal_padding"])
def test_flashinfer_real_cuda_kernel_attention_modes(mode: str) -> None:
    """Exercise native causal, GQA/cross-attention, and combined custom masks."""
    _require_flashinfer_cuda()
    torch.manual_seed(1)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    head_size = 128
    query_len, key_len = (64, 96) if mode == "cross_gqa" else (128, 128)
    query_heads, kv_heads = (4, 2) if mode == "cross_gqa" else (4, 4)
    scale = head_size**-0.5
    query = torch.randn(1, query_len, query_heads, head_size, device=device, dtype=dtype)
    key = torch.randn(1, key_len, kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn_like(key)
    causal = mode in ("causal", "causal_padding")
    key_mask = None
    if mode == "causal_padding":
        key_mask = torch.ones(1, key_len, device=device, dtype=torch.bool)
        key_mask[:, -8:] = False
    metadata = FlashInferMetadata(current_timestep=0, attn_mask=key_mask)
    impl = FlashInferImpl(num_heads=query_heads,
                          num_kv_heads=kv_heads,
                          head_size=head_size,
                          causal=causal,
                          softmax_scale=scale)

    actual = impl.forward(query, key, value, metadata)
    if causal and key_mask is not None:
        full_mask = key_mask[:, None, :].expand(-1, query_len, -1)
        causal_mask = torch.ones((query_len, key_len), dtype=torch.bool,
                                 device=device).tril(key_len - query_len)
        full_mask = full_mask & causal_mask
        expected = F.scaled_dot_product_attention(query.transpose(1, 2),
                                                  key.transpose(1, 2),
                                                  value.transpose(1, 2),
                                                  attn_mask=full_mask[:, None, :, :],
                                                  scale=scale).transpose(1, 2)
    else:
        expected = _sdpa_reference(query, key, value, scale=scale, causal=causal, key_mask=key_mask)

    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
