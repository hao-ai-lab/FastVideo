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
    monkeypatch.setenv("FASTVIDEO_FLASHINFER_PREFILL_BACKEND", "single")
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


def test_cudnn_forward_flattens_batch_and_passes_token_offsets(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_FLASHINFER_PREFILL_BACKEND", "cudnn")
    FlashInferImpl._cudnn_workspaces.clear()
    calls = []

    def fake_cudnn(q, k, v, scale, workspace, **kwargs):
        calls.append((q.shape, k.shape, v.shape, scale, workspace.shape, kwargs))
        return q + 1, None

    prefill = types.ModuleType("flashinfer.prefill")
    prefill.cudnn_batch_prefill_with_kv_cache = fake_cudnn
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.prefill = prefill
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)
    monkeypatch.setattr(torch, "empty", lambda size, **kwargs: torch.zeros(size, dtype=kwargs["dtype"]))

    impl = FlashInferImpl(num_heads=4, head_size=128, causal=False, softmax_scale=0.125)
    query = torch.randn(2, 3, 4, 128, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = impl.forward(query, key, value, FlashInferMetadata(current_timestep=0))

    assert output.shape == query.shape
    assert calls[0][0] == (6, 4, 128)
    assert calls[0][1] == (10, 2, 128)
    assert calls[0][5]["batch_offsets_units"] == "tokens"
    torch.testing.assert_close(calls[0][5]["batch_offsets_q"], torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(calls[0][5]["batch_offsets_k"], torch.tensor([0, 5, 10], dtype=torch.int32))


def test_cudnn_rejects_custom_mask_and_unsupported_head_size(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_FLASHINFER_PREFILL_BACKEND", "cudnn")
    with pytest.raises(ValueError, match="head size 128"):
        FlashInferImpl(num_heads=4, head_size=64, causal=False, softmax_scale=0.125)

    impl = FlashInferImpl(num_heads=4, head_size=128, causal=False, softmax_scale=0.125)
    query = torch.randn(1, 3, 4, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="arbitrary attention masks"):
        impl.forward(query, query, query,
                     FlashInferMetadata(current_timestep=0, attn_mask=torch.ones(1, 3, dtype=torch.bool)))


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


def test_metadata_builder_instantiates_through_backend_api() -> None:
    from fastvideo.attention.backends.abstract import AttentionMetadataBuilder
    from fastvideo.attention.backends.flashinfer import FlashInferBackend
    builder = FlashInferBackend.get_builder_cls()()
    assert isinstance(builder, AttentionMetadataBuilder)
    builder.prepare()
    mask = torch.ones(2, 3, dtype=torch.bool)
    metadata = builder.build(current_timestep=7, attn_mask=mask)
    assert metadata.current_timestep == 7
    assert metadata.attn_mask is mask


@pytest.mark.parametrize("shared_head", [False, True])
@pytest.mark.parametrize("additive", [False, True])
def test_shared_masks_preserve_each_sample(shared_head, additive) -> None:
    mask = torch.tensor([[[True, False, True], [False, True, True]],
                         [[False, True, False], [True, True, False]]])
    source = torch.where(mask, 0.0, float("-inf")) if additive else mask
    if shared_head:
        source = source[:, None]
    for sample in range(2):
        torch.testing.assert_close(_mask_for_sample(source, sample, 2, 3), mask[sample])


def test_per_head_mask_rejected_before_kernel(monkeypatch) -> None:
    prefill = types.ModuleType("flashinfer.prefill")
    def unexpected_kernel(*args, **kwargs):
        pytest.fail("per-head mask reached FlashInfer kernel")
    prefill.single_prefill_with_kv_cache = unexpected_kernel
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)
    query = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    metadata = FlashInferMetadata(current_timestep=0, attn_mask=torch.ones(2, 4, 3, 3, dtype=torch.bool))
    impl = FlashInferImpl(num_heads=4, head_size=64, causal=False, softmax_scale=0.125)
    with pytest.raises(ValueError, match="per-head"):
        impl.forward(query, query, query, metadata)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shared_head", [False, True])
def test_shared_mask_cuda_matches_flash_attention(dtype, shared_head) -> None:
    _require_flashinfer_cuda()
    flash_attn = pytest.importorskip("flash_attn")
    torch.manual_seed(19)
    q = torch.randn(2, 16, 2, 64, device="cuda", dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    mask = torch.zeros(2, 16, 16, device="cuda", dtype=torch.bool)
    mask[0, :, :11] = True
    mask[1, :, :7] = True
    metadata = FlashInferMetadata(current_timestep=0, attn_mask=mask[:, None] if shared_head else mask)
    impl = FlashInferImpl(num_heads=2, head_size=64, causal=False, softmax_scale=0.2)
    actual = impl.forward(q, k, v, metadata)
    expected = torch.cat([
        flash_attn.flash_attn_func(q[i:i + 1], k[i:i + 1, :length], v[i:i + 1, :length], softmax_scale=0.2)
        for i, length in enumerate([11, 7])
    ])
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("i2v", [False, True])
def test_wan_attention_executes_flashinfer(monkeypatch, request, dtype, i2v) -> None:
    _require_flashinfer_cuda()
    request.getfixturevalue("distributed_setup")
    import flashinfer.prefill as prefill
    import fastvideo.envs as envs
    from flash_attn import flash_attn_func
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.wan.config import WanVideoConfig
    from fastvideo.models.wan.transformer import WanTransformerBlock
    from fastvideo.platforms import AttentionBackendEnum

    real_kernel = prefill.single_prefill_with_kv_cache
    calls = []
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    def checked_kernel(q, k, v, **kwargs):
        actual = real_kernel(q, k, v, **kwargs)
        expected = flash_attn_func(q[None], k[None], v[None],
                                  softmax_scale=kwargs["sm_scale"], causal=kwargs["causal"])[0]
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        calls.append((q.shape, k.shape))
        return actual
    monkeypatch.setattr(prefill, "single_prefill_with_kv_cache", checked_kernel)
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("FLASHINFER"), torch.device("cuda"):
        block = WanTransformerBlock(
            dim=256, ffn_dim=512, num_heads=2, cross_attn_norm=True,
            added_kv_proj_dim=256 if i2v else None,
            supported_attention_backends=WanVideoConfig()._supported_attention_backends,
        ).to(dtype=dtype).eval()
    assert block.attn1.backend == AttentionBackendEnum.FLASHINFER
    assert block.attn2.attn.backend == AttentionBackendEnum.FLASHINFER
    for param in block.parameters():
        torch.nn.init.normal_(param, std=0.02)
    q = torch.randn(1, 16, 2, 128, device="cuda", dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    x = torch.randn(1, 16, 256, device="cuda", dtype=dtype)
    context = torch.randn(1, 264 if i2v else 7, 256, device="cuda", dtype=dtype)
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        self_output, _ = block.attn1(q, k, v)
        cross_output = block.attn2(x, context, None)
    assert self_output.shape == q.shape
    assert cross_output.shape == x.shape
    assert len(calls) == (3 if i2v else 2)


def test_wan_cuda_skip_precedes_distributed_setup(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class Request:
        def getfixturevalue(self, name):
            pytest.fail("distributed setup ran before the CUDA availability check")
    with pytest.raises(pytest.skip.Exception, match="NVIDIA CUDA GPU"):
        test_wan_attention_executes_flashinfer(monkeypatch, Request(), torch.float16, False)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_flashinfer_real_cudnn_kernel_matches_sdpa(monkeypatch, batch_size: int) -> None:
    """Launch FlashInfer's dedicated batched cuDNN SDPA entry point."""
    _require_flashinfer_cuda()
    pytest.importorskip("flashinfer.prefill").cudnn_batch_prefill_with_kv_cache
    monkeypatch.setenv("FASTVIDEO_FLASHINFER_PREFILL_BACKEND", "cudnn")
    torch.manual_seed(2)
    device = torch.device("cuda", torch.cuda.current_device())
    query = torch.randn(batch_size, 128, 4, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 192, 2, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    scale = 128**-0.5
    impl = FlashInferImpl(num_heads=4, num_kv_heads=2, head_size=128, causal=False, softmax_scale=scale)

    actual = impl.forward(query, key, value, FlashInferMetadata(current_timestep=0))
    expected = _sdpa_reference(query, key, value, scale=scale)

    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
