# SPDX-License-Identifier: Apache-2.0
"""CPU checks for independent documents with optional masked capacity rows."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastvideo.attention.backends.abstract import PackedAttentionMetadata, PackedSequenceMetadata
from fastvideo.attention.backends.sdpa import SDPAImpl


def _metadata(
    lengths: tuple[int, ...],
    *,
    pad_tokens: int = 0,
    key_value_lengths: tuple[int, ...] | None = None,
) -> PackedAttentionMetadata:
    return PackedAttentionMetadata(
        current_timestep=0,
        packed_sequence=PackedSequenceMetadata.from_lengths(
            lengths, device="cpu", key_value_lengths=key_value_lengths, pad_tokens=pad_tokens),
    )


def _reference_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: tuple[int, ...],
    *,
    scale: float,
) -> torch.Tensor:
    outputs = []
    start = 0
    for length in lengths:
        end = start + length
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, start:end].transpose(1, 2),
                key[:, start:end].transpose(1, 2),
                value[:, start:end].transpose(1, 2),
                scale=scale,
            ).transpose(1, 2))
        start = end
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("pad_tokens", [0, 1, 246])
def test_packed_sequence_metadata_builds_cumulative_lengths(pad_tokens: int) -> None:
    metadata = PackedSequenceMetadata.from_lengths((3, 5, 2), device="cpu", pad_tokens=pad_tokens)

    assert metadata.pad_tokens == pad_tokens

    assert metadata.query_lengths == (3, 5, 2)
    assert metadata.key_value_lengths == (3, 5, 2)
    assert metadata.cu_seqlens_q.tolist() == [0, 3, 8, 10]
    assert metadata.cu_seqlens_kv.tolist() == [0, 3, 8, 10]
    assert metadata.max_seqlen_q == 5
    assert metadata.max_seqlen_kv == 5
    assert metadata.num_documents == 3
    assert metadata.total_query_tokens == 10
    assert metadata.total_key_value_tokens == 10


@pytest.mark.parametrize("bad_lengths", [(), (3, 0), (3, -1)])
def test_packed_sequence_metadata_rejects_empty_or_nonpositive_documents(
    bad_lengths: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="at least one document|positive"):
        PackedSequenceMetadata.from_lengths(bad_lengths, device="cpu")


@pytest.mark.parametrize("pad_tokens", [-1, 1.5, True])
def test_packed_sequence_metadata_rejects_invalid_padding(pad_tokens) -> None:
    with pytest.raises(ValueError, match="pad_tokens.*non-negative integer"):
        PackedSequenceMetadata.from_lengths((3,), device="cpu", pad_tokens=pad_tokens)


def test_packed_sdpa_matches_independent_documents_forward_and_backward() -> None:
    torch.manual_seed(2026)
    lengths = (3, 5, 2)
    scale = 0.25
    impl = SDPAImpl(num_heads=2, head_size=4, causal=False, softmax_scale=scale)
    inputs = [torch.randn(1, sum(lengths), 2, 4, requires_grad=True) for _ in range(3)]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    actual = impl.forward(*inputs, _metadata(lengths))
    expected = _reference_sdpa(*reference_inputs, lengths, scale=scale)
    torch.testing.assert_close(actual, expected)

    output_gradient = torch.randn_like(actual)
    actual.backward(output_gradient)
    expected.backward(output_gradient)
    for tensor, reference_tensor in zip(inputs, reference_inputs, strict=True):
        torch.testing.assert_close(tensor.grad, reference_tensor.grad)


def test_packed_sdpa_prevents_cross_document_influence() -> None:
    torch.manual_seed(17)
    lengths = (4, 3)
    impl = SDPAImpl(num_heads=2, head_size=4, causal=False, softmax_scale=0.5)
    query, key, value = (torch.randn(1, sum(lengths), 2, 4) for _ in range(3))
    metadata = _metadata(lengths)

    baseline = impl.forward(query, key, value, metadata)
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, lengths[0]:] = torch.randn_like(changed_key[:, lengths[0]:]) * 10_000
    changed_value[:, lengths[0]:] = torch.randn_like(changed_value[:, lengths[0]:]) * 10_000
    changed = impl.forward(query, changed_key, changed_value, metadata)

    torch.testing.assert_close(changed[:, :lengths[0]], baseline[:, :lengths[0]], rtol=0, atol=0)
    assert not torch.equal(changed[:, lengths[0]:], baseline[:, lengths[0]:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("query_lengths", "key_lengths", "num_heads", "num_kv_heads"),
    [((7,), (7,), 2, 2), ((3, 5, 2), (3, 5, 2), 2, 2), ((3, 2), (5, 1), 4, 2)],
)
def test_capacity_padded_sdpa_matches_real_tokens_and_gradients(
    dtype, query_lengths, key_lengths, num_heads, num_kv_heads,
) -> None:
    """Single-doc PDD, multi-doc DMD, and unequal Q/KV+GQA share the contract."""
    torch.manual_seed(29)
    pad_tokens = 13
    impl = SDPAImpl(num_heads=num_heads, num_kv_heads=num_kv_heads, head_size=8, causal=False, softmax_scale=0.25)
    real_lengths = (sum(query_lengths), sum(key_lengths), sum(key_lengths))
    heads = (num_heads, num_kv_heads, num_kv_heads)
    real_inputs = [torch.randn(1, length, count, 8, dtype=dtype, requires_grad=True)
                   for length, count in zip(real_lengths, heads, strict=True)]
    padded_inputs = [torch.cat((tensor.detach(), torch.randn(1, pad_tokens, tensor.shape[2], 8, dtype=dtype) * 1000),
                               dim=1).requires_grad_() for tensor in real_inputs]
    expected = impl.forward(*real_inputs, _metadata(query_lengths, key_value_lengths=key_lengths))
    actual = impl.forward(
        *padded_inputs, _metadata(query_lengths, key_value_lengths=key_lengths, pad_tokens=pad_tokens))

    assert actual.shape == (1, real_lengths[0] + pad_tokens, num_heads, 8)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual[:, :real_lengths[0]], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, real_lengths[0]:]) == 0

    # Nonzero upstream gradients on pad outputs must not reach Q, K, or V.
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient[:, :real_lengths[0]].contiguous())
    for real, padded, length in zip(real_inputs, padded_inputs, real_lengths, strict=True):
        assert real.grad is not None and padded.grad is not None
        torch.testing.assert_close(padded.grad[:, :length], real.grad, rtol=0, atol=0)
        assert torch.count_nonzero(padded.grad[:, length:]) == 0


@pytest.mark.parametrize("bad_tensor", [0, 1, 2])
def test_packed_sdpa_rejects_inconsistent_capacity_length(bad_tensor: int) -> None:
    tensors = [torch.randn(1, 12, 2, 4) for _ in range(3)]
    tensors[bad_tensor] = tensors[bad_tensor][:, :-1]
    message = "query length" if bad_tensor == 0 else "key/value lengths"
    with pytest.raises(ValueError, match=message):
        SDPAImpl(num_heads=2, head_size=4, causal=False, softmax_scale=0.5).forward(
            *tensors, _metadata((3, 4), pad_tokens=5))


def test_packed_sdpa_rejects_inconsistent_physical_length() -> None:
    query, key, value = (torch.randn(1, 6, 2, 4) for _ in range(3))
    with pytest.raises(ValueError, match="query length"):
        SDPAImpl(num_heads=2, head_size=4, causal=False, softmax_scale=0.5).forward(
            query,
            key,
            value,
            _metadata((3, 4)),
        )
