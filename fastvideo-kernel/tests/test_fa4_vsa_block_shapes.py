"""Correctness coverage for FastVideo's fine-grained FA4 sparse tiles."""

from __future__ import annotations

import math

import pytest
import torch


@pytest.fixture(autouse=True)
def _require_fa4_sm100(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        pytest.skip("fine-grained FA4 VSA requires SM100/SM110")
    pytest.importorskip(
        "flash_attn.cute.block_sparsity",
        reason="optional FastVideo FA4 source package is not installed",
    )
    monkeypatch.setenv("FASTVIDEO_FA4_VSA_DUAL_STREAM", "1")
    monkeypatch.setenv("FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER", "1")


def _selected_block_map(
    heads: int,
    q_blocks: int,
    kv_blocks: int,
    selected_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    block_map = torch.zeros(
        1,
        heads,
        q_blocks,
        kv_blocks,
        dtype=torch.bool,
        device=device,
    )
    for head in range(heads):
        for q_block in range(q_blocks):
            start = (3 * head + 5 * q_block) % kv_blocks
            indices = [(start + 2 * offset) % kv_blocks for offset in range(selected_blocks)]
            block_map[0, head, q_block, indices] = True
    return block_map


def _ordered_sparse_tensors(
    block_map: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    masked_blocks: int,
):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    batch, heads, q_blocks, kv_blocks = block_map.shape
    selected_blocks = int(block_map.sum(dim=-1).min().item())
    assert bool((block_map.sum(dim=-1) == selected_blocks).all())
    if not 0 <= masked_blocks <= selected_blocks:
        raise ValueError("masked_blocks must be within the selected-block count")

    selected = torch.arange(kv_blocks, dtype=torch.int32, device=block_map.device)
    selected = selected.view(1, 1, 1, kv_blocks).expand(batch, heads, q_blocks, kv_blocks)
    selected = selected.masked_select(block_map).view(batch, heads, q_blocks, selected_blocks)
    selected = selected.sort(dim=-1).values

    mask_count = masked_blocks
    full_count = selected_blocks - masked_blocks
    mask_idx = torch.zeros(
        batch,
        heads,
        q_blocks,
        max(mask_count, 1),
        dtype=torch.int32,
        device=block_map.device,
    )
    full_idx = torch.zeros(
        batch,
        heads,
        q_blocks,
        max(full_count, 1),
        dtype=torch.int32,
        device=block_map.device,
    )
    if mask_count:
        mask_idx[..., :mask_count] = selected[..., :mask_count]
    if full_count:
        full_idx[..., :full_count] = selected[..., mask_count:]

    return BlockSparseTensorsTorch(
        mask_block_cnt=torch.full(
            (batch, heads, q_blocks),
            mask_count,
            dtype=torch.int32,
            device=block_map.device,
        ),
        mask_block_idx=mask_idx,
        full_block_cnt=torch.full(
            (batch, heads, q_blocks),
            full_count,
            dtype=torch.int32,
            device=block_map.device,
        ),
        full_block_idx=full_idx,
        block_size=(q_block_size, kv_block_size),
    )


def _torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    variable_block_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    token_mask = block_map.repeat_interleave(q_block_size, dim=2)
    token_mask = token_mask.repeat_interleave(kv_block_size, dim=3)
    if variable_block_sizes is not None:
        kv_positions = torch.arange(k.shape[1], device=k.device)
        kv_blocks = kv_positions // kv_block_size
        kv_offsets = kv_positions % kv_block_size
        valid_tokens = kv_offsets < variable_block_sizes[kv_blocks]
        token_mask = token_mask & valid_tokens.view(1, 1, 1, -1)
    q_heads = q.permute(0, 2, 1, 3).float()
    k_heads = k.permute(0, 2, 1, 3).float()
    v_heads = v.permute(0, 2, 1, 3).float()
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~token_mask, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v_heads)
    return out.permute(0, 2, 1, 3).contiguous()


@pytest.mark.cuda
@pytest.mark.parametrize(
    (
        "q_block_size",
        "kv_block_size",
        "selected_blocks",
        "masked_blocks",
        "sp_double_buffer",
    ),
    [
        pytest.param(128, 64, 1, 0, True, id="q128_kv64_one_stream_empty"),
        pytest.param(128, 64, 2, 1, True, id="q128_kv64_even_mixed"),
        pytest.param(128, 64, 3, 1, True, id="q128_kv64_odd_mixed"),
        pytest.param(128, 64, 3, 0, True, id="q128_kv64_odd_full_only"),
        pytest.param(128, 64, 3, 1, False, id="q128_kv64_legacy_optout"),
        pytest.param(64, 64, 3, 1, True, id="q64_kv64_odd_mixed"),
    ],
)
def test_fa4_vsa_fine_grained_forward(
    q_block_size: int,
    kv_block_size: int,
    selected_blocks: int,
    masked_blocks: int,
    sp_double_buffer: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flash_attn.cute.interface import _flash_attn_fwd

    monkeypatch.setenv(
        "FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER",
        "1" if sp_double_buffer else "0",
    )
    torch.manual_seed(2026 + q_block_size + selected_blocks)
    device = torch.device("cuda")
    batch, heads, head_dim = 1, 8, 128
    q_blocks, kv_blocks = 4, 7
    q_len = q_blocks * q_block_size
    kv_len = kv_blocks * kv_block_size
    q = torch.randn(batch, q_len, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, kv_len, heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    block_map = _selected_block_map(heads, q_blocks, kv_blocks, selected_blocks, device)
    sparse_tensors = _ordered_sparse_tensors(
        block_map,
        q_block_size,
        kv_block_size,
        masked_blocks,
    )

    out = _flash_attn_fwd(
        q,
        k,
        v,
        tile_mn=(q_block_size, kv_block_size),
        block_sparse_tensors=sparse_tensors,
        causal=False,
        return_lse=True,
    )[0]
    ref = _torch_reference(q, k, v, block_map, q_block_size, kv_block_size)

    assert bool(torch.isfinite(out).all())
    error = (out.float() - ref).abs()
    avg_abs = float(error.mean())
    max_rel = float(error.max() / (ref.abs().mean() + 1e-6))
    print(
        f"[fa4 {q_block_size}x{kv_block_size} selected={selected_blocks} "
        f"masked={masked_blocks} spdb={sp_double_buffer}] "
        f"avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}"
    )
    assert avg_abs < 1e-3
    assert max_rel < 0.2


@pytest.mark.cuda
@pytest.mark.parametrize("q_block_size", [128, 64], ids=["q128_kv64", "q64_kv64"])
def test_fa4_vsa_dual_stream_persistent_phase_transitions(q_block_size: int) -> None:
    """Exercise even, odd, empty, and one-block tiles in one persistent launch."""
    if q_block_size == 64 and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the Q64/KV64 specialization currently requires SM100")

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    torch.manual_seed(4554 + q_block_size)
    device = torch.device("cuda")
    kv_block_size = 64
    counts = [2, 3, 0, 1, 4]
    masked_counts = [1, 1, 0, 1, 2]
    batch, heads, head_dim, kv_blocks = 1, 4, 128, 8
    q_blocks = len(counts)
    q = torch.randn(
        batch,
        q_blocks * q_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        batch,
        kv_blocks * kv_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)

    mask_count = torch.empty(batch, heads, q_blocks, dtype=torch.int32, device=device)
    full_count = torch.empty_like(mask_count)
    mask_idx = torch.zeros(
        batch,
        heads,
        q_blocks,
        max(masked_counts),
        dtype=torch.int32,
        device=device,
    )
    full_idx = torch.zeros(
        batch,
        heads,
        q_blocks,
        max(count - masked for count, masked in zip(counts, masked_counts)),
        dtype=torch.int32,
        device=device,
    )
    block_map = torch.zeros(
        batch,
        heads,
        q_blocks,
        kv_blocks,
        dtype=torch.bool,
        device=device,
    )
    for head in range(heads):
        for q_block, (count, masked) in enumerate(zip(counts, masked_counts)):
            chosen = sorted({(head + 3 * q_block + 2 * offset) % kv_blocks for offset in range(count)})
            assert len(chosen) == count
            mask_count[0, head, q_block] = masked
            full_count[0, head, q_block] = count - masked
            if masked:
                mask_idx[0, head, q_block, :masked] = torch.tensor(
                    chosen[:masked],
                    dtype=torch.int32,
                    device=device,
                )
            if count > masked:
                full_idx[0, head, q_block, :count - masked] = torch.tensor(
                    chosen[masked:],
                    dtype=torch.int32,
                    device=device,
                )
            if count:
                block_map[0, head, q_block, chosen] = True

    sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=mask_count,
        mask_block_idx=mask_idx,
        full_block_cnt=full_count,
        full_block_idx=full_idx,
        block_size=(q_block_size, kv_block_size),
    )
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        tile_mn=(q_block_size, kv_block_size),
        block_sparse_tensors=sparse_tensors,
        causal=False,
        return_lse=True,
    )[:2]
    ref = _torch_reference(q, k, v, block_map, q_block_size, kv_block_size)

    nonempty_rows = torch.tensor(
        [count > 0 for count in counts],
        dtype=torch.bool,
        device=device,
    ).repeat_interleave(q_block_size)
    error = (out[:, nonempty_rows].float() - ref[:, nonempty_rows]).abs()
    empty_out = out[:, ~nonempty_rows].float()
    empty_lse = lse[:, :, ~nonempty_rows]
    avg_abs = float(error.mean())
    max_abs = float(error.max())
    print(
        f"[fa4 persistent {q_block_size}x{kv_block_size}] "
        f"avg_abs={avg_abs:.6e}, max_abs={max_abs:.6e}"
    )
    assert avg_abs < 1e-3
    assert max_abs < 1e-2
    assert float(empty_out.abs().max()) == 0.0
    assert bool(torch.isneginf(empty_lse).all())


@pytest.mark.cuda
def test_fa4_vsa_q128_kv64_bshd_wrapper_variable_blocks() -> None:
    """Cover partial and empty KV blocks through FastVideo's VBS mask mod."""
    from fastvideo_kernel.block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd_bshd

    torch.manual_seed(5128)
    device = torch.device("cuda")
    batch, heads, head_dim = 1, 8, 128
    q_block_size, kv_block_size = 128, 64
    q_blocks, kv_blocks, selected_blocks = 4, 7, 3
    q = torch.randn(
        batch,
        q_blocks * q_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        batch,
        kv_blocks * kv_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    block_map = _selected_block_map(heads, q_blocks, kv_blocks, selected_blocks, device)
    variable_block_sizes = torch.tensor(
        [64, 23, 64, 7, 0, 51, 64],
        dtype=torch.int32,
        device=device,
    )

    out, lse = block_sparse_attn_cute_fwd_bshd(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
    )
    ref = _torch_reference(
        q,
        k,
        v,
        block_map,
        q_block_size,
        kv_block_size,
        variable_block_sizes,
    )

    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(lse).all())
    error = (out.float() - ref).abs()
    avg_abs = float(error.mean())
    max_rel = float(error.max() / (ref.abs().mean() + 1e-6))
    print(f"[fa4 wrapper Q128/KV64 VBS] avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < 1e-3
    assert max_rel < 0.2


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("q_block_size", "kv_block_size"),
    [
        pytest.param(128, 64, id="q128_kv64"),
        pytest.param(64, 64, id="q64_kv64"),
    ],
)
def test_fa4_vsa_fine_grained_bshd_wrapper(
    q_block_size: int,
    kv_block_size: int,
) -> None:
    """Exercise the FastVideo BSHD entrypoint used by the H3 backend."""
    if q_block_size == 64 and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the Q64/KV64 specialization currently requires SM100")

    from fastvideo_kernel.block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd_bshd

    torch.manual_seed(4048 + q_block_size)
    device = torch.device("cuda")
    batch, heads, head_dim = 1, 8, 128
    q_blocks, kv_blocks, selected_blocks = 4, 7, 3
    q = torch.randn(
        batch,
        q_blocks * q_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        batch,
        kv_blocks * kv_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    block_map = _selected_block_map(heads, q_blocks, kv_blocks, selected_blocks, device)
    variable_block_sizes = torch.full(
        (kv_blocks,),
        kv_block_size,
        dtype=torch.int32,
        device=device,
    )

    out, lse = block_sparse_attn_cute_fwd_bshd(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
    )
    ref = _torch_reference(q, k, v, block_map, q_block_size, kv_block_size)

    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(lse).all())
    error = (out.float() - ref).abs()
    avg_abs = float(error.mean())
    max_rel = float(error.max() / (ref.abs().mean() + 1e-6))
    print(
        f"[fa4 wrapper {q_block_size}x{kv_block_size}] "
        f"avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}"
    )
    assert avg_abs < 1e-3
    assert max_rel < 0.2


@pytest.mark.cuda
@pytest.mark.parametrize("block_shape", ["256x256", "128x64", "64x64"])
def test_vsa256_bshd_fa4_block_shape_routes(
    block_shape: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route one logical Q256/KV256 mask through every FA4 specialization."""
    if block_shape == "64x64" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the Q64/KV64 specialization currently requires SM100")

    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd

    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_FA4_BLOCK_SHAPE", block_shape)
    torch.manual_seed(6256)
    device = torch.device("cuda")
    batch, heads, head_dim = 1, 8, 128
    logical_block_size = 256
    q_blocks, kv_blocks, selected_blocks = 4, 7, 3
    q = torch.randn(
        batch,
        q_blocks * logical_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        batch,
        kv_blocks * logical_block_size,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    block_map = _selected_block_map(heads, q_blocks, kv_blocks, selected_blocks, device)
    variable_block_sizes = torch.tensor(
        [256, 137, 255, 64, 201, 1, 192],
        dtype=torch.int32,
        device=device,
    )

    out, lse = block_sparse_attn_256_bshd(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
    )
    ref = _torch_reference(
        q,
        k,
        v,
        block_map,
        logical_block_size,
        logical_block_size,
        variable_block_sizes,
    )

    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(lse).all())
    error = (out.float() - ref).abs()
    avg_abs = float(error.mean())
    max_rel = float(error.max() / (ref.abs().mean() + 1e-6))
    print(f"[VSA256 route {block_shape}] avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < 1e-3
    assert max_rel < 0.2
