import math
import torch
from .block_sparse_attn import block_sparse_attn, block_sparse_attn_from_indices
from .block_sparse_attn_256 import (
    block_sparse_attn_256,
    block_sparse_attn_256_bshd,
)
from .triton_kernels.st_attn_triton import sliding_tile_attention_triton

# Try to load the C++ extension
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    sta_fwd = getattr(fastvideo_kernel_ops, "sta_fwd", None)
except ImportError:
    sta_fwd = None


def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: list,
    text_length: int,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    # Check if the specific op is available
    if sta_fwd is None:
        return sliding_tile_attention_triton(
            q, k, v, window_size, text_length, has_text, seq_shape
        )

    seq_length = q.shape[2]
    shape_map = {"30x48x80": 1, "36x48x48": 2, "18x48x80": 3}

    if has_text:
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)

    output = torch.empty_like(q)
    flag = shape_map[seq_shape]

    for head_idx, (t, h, w) in enumerate(window_size):
        # Per-head slices are not contiguous in the batch dimension when batch>1
        # (they keep the original head-stride). The TK kernel assumes contiguous
        # [B, H, S, D] layout, so we materialize a contiguous [B,1,S,D] view.
        q_h = q[:, head_idx:head_idx + 1].contiguous()
        k_h = k[:, head_idx:head_idx + 1].contiguous()
        v_h = v[:, head_idx:head_idx + 1].contiguous()
        o_h = torch.empty_like(q_h)
        sta_fwd(
            q_h, k_h,
            v_h, o_h,
            t, h, w, text_length, False, has_text, flag
        )
        output[:, head_idx:head_idx + 1] = o_h

    if has_text:
        sta_fwd(q.contiguous(), k.contiguous(), v.contiguous(), output, 3, 3, 3, text_length, True, True, flag)

    return output[:, :, :seq_length]


def video_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor = None,
) -> torch.Tensor:
    """VSA entrypoint for [B, H, S, D] tensors.

    Dispatches the sparse branch by ``block_elements = prod(block_size)``:
    - 64  -> existing TK/Triton path (see ``block_sparse_attn_from_indices``).
    - 256 -> CuTe FA4 block-sparse attention (see ``block_sparse_attn_256``).

    Backend overrides:
    - ``FASTVIDEO_VSA_TRITON=1`` forces Triton in either path.
    - ``FASTVIDEO_VSA_TK=1`` prefers the sm_90 TK kernel in the 64-block path.
    - ``FASTVIDEO_VSA_CUTEDSL=1`` prefers CuTe in the 256-block path.
    """
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)
    block_elements = block_size[0] * block_size[1] * block_size[2]

    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("Expected q/k/v to have the same batch and head dimensions.")
    if v.shape[2] != kv_seq_len:
        raise ValueError(
            f"Expected k and v to have the same sequence length, got "
            f"k.shape[2]={kv_seq_len}, v.shape[2]={v.shape[2]}"
        )

    if q_seq_len % block_elements != 0 or kv_seq_len % block_elements != 0:
        raise ValueError(
            f"q_seq_len and kv_seq_len must be divisible by block_elements={block_elements}, "
            f"got q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}"
        )
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements
    if variable_block_sizes.numel() != kv_num_blocks:
        raise ValueError(
            f"variable_block_sizes must have length kv_num_blocks={kv_num_blocks}, "
            f"got {variable_block_sizes.numel()}"
        )
    if q_variable_block_sizes.numel() != q_num_blocks:
        raise ValueError(
            f"q_variable_block_sizes must have length q_num_blocks={q_num_blocks}, "
            f"got {q_variable_block_sizes.numel()}"
        )

    # Compression branch (token-level average per block + dense block-level attn).
    q_c = q.view(batch, heads, q_num_blocks, block_elements, dim)
    k_c = k.view(batch, heads, kv_num_blocks, block_elements, dim)
    v_c = v.view(batch, heads, kv_num_blocks, block_elements, dim)
    q_c = (q_c.float().sum(dim=3) / q_variable_block_sizes.view(1, 1, -1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements, 1).view(batch, heads, q_seq_len, dim)

    # Sparse branch.
    topk_idx = torch.topk(scores, topk, dim=-1).indices

    if block_elements == 256:
        # CuTe path consumes a bool mask (full/partial split inside the wrapper).
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
        out_s = block_sparse_attn_256(q, k, v, mask, variable_block_sizes)[0]
    else:
        # Index-native path for 64-block (TK/Triton).
        q2k_idx = topk_idx.to(torch.int32).contiguous()
        q2k_num = torch.full(
            (batch, heads, q_num_blocks),
            topk,
            dtype=torch.int32,
            device=q.device,
        )
        out_s = block_sparse_attn_from_indices(
            q, k, v, q2k_idx, q2k_num, variable_block_sizes
        )[0]

    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s


def video_sparse_attn_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor = None,
) -> torch.Tensor:
    """VSA entrypoint for [B, S, H, D] tensors.

    Avoids the BHSD<->BSHD round-trip that ``video_sparse_attn`` performs on
    the CuTe 256-block path; the 64-block path still expects BHSD and is not
    supported here.
    """
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)
    block_elements = block_size[0] * block_size[1] * block_size[2]
    if block_elements != 256:
        raise ValueError(
            "video_sparse_attn_bshd is only defined for block_elements=256 "
            f"(got {block_elements}); use video_sparse_attn for the 64-block path."
        )

    batch, q_seq_len, heads, dim = q.shape
    kv_seq_len = k.shape[1]
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[2] != heads or v.shape[2] != heads:
        raise ValueError("Expected q/k/v to have the same batch and head dimensions.")
    if v.shape[1] != kv_seq_len:
        raise ValueError(
            f"Expected k and v to have the same sequence length, got "
            f"k.shape[1]={kv_seq_len}, v.shape[1]={v.shape[1]}"
        )
    if q_seq_len % block_elements != 0 or kv_seq_len % block_elements != 0:
        raise ValueError(
            f"q_seq_len and kv_seq_len must be divisible by block_elements={block_elements}, "
            f"got q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}"
        )
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements
    if variable_block_sizes.numel() != kv_num_blocks:
        raise ValueError(
            f"variable_block_sizes must have length kv_num_blocks={kv_num_blocks}, "
            f"got {variable_block_sizes.numel()}"
        )
    if q_variable_block_sizes.numel() != q_num_blocks:
        raise ValueError(
            f"q_variable_block_sizes must have length q_num_blocks={q_num_blocks}, "
            f"got {q_variable_block_sizes.numel()}"
        )

    # Compression branch (BSHD-native: mean over the 256-token axis).
    token_idx = torch.arange(block_elements, device=q.device, dtype=torch.int32)
    q_token_valid = (token_idx.view(1, -1) < q_variable_block_sizes.view(-1, 1)).view(
        1, q_num_blocks, block_elements, 1, 1
    )
    kv_token_valid = (token_idx.view(1, -1) < variable_block_sizes.view(-1, 1)).view(
        1, kv_num_blocks, block_elements, 1, 1
    )

    q_c = q.view(batch, q_num_blocks, block_elements, heads, dim)
    k_c = k.view(batch, kv_num_blocks, block_elements, heads, dim)
    v_c = v.view(batch, kv_num_blocks, block_elements, heads, dim)
    q_c = ((q_c.float() * q_token_valid).sum(dim=2)
           / q_variable_block_sizes.view(1, -1, 1, 1)).to(q.dtype)
    k_c = ((k_c.float() * kv_token_valid).sum(dim=2)
           / variable_block_sizes.view(1, -1, 1, 1)).to(k.dtype)
    v_c = ((v_c.float() * kv_token_valid).sum(dim=2)
           / variable_block_sizes.view(1, -1, 1, 1)).to(v.dtype)
    q_ch = q_c.permute(0, 2, 1, 3).contiguous()
    k_ch = k_c.permute(0, 2, 1, 3).contiguous()
    v_ch = v_c.permute(0, 2, 1, 3).contiguous()

    scores = torch.matmul(q_ch, k_ch.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c_ch = torch.matmul(attn, v_ch)
    out_c_blk = out_c_ch.permute(0, 2, 1, 3).contiguous()

    # Sparse branch (CuTe BSHD).
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
    out_s, _ = block_sparse_attn_256_bshd(q, k, v, mask, variable_block_sizes)

    out = out_s
    out_view = out.view(batch, q_num_blocks, block_elements, heads, dim)
    if compress_attn_weight is not None:
        gate_view = compress_attn_weight.view(
            batch, q_num_blocks, block_elements, heads, dim
        )
        out_view.add_(out_c_blk.unsqueeze(2) * gate_view)
    else:
        out_view.add_(out_c_blk.unsqueeze(2))
    return out
