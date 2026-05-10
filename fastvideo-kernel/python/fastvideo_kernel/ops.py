import math
import torch
from .block_sparse_attn import block_sparse_attn, block_sparse_attn_from_indices
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


def video_sparse_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    variable_block_sizes_list: list,
    q_variable_block_sizes_list: list,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor = None,
) -> torch.Tensor:
    """
    Varlen variant of video_sparse_attn following flash_attn_varlen_func convention.

    Args:
        q:  [total_q, heads, dim]  — flat packed query tokens
        k:  [total_kv, heads, dim] — flat packed key tokens
        v:  [total_kv, heads, dim] — flat packed value tokens
        cu_seqlens_q:  [batch+1] int32 cumulative Q sequence lengths
        cu_seqlens_kv: [batch+1] int32 cumulative KV sequence lengths
        variable_block_sizes_list:   list of per-sequence KV block size tensors
        q_variable_block_sizes_list: list of per-sequence Q block size tensors
        topk:        number of KV blocks each Q block attends to
        block_size:  tile size (int or 3-tuple), default 64
        compress_attn_weight: optional [total_q, heads, dim] gate tensor

    Returns:
        [total_q, heads, dim]
    """
    batch = cu_seqlens_q.shape[0] - 1
    outputs = []

    for i in range(batch):
        q_s = cu_seqlens_q[i].item()
        q_e = cu_seqlens_q[i + 1].item()
        kv_s = cu_seqlens_kv[i].item()
        kv_e = cu_seqlens_kv[i + 1].item()

        # Slice and reshape: [S, H, D] → [1, H, S, D]
        q_i = q[q_s:q_e].unsqueeze(0).transpose(1, 2).contiguous()
        k_i = k[kv_s:kv_e].unsqueeze(0).transpose(1, 2).contiguous()
        v_i = v[kv_s:kv_e].unsqueeze(0).transpose(1, 2).contiguous()

        caw_i = None
        if compress_attn_weight is not None:
            caw_i = compress_attn_weight[q_s:q_e].unsqueeze(0).transpose(1, 2).contiguous()

        # [1, H, S_q, D]
        out_i = video_sparse_attn(
            q_i, k_i, v_i,
            variable_block_sizes_list[i],
            q_variable_block_sizes_list[i],
            topk,
            block_size=block_size,
            compress_attn_weight=caw_i,
        )

        # [1, H, S_q, D] → [S_q, H, D]
        outputs.append(out_i.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)


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
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    if v.shape[2] != kv_seq_len:
        raise ValueError(
            f"Expected k and v to have the same sequence length, got "
            f"k.shape[2]={kv_seq_len}, v.shape[2]={v.shape[2]}"
        )
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("Expected q/k/v to have the same batch and head dimensions.")

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

    # Compression branch
    q_c = q.view(batch, heads, q_num_blocks, block_elements, dim)
    k_c = k.view(batch, heads, kv_num_blocks, block_elements, dim)
    v_c = v.view(batch, heads, kv_num_blocks, block_elements, dim)

    q_c = (q_c.float().sum(dim=3) / q_variable_block_sizes.view(1, 1, -1, 1)).to(
        q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)

    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements,
                         1).view(batch, heads, q_seq_len, dim)

    # Sparse branch: feed top-k indices directly, skipping the bool-mask round-trip.
    topk_idx = torch.topk(scores, topk, dim=-1).indices
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
