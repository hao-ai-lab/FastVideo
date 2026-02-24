import math
import torch
from .block_sparse_attn import block_sparse_attn
from .triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward
from .triton_kernels.st_attn_triton import sliding_tile_attention_triton
from .triton_kernels.index import map_to_index

# Try to load the C++ extension
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    sta_fwd = getattr(fastvideo_kernel_ops, "sta_fwd", None)
    block_sparse_fwd = getattr(fastvideo_kernel_ops, "block_sparse_fwd", None)
    block_sparse_bwd = getattr(fastvideo_kernel_ops, "block_sparse_bwd", None)
except ImportError:
    sta_fwd = None
    block_sparse_fwd = None
    block_sparse_bwd = None

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

    # Sparse branch
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores,
                            dtype=torch.bool).scatter_(-1, topk_idx, True)

    idx, num = map_to_index(mask)
    
    if block_sparse_fwd is not None:
        # Use autograd-enabled wrapper so backward works (and still uses SM90 kernel when available)
        out_s = block_sparse_attn(q, k, v, mask, variable_block_sizes)[0]
    else:
        # Triton-only forward (kept for environments without the wrapper deps)
        out_s, _ = triton_block_sparse_attn_forward(q, k, v, idx, num, variable_block_sizes)

    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s
