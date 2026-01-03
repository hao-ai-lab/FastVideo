import math
import torch
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
        sta_fwd(
            q[:, head_idx:head_idx + 1], k[:, head_idx:head_idx + 1],
            v[:, head_idx:head_idx + 1], output[:, head_idx:head_idx + 1],
            t, h, w, text_length, False, has_text, flag
        )

    if has_text:
        sta_fwd(q, k, v, output, 3, 3, 3, text_length, True, True, flag)

    return output[:, :, :seq_length]


def video_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor = None,
) -> torch.Tensor:
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, heads, seq_len, dim = q.shape

    # Compression branch
    q_c = q.view(batch, heads, seq_len // block_elements, block_elements, dim)
    k_c = k.view(batch, heads, seq_len // block_elements, block_elements, dim)
    v_c = v.view(batch, heads, seq_len // block_elements, block_elements, dim)

    q_c = (q_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)

    out_c = out_c.view(batch, heads, seq_len // block_elements, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements,
                         1).view(batch, heads, seq_len, dim)

    # Sparse branch
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores,
                            dtype=torch.bool).scatter_(-1, topk_idx, True)

    idx, num = map_to_index(mask)
    
    if block_sparse_fwd is not None:
        out_s = block_sparse_fwd(
            q, k, v, idx, num, variable_block_sizes.int()
        )[0] # block_sparse_fwd returns vector<Tensor>
    else:
        out_s, _ = triton_block_sparse_attn_forward(q, k, v, idx, num,
                                                    variable_block_sizes)

    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s
