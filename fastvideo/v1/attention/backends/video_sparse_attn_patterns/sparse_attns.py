try:
    from st_attn import block_sparse_attn
except:
    block_sparse_attn = None
from functools import lru_cache
from typing import List, Tuple

import torch

from .mask import generate_topk_block_sparse_pattern


# TODO: write it in cuda and put it under st_attn
def generate_topk_block_sparse_pattern_torch(block_attn_score, topk):
    """
    Generate a block sparse pattern where each q block attends to exactly topk kv blocks,
    based on the provided attention scores.
    
    Args:
        block_attn_score: [bs, h, num_q_blocks, num_kv_blocks]
            Attention scores between query and key blocks
        topk: int
            Number of kv blocks each q block attends to
        
    Returns:
        q2k_block_sparse_index: [bs, h, num_q_blocks, topk]
            Contains the indices of kv blocks that each q block attends to.
        q2k_block_sparse_num: [bs, h, num_q_blocks]
            Contains the number of kv blocks that each q block attends to (all equal to topk).
        k2q_block_sparse_index: [bs, h, num_kv_blocks, max_q_per_kv]
            Contains the indices of q blocks that attend to each kv block.
        k2q_block_sparse_num: [bs, h, num_kv_blocks]
            Contains the number of q blocks that attend to each kv block.
        block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]
            Binary mask where 1 indicates attention connection.
    """
    device = block_attn_score.device
    # Extract dimensions from block_attn_score
    bs, h, num_q_blocks, num_kv_blocks = block_attn_score.shape

    # Ensure topk is not larger than num_kv_blocks
    assert topk <= num_kv_blocks, f"topk ({topk}) must be less than or equal to num_kv_blocks ({num_kv_blocks})"

    # Get top-k indices for each q block
    _, q2k_block_sparse_index = torch.topk(block_attn_score, topk, dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(torch.int32)

    # sort q2k_block_sparse_index
    q2k_block_sparse_index, _ = torch.sort(q2k_block_sparse_index, dim=-1)

    # All q blocks attend to exactly topk kv blocks
    q2k_block_sparse_num = torch.full((bs, h, num_q_blocks),
                                      topk,
                                      dtype=torch.int32,
                                      device=device)

    # Create the reverse mapping (k2q)
    # First, initialize lists to collect q indices for each kv block
    k2q_indices_list: List[List[List[int]]] = [[[]
                                                for _ in range(num_kv_blocks)]
                                               for _ in range(bs * h)]

    # Populate the lists based on q2k mapping
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx].tolist()
                for kv_idx in kv_indices:
                    k2q_indices_list[flat_idx][kv_idx].append(q_idx)

    # Find the maximum number of q blocks that attend to any kv block
    max_q_per_kv = 0
    for flat_idx in range(bs * h):
        for kv_idx in range(num_kv_blocks):
            max_q_per_kv = max(max_q_per_kv,
                               len(k2q_indices_list[flat_idx][kv_idx]))

    # Create tensors for k2q mapping
    k2q_block_sparse_index = torch.full((bs, h, num_kv_blocks, max_q_per_kv),
                                        -1,
                                        dtype=torch.int32,
                                        device=device)
    k2q_block_sparse_num = torch.zeros((bs, h, num_kv_blocks),
                                       dtype=torch.int32,
                                       device=device)

    # Fill the tensors
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for kv_idx in range(num_kv_blocks):
                q_indices = k2q_indices_list[flat_idx][kv_idx]
                num_q = len(q_indices)
                k2q_block_sparse_num[b, head, kv_idx] = num_q
                if num_q > 0:
                    k2q_block_sparse_index[b, head,
                                           kv_idx, :num_q] = torch.tensor(
                                               q_indices,
                                               dtype=torch.int32,
                                               device=device)

    return q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def sparse_attn_c_s_p(q, k, v, topk, block_size, compress_attn_weight=None):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements % 64 == 0 and block_elements >= 64
    assert q.shape[2] % block_elements == 0  # [1, 12, 29120, 128]
    batch_size, num_heads, seq_len, head_dim = q.shape
    # compress attn
    q_compress = q.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements,
                        head_dim).mean(dim=3)  # [1, 12, 455, 128]
    k_compress = k.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    v_compress = v.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)

    output_compress, block_attn_score = torch_attention(
        q_compress, k_compress, v_compress)  # [1, 12, 455, 128]

    output_compress = output_compress.view(batch_size, num_heads,
                                           seq_len // block_elements, 1,
                                           head_dim)  # [1, 12, 455, 1, 128]
    output_compress = output_compress.repeat(1, 1, 1, block_elements, 1).view(
        batch_size, num_heads, seq_len, head_dim)  # [1, 12, 29120, 128]

    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num = generate_topk_block_sparse_pattern(
        block_attn_score, topk)  #kind of slow # [1, 12, 455, 64] [1, 12, 455]
    # (q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
    output_select = block_sparse_attn(
        q, k, v, q2k_block_sparse_index, q2k_block_sparse_num,
        k2q_block_sparse_index, k2q_block_sparse_num)  # [1, 12, 29120, 128]

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select
    return final_output


def sparse_attn_c_s(q, k, v, topk, block_size):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements % 64 == 0 and block_elements >= 64
    assert q.shape[2] % block_elements == 0
    batch_size, num_heads, seq_len, head_dim = q.shape

    # compress attn
    q_compress = q.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    k_compress = k.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    v_compress = v.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)

    output_compress, block_attn_score = torch_attention(q_compress, k_compress,
                                                        v_compress)

    output_compress = output_compress.view(batch_size, num_heads,
                                           seq_len // block_elements, 1,
                                           head_dim)
    output_compress = output_compress.repeat(1, 1, 1, block_elements,
                                             1).view(batch_size, num_heads,
                                                     seq_len, head_dim)

    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num = generate_topk_block_sparse_pattern(
        block_attn_score, topk)
    # (q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
    output_select = block_sparse_attn(q, k, v, q2k_block_sparse_index,
                                      q2k_block_sparse_num,
                                      k2q_block_sparse_index,
                                      k2q_block_sparse_num)

    final_output = output_compress + output_select
    return final_output


def sparse_attn_s(q, k, v, topk, block_size):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements % 64 == 0 and block_elements >= 64
    assert q.shape[2] % block_elements == 0
    batch_size, num_heads, seq_len, head_dim = q.shape

    # compress attn
    q_compress = q.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    k_compress = k.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)

    block_attn_score = torch.matmul(q_compress, k_compress.transpose(-2, -1))

    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num = generate_topk_block_sparse_pattern(
        block_attn_score, topk)
    # (q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
    output_select = block_sparse_attn(q, k, v, q2k_block_sparse_index,
                                      q2k_block_sparse_num,
                                      k2q_block_sparse_index,
                                      k2q_block_sparse_num)

    final_output = output_select
    return final_output


@lru_cache(maxsize=None)
def generate_stav1_block_sparse_pattern(batch_size,
                                        head_size,
                                        T,
                                        H,
                                        W,
                                        device,
                                        window_size=(1, 1, 1)):
    block_attn_score = torch.zeros(batch_size,
                                   head_size,
                                   T * H * W,
                                   T * H * W,
                                   device=device)
    seq_length = T * H * W
    for i in range(seq_length):
        from_t = i // (H * W)
        from_h = (i % (H * W)) // W
        from_w = i % W

        from_t = max(window_size[0] // 2,
                     min(from_t, (T - 1) - window_size[0] // 2))
        from_h = max(window_size[1] // 2,
                     min(from_h, (H - 1) - window_size[1] // 2))
        from_w = max(window_size[2] // 2,
                     min(from_w, (W - 1) - window_size[2] // 2))
        for j in range(seq_length):
            to_t = j // (H * W)
            to_h = (j % (H * W)) // W
            to_w = j % W
            if abs(from_t - to_t) <= window_size[0] // 2 and abs(
                    from_h - to_h) <= window_size[1] // 2 and abs(
                        from_w - to_w) <= window_size[2] // 2:
                block_attn_score[:, :, i, j] = 1
    topk = window_size[0] * window_size[1] * window_size[2]
    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num = generate_topk_block_sparse_pattern(
        block_attn_score, topk)
    return q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num


def sparse_attn_l(q, k, v, topk, block_size):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements % 64 == 0 and block_elements >= 64
    assert q.shape[2] % block_elements == 0
    batch_size, num_heads, seq_len, head_dim = q.shape

    # compress attn
    q_compress = q.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    k_compress = k.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)

    block_attn_score = torch.matmul(q_compress, k_compress.transpose(-2, -1))

    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num = generate_topk_block_sparse_pattern(
        block_attn_score, topk)
    # (q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
    output_select = block_sparse_attn(q, k, v, q2k_block_sparse_index,
                                      q2k_block_sparse_num,
                                      k2q_block_sparse_index,
                                      k2q_block_sparse_num)

    final_output = output_select
    return final_output
