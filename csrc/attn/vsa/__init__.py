import torch
from typing import Tuple
from vsa.block_sparse_wrapper import block_sparse_attn
try:
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
except ImportError:
    block_sparse_fwd = None
    block_sparse_bwd = None
from vsa.block_sparse_attn_triton import attention as triton_attention, attention_sparse as triton_attention_sparse
from vsa.index import topk_index_to_map, map_to_index
BLOCK_M = 64
BLOCK_N = 64


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def video_sparse_attn(q, k, v, topk, block_size, compress_attn_weight=None):
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

    output_select = block_sparse_attn(q, k, v, q2k_block_sparse_index,
                                      q2k_block_sparse_num,
                                      k2q_block_sparse_index,
                                      k2q_block_sparse_num)

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select
    return final_output


def generate_topk_block_sparse_pattern(block_attn_score: torch.Tensor,
                                       topk: int):
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
    """
    device = block_attn_score.device
    # Extract dimensions from block_attn_score
    bs, h, num_q_blocks, num_kv_blocks = block_attn_score.shape

    sorted_result = torch.sort(block_attn_score, dim=-1, descending=True)

    sorted_indice = sorted_result.indices

    q2k_block_sparse_index, _ = torch.sort(sorted_indice[:, :, :, :topk],
                                           dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(dtype=torch.int32)
    q2k_block_sparse_num = torch.full((bs, h, num_q_blocks),
                                      topk,
                                      device=device,
                                      dtype=torch.int32)

    block_map = topk_index_to_map(q2k_block_sparse_index,
                                  num_kv_blocks,
                                  transpose_map=True)
    k2q_block_sparse_index, k2q_block_sparse_num = map_to_index(
        block_map.transpose(2, 3))

    return q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num


