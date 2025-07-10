import torch
try:
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
except ImportError:
    block_sparse_fwd = None
    block_sparse_bwd = None
from .block_sparse_attn_triton import attention_sparse as block_sparse_attn_triton

class BlockSparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
        o, lse = block_sparse_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num)
        ctx.save_for_backward(q, k, v, o, lse, k2q_block_sparse_index, k2q_block_sparse_num)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, k2q_block_sparse_index, k2q_block_sparse_num = ctx.saved_tensors
        grad_q, grad_k, grad_v = block_sparse_bwd(
            q, k, v, o, lse, grad_output, k2q_block_sparse_index, k2q_block_sparse_num
        )
        return grad_q, grad_k, grad_v, None, None, None, None


@torch._dynamo.disable
def block_sparse_attn(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
    """
    Differentiable block sparse attention function.
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len_kv, head_dim]
        v: Value tensor [batch_size, num_heads, seq_len_kv, head_dim]
        q2k_block_sparse_index: Indices for query-to-key sparse blocks
        q2k_block_sparse_num: Number of sparse blocks for each query block
        k2q_block_sparse_index: Indices for key-to-query sparse blocks (for backward pass)
        k2q_block_sparse_num: Number of sparse blocks for each key block (for backward pass)
    
    Returns:
        output: Attention output tensor [batch_size, num_heads, seq_len_q, head_dim]
    """
    if block_sparse_fwd is not None:
        return BlockSparseAttentionFunction.apply(
            q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num
        )
    else:
        return block_sparse_attn_triton(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num)
