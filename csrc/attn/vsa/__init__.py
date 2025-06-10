import math

import torch
from torch.utils.checkpoint import detach_variable

try:
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
except ImportError:
    block_sparse_fwd = None
    block_sparse_bwd = None


BLOCK_M = 64
BLOCK_N = 64
def block_sparse_attention_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num):
    """
    block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]. 
        [*, *, i, j] = 1 means the i-th q block should attend to the j-th kv block.
    """
    # assert all elements in q2k_block_sparse_num can be devisible by 2
    o, lse = block_sparse_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num)
    return o, lse

def block_sparse_attention_backward(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num):
    grad_q, grad_k, grad_v = block_sparse_bwd(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num)
    return grad_q, grad_k, grad_v

class BlockSparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num):
        o, lse = block_sparse_attention_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num)
        ctx.save_for_backward(q, k, v, o, lse, k2q_block_sparse_index, k2q_block_sparse_num)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, k2q_block_sparse_index, k2q_block_sparse_num = ctx.saved_tensors
        grad_q, grad_k, grad_v = block_sparse_attention_backward(
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
    return BlockSparseAttentionFunction.apply(
        q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num
    )

## pytorch sdpa version of block sparse ##
import triton
import triton.language as tl

@triton.jit
def index_to_mask_kernel(
    q2k_block_sparse_index_ptr,
    q2k_block_sparse_num_ptr,
    mask_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_q_blocks: tl.constexpr,
    num_k_blocks: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh, q, id = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    b = bh // num_heads
    h = bh % num_heads

    num_valid_blocks = tl.load(q2k_block_sparse_num_ptr + b * num_heads * num_q_blocks + h * num_q_blocks + q)

    if num_valid_blocks <= id:
        return
    k = tl.load(q2k_block_sparse_index_ptr + b * num_heads * num_q_blocks * max_kv_blocks + h * num_q_blocks * max_kv_blocks + q * max_kv_blocks + id)

    full_mask = (tl.arange(0, BLOCK_Q)[:, None] < BLOCK_Q) & (tl.arange(0, BLOCK_K)[None, :] < BLOCK_K)

    q_lengths = num_q_blocks * BLOCK_Q
    k_lengths = num_k_blocks * BLOCK_K
    mask_ptr_base = mask_ptr + b * num_heads * q_lengths * k_lengths + h * q_lengths * k_lengths + q * BLOCK_Q * k_lengths + k * BLOCK_K

    tl.store(mask_ptr_base + tl.arange(0, BLOCK_Q)[:, None] * k_lengths + tl.arange(0, BLOCK_K)[None, :], full_mask)

def index_to_mask(q2k_block_sparse_index, q2k_block_sparse_num, BLOCK_Q, BLOCK_K, num_k_blocks):
    """
    Convert block sparse indices to a mask.
    
    Args:
        q2k_block_sparse_index: Indices for query-to-key sparse blocks
        q2k_block_sparse_num: Number of sparse blocks for each query block
    
    Returns:
        mask: Block sparse mask tensor
    """
    batch_size, num_heads, num_q_blocks, max_kv_blocks = q2k_block_sparse_index.shape
    assert q2k_block_sparse_num.shape == (batch_size, num_heads, num_q_blocks)

    mask = torch.zeros((batch_size, num_heads, num_q_blocks * BLOCK_Q, num_k_blocks * BLOCK_K), dtype=torch.bool, device=q2k_block_sparse_index.device)

    grid = (batch_size * num_heads, num_q_blocks, max_kv_blocks)
    index_to_mask_kernel[grid](
        q2k_block_sparse_index,
        q2k_block_sparse_num,
        mask,
        batch_size,
        num_heads,
        num_q_blocks,
        num_k_blocks,
        max_kv_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )

    return mask

class DummyOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class CheckpointSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, obj, q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, block_q, block_k):
        """Forward pass."""
        with torch.no_grad():
            mask = index_to_mask(q2k_block_sparse_index, q2k_block_sparse_num, block_q, block_k, k.shape[2] // block_k)
            outputs = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        ctx.save_for_backward(*detach_variable((q, k, v, q2k_block_sparse_index, q2k_block_sparse_num)))
        ctx.block_q = block_q
        ctx.block_k = block_k
        # the obj is passed in, then it can access the saved input
        # tensors later for recomputation
        obj.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        inputs = ctx.saved_tensors
        output = ctx.output
        torch.autograd.backward(output, grad_output)
        ctx.output = None
        grads = tuple(inp.grad for inp in inputs)
        return (None, ) + grads + (None, None)


class BlockSparseAttnTorch:
    def __init__(self):
        self.ctx = None
    
    def recompute_mask(self, _):
        recomputed_mask = index_to_mask(self.q2k_block_sparse_index, self.q2k_block_sparse_num, self.block_q, self.block_k, self.num_kv_blocks)
        mask_size = recomputed_mask.untyped_storage().size()
        self.mask.untyped_storage().resize_(mask_size)
        self.mask.untyped_storage().copy_(recomputed_mask.untyped_storage()) 
    
    def recompute(self, _):
        q, k, v, q2k_block_sparse_index, q2k_block_sparse_num = self.ctx.saved_tensors
        block_q = self.ctx.block_q
        block_k = self.ctx.block_k
        mask = index_to_mask(q2k_block_sparse_index, q2k_block_sparse_num, block_q, block_k, k.shape[2] // block_k)
        with torch.enable_grad():
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.ctx.output = output
        self.ctx = None

    @torch._dynamo.disable
    def forward(self, q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, block_q, block_k):
        """
        Differentiable block sparse attention function using PyTorch.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len_kv, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len_kv, head_dim]
            q2k_block_sparse_index: Indices for query-to-key sparse blocks
            q2k_block_sparse_num: Number of sparse blocks for each query block
            block_q: Block size for query
            block_k: Block size for key-value

        Returns:
            output: Attention output tensor [batch_size, num_heads, seq_len_q, head_dim]
        """

        output = CheckpointSDPA.apply(
            self, q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, block_q, block_k
        )

        o = DummyOperator.apply(output)
        o.register_hook(self.recompute)
        return o