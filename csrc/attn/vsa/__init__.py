import math

import torch
from torch.utils.checkpoint import detach_variable
from typing import Tuple
try:
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
except ImportError:
    block_sparse_fwd = None
    block_sparse_bwd = None


BLOCK_M = 64
BLOCK_N = 64

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

def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK

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

def block_sparse_attention_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num):
    """
    block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]. 
        [*, *, i, j] = 1 means the i-th q block should attend to the j-th kv block.
    """
    # assert all elements in q2k_block_sparse_num can be devisible by 2
    o, lse = block_sparse_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num)
    return o, lse

def block_sparse_attention_backward(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num):
    grad_output = grad_output.contiguous()
    grad_q, grad_k, grad_v = block_sparse_bwd(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num)
    return grad_q, grad_k, grad_v

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

@triton.jit
def topk_index_to_map_kernel(
    map_ptr,
    index_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    topk: tl.constexpr,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    for i in tl.static_range(topk):
        index = tl.load(index_ptr_base + i * index_kv_stride)
        tl.store(map_ptr_base + index * map_kv_stride, 1.0)

@triton.jit
def map_to_index_kernel(
    map_ptr,
    index_ptr,
    index_num_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    index_num_bs_stride,
    index_num_h_stride,
    index_num_q_stride,
    num_kv_blocks: tl.constexpr,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.static_range(num_kv_blocks):
        map_entry = tl.load(map_ptr_base + i * map_kv_stride)
        if map_entry:
            tl.store(index_ptr_base + num * index_kv_stride, i)
            num += 1

    tl.store(
        index_num_ptr + b * index_num_bs_stride + h * index_num_h_stride +
        q * index_num_q_stride, num)

def topk_index_to_map(index: torch.Tensor,
                      num_kv_blocks: int,
                      transpose_map: bool = False):
    """
    Convert topk indices to a map.
    
    Args:
        index: [bs, h, num_q_blocks, topk]
            The topk indices tensor.
        num_kv_blocks: int
            The number of key-value blocks in the block_map returned
        transpose_map: bool
            If True, the block_map will be transposed on the final two dimensions.
    
    Returns:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            A binary map where 1 indicates that the q block attends to the kv block.
    """
    bs, h, num_q_blocks, topk = index.shape

    if transpose_map is False:
        block_map = torch.zeros((bs, h, num_q_blocks, num_kv_blocks),
                                dtype=torch.bool,
                                device=index.device)
    else:
        block_map = torch.zeros((bs, h, num_kv_blocks, num_q_blocks),
                                dtype=torch.bool,
                                device=index.device)
        block_map = block_map.transpose(2, 3)

    grid = (bs, h, num_q_blocks)
    topk_index_to_map_kernel[grid](
        block_map,
        index,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        topk=topk,
    )

    return block_map

def map_to_index(block_map: torch.Tensor):
    """
    Convert a block map to indices and counts.
    
    Args:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            The block map tensor.
    
    Returns:
        index: [bs, h, num_q_blocks, num_kv_blocks]
            The indices of the blocks.
        index_num: [bs, h, num_q_blocks]
            The number of blocks for each q block.
    """
    bs, h, num_q_blocks, num_kv_blocks = block_map.shape

    index = torch.full((block_map.shape),
                       -1,
                       dtype=torch.int32,
                       device=block_map.device)
    index_num = torch.empty((bs, h, num_q_blocks),
                            dtype=torch.int32,
                            device=block_map.device)

    grid = (bs, h, num_q_blocks)
    map_to_index_kernel[grid](
        block_map,
        index,
        index_num,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        index_num.stride(0),
        index_num.stride(1),
        index_num.stride(2),
        num_kv_blocks=num_kv_blocks,
    )

    return index, index_num

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
