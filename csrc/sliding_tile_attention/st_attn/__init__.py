import math

import torch
from torch.utils.checkpoint import detach_variable
from .selective_tile_attention import get_block_sparse_mask
try:
    from st_attn_cuda import sta_fwd
except ImportError:
    sta_fwd = None
try:
    from st_attn_cuda import block_sparse_fwd, block_sparse_bwd
except ImportError:
    block_sparse_fwd = None
    block_sparse_bwd = None
try:
    from st_attn_cuda import mha_fwd, mha_bwd
except ImportError:
    mha_fwd = None
    mha_bwd = None

BLOCK_M = 64
BLOCK_N = 64
def sliding_tile_attention(q_all, k_all, v_all, window_size, text_length, has_text=True, dit_seq_shape='30*48*80'):
    seq_length = q_all.shape[2]
    dit_seq_shape_mapping = {
        '30x48x80':1,
        '36x48x48':2,
        '18x48x80':3, 
    }
    if has_text:
        assert q_all.shape[
            2] >= 115200, "STA currently only supports video with latent size (30, 48, 80), which is 117 frames x 768 x 1280 pixels"
        assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
            k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
            v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
    else:
        if dit_seq_shape == '36x48x48': # Stepvideo 204x768x68
            assert q_all.shape[2] == 82944
        elif dit_seq_shape == '18x48x80': # Wan 69x768x1280
            assert q_all.shape[2] == 69120
        else:
            raise ValueError(f"Unsupported {dit_seq_shape}, current shape is {q_all.shape}, only support '36x48x48' for Stepvideo and '18x48x80' for Wan")

    kernel_aspect_ratio_flag = dit_seq_shape_mapping[dit_seq_shape]
    hidden_states = torch.empty_like(q_all)
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1, head_index:head_index + 1],
                                              k_all[batch:batch + 1,
                                                    head_index:head_index + 1], v_all[batch:batch + 1,
                                                                                      head_index:head_index + 1],
                                              hidden_states[batch:batch + 1, head_index:head_index + 1])

            _ = sta_fwd(q_head, k_head, v_head, o_head, t_kernel, h_kernel, w_kernel, text_length, False, has_text, kernel_aspect_ratio_flag)
    if has_text:
        _ = sta_fwd(q_all, k_all, v_all, hidden_states, 3, 3, 3, text_length, True, True, kernel_aspect_ratio_flag)
    return hidden_states[:, :, :seq_length]

def block_sparse_attention_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num):
    """
    block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]. 
        [*, *, i, j] = 1 means the i-th q block should attend to the j-th kv block.
    """
    # assert all elements in q2k_block_sparse_num can be devisible by 2
    o, lse = block_sparse_fwd(q, k, v, q2k_block_sparse_index, q2k_block_sparse_num)
    return o, lse

def block_sparse_attention_fwd_16x16(q, k, v, block_sparse_map):
    """
    block_sparse_map: [bs, h, num_q_blocks, num_kv_blocks]. 
        [*, *, i, j] = 1 means the i-th q block should attend to the j-th kv block.
    """
    # assert all elements in q2k_block_sparse_num can be devisible by 2
    o, lse = block_sparse_fwd(q, k, v, block_sparse_map)
    return o, lse

def block_sparse_attention_backward(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num):
    grad_q, grad_k, grad_v = block_sparse_bwd(q, k, v, o, l_vec, grad_output, k2q_block_sparse_index, k2q_block_sparse_num)
    return grad_q, grad_k, grad_v

def block_sparse_attention_backward_16x16(q, k, v, o, l_vec, grad_output, block_sparse_map_transposed):
    grad_q, grad_k, grad_v = block_sparse_bwd(q, k, v, o, l_vec, grad_output, block_sparse_map_transposed)
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

class BlockSparseAttentionFunction16x16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_sparse_map):
        block_sparse_map = block_sparse_map.to(torch.uint8)
        o, lse = block_sparse_attention_fwd_16x16(q, k, v, block_sparse_map)
        block_sparse_map_trans = block_sparse_map.transpose(2, 3).contiguous()
        ctx.save_for_backward(q, k, v, o, lse, block_sparse_map_trans)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, block_sparse_map_trans = ctx.saved_tensors
        grad_q, grad_k, grad_v = block_sparse_attention_backward_16x16(
            q, k, v, o, lse, grad_output, block_sparse_map_trans
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


def mha_forward(q, k, v):
    o, l_vec = mha_fwd(q, k, v)
    return o, l_vec

def mha_backward(q, k, v, o, l_vec, grad_output):
    grad_q, grad_k, grad_v = mha_bwd(q, k, v, o, l_vec, grad_output)
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