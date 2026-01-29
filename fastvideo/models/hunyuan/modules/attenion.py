import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from st_attn import sliding_tile_attention
except ImportError:
    print("Could not load Sliding Tile Attention.")
    sliding_tile_attention = None

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import get_sequence_parallel_state, mccl_info

import torch.distributed as dist
group = None
try:
    from videogenkern import CeComm
    USE_CE = True
except:
    USE_CE = False

_GLOBAL_CE_COMM = None


def attention(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def tile(x, sp_size):
    x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=30 // sp_size, h=48, w=80)
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                     n_t=5,
                     n_h=6,
                     n_w=10,
                     ts_t=6,
                     ts_h=8,
                     ts_w=8)


def untile(x, sp_size):
    x = rearrange(x,
                  "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
                  n_t=5,
                  n_h=6,
                  n_w=10,
                  ts_t=6,
                  ts_h=8,
                  ts_w=8)
    return rearrange(x, "b (t sp h w) head d -> b (sp t h w) head d", sp=sp_size, t=30 // sp_size, h=48, w=80)


def _specific_all_to_all_4D(input: torch.tensor,
                  scatter_idx: int = 2,
                  gather_idx: int = 1,
                  alloc_id: int = None,
                  use_sync: bool = False,
                  async_op: bool = False):
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    from fastvideo.utils.parallel_states import mccl_info
    global group
    if group is None:
        group = mccl_info.group
    
    # ulyssess pg
    seq_world_size = torch.distributed.get_world_size(group)
    # if not USE_CE: 
    if USE_CE: # disable ce_comm for a stable commit
        global _GLOBAL_CE_COMM
        if _GLOBAL_CE_COMM is None:
            import torch.distributed as dist
            _GLOBAL_CE_COMM = CeComm(group=group, rank=dist.get_rank(group=group), num_ranks=dist.get_world_size(group=group))
        ce_comm = _GLOBAL_CE_COMM

        if scatter_idx == 2 and gather_idx == 1:
            bs, shard_seq_len, hc, hs = input.shape
            seqlen = shard_seq_len * seq_world_size
            shard_hc = hc // seq_world_size
            
            # (bs, seqlen/P, hc, hs) -> (P, seq_len/P, bs, hc/P, hs)
            input_t: torch.Tensor = input.reshape(bs, shard_seq_len, seq_world_size, shard_hc, hs).transpose(0, 2)
            alloc_sizes = list(input_t.size())
            alloc_strides = [1] * len(alloc_sizes)
            for i in range(len(alloc_sizes) - 2, -1 ,-1):
                alloc_strides[i] = alloc_sizes[i + 1] * alloc_strides[i + 1]
            all2all_input = ce_comm.alloc_tensor(
                sizes=alloc_sizes,
                strides=alloc_strides,
                dtype=input_t.dtype,
                alloc_id=alloc_id,
            )  # all2all_input should be in contiguous memory format
            all2all_input.copy_(input_t)
            all2all_output = torch.empty_like(all2all_input, device="musa")
            assert (
                not all2all_input.is_cpu and not all2all_output.is_cpu
            ), "all2all buffer tensor must on device"
                
            if seq_world_size > 1:
                ce_comm.all_to_all_single(all2all_output, all2all_input, async_op=async_op)
                if use_sync:
                    torch.musa.synchronize()
            else:
                all2all_output = all2all_input
            
            if not async_op:
                all2all_output = all2all_output.reshape(seqlen, bs, shard_hc, hs)
                all2all_output = all2all_output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

            return all2all_output
        
        elif scatter_idx == 1 and gather_idx == 2:
            bs, seqlen, shard_hc, hs = input.shape
            hc = shard_hc * seq_world_size
            shard_seqlen = seqlen // seq_world_size
            
            # (bs, seqlen, hc/P, hs) -> (P, hc/P, seqlen/P, bs, hs)
            input_t = input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs).transpose(0, 3).transpose(0, 1)
            alloc_sizes = list(input_t.size())
            alloc_strides = [1] * len(alloc_sizes)
            for i in range(len(alloc_sizes) - 2, -1 ,-1):
                alloc_strides[i] = alloc_sizes[i + 1] * alloc_strides[i + 1]
            all2all_input = ce_comm.alloc_tensor(
                sizes=alloc_sizes,
                strides=alloc_strides,
                dtype=input_t.dtype,
                alloc_id=alloc_id,
            )
            all2all_input.copy_(input_t)
            all2all_output = torch.empty_like(all2all_input, device="musa")
            assert (
                not all2all_input.is_cpu and not all2all_output.is_cpu
            ), "all2all buffer tensor must on device"
            
            if seq_world_size > 1:
                ce_comm.all_to_all_single(all2all_output, all2all_input, async_op=async_op)
                if use_sync:
                    torch.musa.synchronize()
            else:
                all2all_output = all2all_input
            
            if not async_op:
                all2all_output = all2all_output.reshape(hc, shard_seqlen, bs, hs)
                all2all_output = all2all_output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            
            return all2all_output
        else:
            raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")
        
    import torch.distributed as dist
    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seq_len, hc, hs = input.shape
        seqlen = shard_seq_len * seq_world_size
        shard_hc = hc // seq_world_size
        
        # (bs, seqlen/P, hc, hs) ---> (P, seq_len/P, bs, hc/P, hs)
        input_t: torch.Tensor = input.reshape(bs, shard_seq_len, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()
        assert (
            not input_t.is_cpu
        ), "input tensor must on device"
        all2all_input = input_t

        all2all_output = torch.empty_like(all2all_input, device="musa")

        if seq_world_size > 1:
            dist.all_to_all_single(all2all_output, all2all_input, async_op=async_op)
            if use_sync:
                torch.musa.synchronize()
        else:
           all2all_output = all2all_input
        if not async_op:
            all2all_output = all2all_output.reshape(seqlen, bs, shard_hc, hs)
            all2all_output = all2all_output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return all2all_output
    
    elif scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        # (bs, seqlen, hc/P, hs) ---> (bs, P, seqlen/P, hc/P, hs) ---> (hc/P, P, seqlen/P, bs, hs) ---> (P, hc/P, seqlen/P, bs, hs)
        input_t = input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs).transpose(0, 3).transpose(0, 1).contiguous()
        all2all_input = input_t

        all2all_output = torch.empty_like(all2all_input, device="musa")
        if seq_world_size > 1:
            dist.all_to_all_single(all2all_output, all2all_input, async_op=async_op)
            if use_sync:
                torch.musa.synchronize()
        else:
            all2all_output = all2all_input
        if not async_op:
            all2all_output = all2all_output.reshape(hc, shard_seqlen, bs, hs)
            all2all_output = all2all_output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return all2all_output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SpecificSeqAllToAll4D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scatter_idx=2, gather_idx=1, alloc_id=None, use_sync=False, async_op=False):
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        ctx.async_op = async_op
        ctx.alloc_id = alloc_id
        # Save input shape for backward when async_op=True to reconstruct 4D shape
        ctx.input_shape = input.shape
        return _specific_all_to_all_4D(input, scatter_idx, gather_idx, alloc_id, use_sync, async_op)

    @staticmethod
    def backward(ctx, grad_output):
        # When async_op=True in forward, the output was 5D (raw buffer).
        # Thus, grad_output arriving here is also 5D.
        # But the underlying _specific_all_to_all_4D expects a 4D tensor.
        if ctx.async_op and grad_output.dim() == 5:
            # We need to reshape/transpose the 5D gradient back to the expected 4D shape.
            # We can reuse transpose_all2all_output logic which handles the 5D->4D conversion.
            # Note: transpose_all2all_output is defined later in this file, so we assume it's available.
            from fastvideo.utils.parallel_states import mccl_info
            
            # Create a dummy tensor with the original 4D shape to guide the transposition logic
            # We use empty() to avoid memory overhead, just need shape/dtype/device
            dummy_input = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            
            grad_output = transpose_all2all_output(
                dummy_input,
                grad_output,
                scatter_idx=ctx.scatter_idx,
                gather_idx=ctx.gather_idx,
                group=mccl_info.group
            )
        
        # We must use synchronous execution for backward pass
        d_input = SpecificSeqAllToAll4D.apply(grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.alloc_id, ctx.use_sync,
                                              False)
        return d_input, None, None, None, None, None


@torch.compiler.disable
def specific_all_to_all_4D(input: torch.tensor,
                           scatter_idx: int = 2,
                           gather_idx: int = 1,
                           alloc_id: int  = None,
                           use_sync: bool = False,
                           async_op: bool = False):
    return SpecificSeqAllToAll4D.apply(input, scatter_idx, gather_idx, alloc_id, use_sync, async_op)


def transpose_all2all_output(all2all_input, all2all_output, scatter_idx=2, gather_idx=1, group=None):
    """
    Manually transpose the raw all2all output buffer after async communication completes.
    This is used when async_op=True, since the output buffer is not reshaped in that case.
    """
    seq_world_size = torch.distributed.get_world_size(group)
    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seq_len, hc, hs = all2all_input.shape
        seqlen = shard_seq_len * seq_world_size
        shard_hc = hc // seq_world_size
        all2all_output = all2all_output.reshape(seqlen, bs, shard_hc, hs)
        all2all_output = all2all_output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        return all2all_output
    elif scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = all2all_input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        all2all_output = all2all_output.reshape(hc, shard_seqlen, bs, hs)
        all2all_output = all2all_output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
        return all2all_output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask, mask_strategy=None):
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    text_length = text_mask.sum()

    if get_sequence_parallel_state():
        query_input = query  # save original input shape for transpose
        key_input = key
        query_raw = specific_all_to_all_4D(query, scatter_idx=2, gather_idx=1, alloc_id=0, async_op=True)
        key_raw = specific_all_to_all_4D(key, scatter_idx=2, gather_idx=1, alloc_id=1, async_op=True)
        # V sync acts as implicit barrier for Q/K
        value = specific_all_to_all_4D(value, scatter_idx=2, gather_idx=1, alloc_id=2, async_op=False)
        # Now Q/K comms are guaranteed complete, transpose their buffers
        query = transpose_all2all_output(query_input, query_raw, scatter_idx=2, gather_idx=1, group=mccl_info.group)
        key = transpose_all2all_output(key_input, key_raw, scatter_idx=2, gather_idx=1, group=mccl_info.group)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // mccl_info.sp_size
            return encoder_state.narrow(dim, mccl_info.rank_within_group * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    if mask_strategy[0] is not None:
        query = torch.cat([tile(query, mccl_info.sp_size), encoder_query], dim=1).transpose(1, 2)
        key = torch.cat([tile(key, mccl_info.sp_size), encoder_key], dim=1).transpose(1, 2)
        value = torch.cat([tile(value, mccl_info.sp_size), encoder_value], dim=1).transpose(1, 2)

        head_num = query.size(1)
        current_rank = mccl_info.rank_within_group
        start_head = current_rank * head_num
        windows = [mask_strategy[head_idx + start_head] for head_idx in range(head_num)]

        hidden_states = sliding_tile_attention(query, key, value, windows, text_length).transpose(1, 2)
    else:
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length),
                                                                          dim=1)

    if mask_strategy[0] is not None:
        hidden_states = untile(hidden_states, mccl_info.sp_size)

    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()

    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
