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
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
from csrc.sliding_tile_attention.st_attn import flex_sliding_tile_attention

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


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask, mask_strategy=None):
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    text_length = text_mask.sum()

    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    if mask_strategy[0] is not None:
        query = torch.cat([tile(query, nccl_info.sp_size), encoder_query], dim=1).transpose(1, 2)
        key = torch.cat([tile(key, nccl_info.sp_size), encoder_key], dim=1).transpose(1, 2)
        value = torch.cat([tile(value, nccl_info.sp_size), encoder_value], dim=1).transpose(1, 2)

        head_num = query.size(1)
        current_rank = nccl_info.rank_within_group
        start_head = current_rank * head_num
        windows = [mask_strategy[head_idx + start_head] for head_idx in range(head_num)]

        if sliding_tile_attention is not None:
            hidden_states = sliding_tile_attention(query, key, value, windows, text_length).transpose(1, 2)
        else:
            hidden_states = torch.empty_like(query)
            strategy_to_heads = {}
            for head_index in range(head_num):
                strategy = tuple(windows[head_index])  # Convert list to tuple for dict key
                if strategy not in strategy_to_heads:
                    strategy_to_heads[strategy] = []
                strategy_to_heads[strategy].append(head_index)
            for strategy, heads in strategy_to_heads.items():                
                # Gather all heads with this strategy
                query_heads = torch.cat([query[:, head_idx:head_idx + 1, :, :] for head_idx in heads], dim=1)
                key_heads = torch.cat([key[:, head_idx:head_idx + 1, :, :] for head_idx in heads], dim=1)
                value_heads = torch.cat([value[:, head_idx:head_idx + 1, :, :] for head_idx in heads], dim=1)
                
                # Process all heads with this strategy at once
                # processed_heads = selected_attn_processor[processor_idx](query_heads, key_heads, value_heads)
                processed_heads = flex_sliding_tile_attention(query_heads, key_heads, value_heads, strategy, (6, 8, 8), (30, 48, 80), text_length)
                
                # Distribute results back to the correct positions
                for i, head_idx in enumerate(heads):
                    hidden_states[:, head_idx:head_idx + 1, :, :] = processed_heads[:, i:i + 1, :, :]
                
            hidden_states = hidden_states.transpose(1, 2)
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
        hidden_states = untile(hidden_states, nccl_info.sp_size)

    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()

    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
