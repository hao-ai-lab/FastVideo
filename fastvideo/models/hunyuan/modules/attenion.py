import torch
import torch.nn.functional as F
from einops import rearrange
import time
# try:
#     from st_attn import sliding_tile_attention
# except ImportError:
#     print("Could not load Sliding Tile Attention.")
#     sliding_tile_attention = None

import tk_4090_cuda as tk

from csrc.sliding_tile_attention.test.sba import sliding_tile_attention
from torch.nn.attention.flex_attention import flex_attention
from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info

flex_attention = torch.compile(flex_attention, dynamic=False)
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
    x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=12 // sp_size, h=48, w=80)
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                     n_t=2,
                     n_h=6,
                     n_w=10,
                     ts_t=6,
                     ts_h=8,
                     ts_w=8)


def untile(x, sp_size):
    x = rearrange(x,
                  "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
                  n_t=2,
                  n_h=6,
                  n_w=10,
                  ts_t=6,
                  ts_h=8,
                  ts_w=8)
    return rearrange(x, "b (t sp h w) head d -> b (sp t h w) head d", sp=sp_size, t=12 // sp_size, h=48, w=80)


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask, mask_strategy=None, selected_attn_processor=None):
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
        
        hidden_states = torch.empty_like(query)
        
        time_start = time.time()
        torch.cuda.synchronize()
        
        # process the qkv head by head
        for head_index in range(head_num):
            query_head = query[:, head_index:head_index + 1, :, :]
            key_head = key[:, head_index:head_index + 1, :, :]
            value_head = value[:, head_index:head_index + 1, :, :]
            mask = windows[head_index]
            
            # selected_attn_processor [(2, 6, 1), (1, 6, 10), (2, 3, 3), (2, 6, 10), (2, 1, 10), (2, 3, 5)]
            
            # print(f"mask_strategy: {mask_strategy[head_index]}")
            
            if mask_strategy[head_index] == [2, 6, 1]:
                head_hidden_states = selected_attn_processor[0](query_head, key_head, value_head).unsqueeze(0).transpose(1, 2)
            elif mask_strategy[head_index] == [1, 6, 10]:
                head_hidden_states = selected_attn_processor[1](query_head, key_head, value_head).unsqueeze(0).transpose(1, 2)
            elif mask_strategy[head_index] == [2, 3, 3]:
                head_hidden_states = selected_attn_processor[2](query_head, key_head, value_head).unsqueeze(0).transpose(1, 2)
            elif mask_strategy[head_index] == [2, 1, 10]:
                head_hidden_states = selected_attn_processor[4](query_head, key_head, value_head).unsqueeze(0).transpose(1, 2)
            elif mask_strategy[head_index] == [2, 3, 5]:
                head_hidden_states = selected_attn_processor[5](query_head, key_head, value_head).unsqueeze(0).transpose(1, 2)
            else:
                head_hidden_states = selected_attn_processor[3](query_head, key_head, value_head).unsqueeze(0).transpose(1, 2)
                
            hidden_states[:, head_index:head_index + 1, :, :] = head_hidden_states
        torch.cuda.synchronize()
        time_end = time.time()
        print(f"Time taken for sliding tile attention: {time_end - time_start}")
    else:
        # query = torch.cat([query, encoder_query], dim=1)
        # key = torch.cat([key, encoder_key], dim=1)
        # value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        # result = torch.empty_like(query)
        # start_time = time.time()
        # torch.cuda.synchronize()

        # output = tk.attention_fwd_4090(query, key, value, result, text_length)
        
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Time taken for tk attention: {end_time - start_time}")
        
        # qkv = torch.stack([query, key, value], dim=2)

        # attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        

        # flash_start_time = time.time()
        # torch.cuda.synchronize()        
        # hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)
        # torch.cuda.synchronize()
        # flash_end_time = time.time()
        # print(f"Time taken for flash attention: {flash_end_time - flash_start_time}")
        
        
        flex_start_time = time.time()
        query = torch.cat([tile(query, nccl_info.sp_size), encoder_query], dim=1).transpose(1, 2)
        key = torch.cat([tile(key, nccl_info.sp_size), encoder_key], dim=1).transpose(1, 2)
        value = torch.cat([tile(value, nccl_info.sp_size), encoder_value], dim=1).transpose(1, 2)
        torch.cuda.synchronize()
        mask = get_sliding_block_attention_mask((2,3,5), (6, 8, 8), (12, 48, 80), text_length, 'cuda')
        # print(f"query shape: {query.shape}")
        # print(f"key shape: {key.shape}")
        # print(f"value shape: {value.shape}")
        
        flex_hidden_states = flex_attention(query, key, value, block_mask=mask).transpose(1, 2)
        torch.cuda.synchronize()
        flex_end_time = time.time()
        print(f"Time taken for flex attention: {flex_end_time - flex_start_time}")
        
        
        hidden_states = flex_hidden_states

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
