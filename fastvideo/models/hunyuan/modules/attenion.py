import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad


def attention(
    q, k, v, drop_rate=0, attn_mask=None, causal=False,
):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(
        qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None
    )

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask):
    # 1GPU torch.Size([1, 11264, 24, 128]) tensor([    0, 11275, 11520], device='cuda:0', dtype=torch.int32)
    # 2GPU torch.Size([1, 5632, 24, 128]) tensor([   0, 5643, 5888], device='cuda:0', dtype=torch.int32)
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(
                dim, nccl_info.rank_within_group * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    # Hint: please check encoder_query.shape
    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)
    # B, S, 3, H, D
    qkv = torch.stack([query, key, value], dim=2)

    attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
    hidden_states = flash_attn_no_pad(
        qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
    )

    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
        (sequence_length, encoder_sequence_length), dim=1
    )
    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()
    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
