import torch
import torch.nn.functional as F

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,
                                             nccl_info)


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

    x = flash_attn_no_pad(qkv,
                          attn_mask,
                          causal=causal,
                          dropout_p=drop_rate,
                          softmax_scale=None)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask, sample_step, layer_id):
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
                dim, nccl_info.rank_within_group * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    # attn_map plot
    num_heads = query.size(2) 
    map_q = query.transpose(1, 2)  # [B, H, S, D]
    map_k = key.transpose(1, 2)    # [B, H, S, D]
    d_k = map_q.size(-1)

    q_coords = torch.tensor([get_block(i) for i in range(sequence_length)],
                           device='cuda', dtype=torch.int32)
    k_coords = torch.tensor([get_block(i) for i in range(img_kv_len)],
                           device='cuda', dtype=torch.int32)
    print("LENNNNN", sequence_length, img_kv_len)

    diffs = (q_coords.unsqueeze(1) - k_coords.unsqueeze(0)).abs()  # [seq_len, kv_len, 3]
    mask_t_diff = 3
    mask_x_diff = 5
    mask_y_diff = 5
    
    mask_t = diffs[..., 0] <= mask_t_diff
    mask_x = diffs[..., 1] <= mask_x_diff
    mask_y = diffs[..., 2] <= mask_y_diff
    valid_mask_2d = mask_t & mask_x & mask_y  # [seq_len, kv_len]

    # 逐head计算
    for head_idx in range(num_heads):
        # 只处理当前head
        current_q = map_q[:, head_idx:head_idx+1].to(dtype=torch.float32)  # [B, 1, S, D]
        current_k = map_k[:, head_idx:head_idx+1].to(dtype=torch.float32)  # [B, 1, S, D]
        
        scores_32 = torch.matmul(
            current_q, 
            current_k.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device='cuda'))
        
        attn_weights = F.softmax(scores_32, dim=-1)
        
        valid_score_sum = (attn_weights * valid_mask_2d.unsqueeze(0).unsqueeze(0)).sum()
        all_score_sum = attn_weights.sum()
        del scores_32, attn_weights
        
        recall = valid_score_sum / (all_score_sum + 1e-9)
        
        print(f"step{sample_step}_layer{layer_id}_head{head_idx}_window_diff_{mask_t_diff}_{mask_x_diff}_{mask_y_diff}'s recall:", recall.item())
        torch.cuda.empty_cache()  # 清理GPU缓存
    
    # Hint: please check encoder_query.shape
    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)
    # B, S, 3, H, D
    qkv = torch.stack([query, key, value], dim=2)

    attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
    hidden_states = flash_attn_no_pad(qkv,
                                      attn_mask,
                                      causal=False,
                                      dropout_p=0.0,
                                      softmax_scale=None)
    
    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
        (sequence_length, encoder_sequence_length), dim=1)
    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states,
                                      scatter_dim=1,
                                      gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states,
                                           dim=2).contiguous()
    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn

def get_block(idx, shape=(7, 90, 160)):
    t_size, x_size, y_size = shape
    xy = x_size * y_size
    t = idx // xy
    r = idx % xy
    x = r // y_size
    y = r % y_size
    return t, x, y
