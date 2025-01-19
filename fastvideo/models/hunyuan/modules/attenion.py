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
    map_k = torch.cat([key, encoder_key], dim=1).transpose(1, 2)    # [B, H, S+T, D]
    d_k = map_q.size(-1)

    shape = (16, 45, 45)
    q_coords = torch.tensor([get_block(i, shape=shape) for i in range(sequence_length)],
                           device='cuda', dtype=torch.int16)
    k_coords = torch.tensor([get_block(i, shape=shape) for i in range(img_kv_len)],
                           device='cuda', dtype=torch.int16)

    diffs = (q_coords.unsqueeze(1) - k_coords.unsqueeze(0)).abs()  # [seq_len, kv_len, 3]
    mask_t_diff = 6
    mask_x_diff = 12
    mask_y_diff = 12
    
    mask_t = diffs[..., 0] <= mask_t_diff
    mask_x = diffs[..., 1] <= mask_x_diff
    mask_y = diffs[..., 2] <= mask_y_diff
    valid_mask_2d = mask_t & mask_x & mask_y  # [seq_len, kv_len]
    del mask_t, mask_x, mask_y, diffs
    
    # Pad mask for text part (all True)
    full_valid_mask = F.pad(valid_mask_2d, (0, encoder_sequence_length), value=True)  # [seq_len, kv_len+text_len]
    img_mask_density = valid_mask_2d.float().mean().item()
    full_mask_density = full_valid_mask.float().mean().item()
    del valid_mask_2d

    chunk_size = 512
    
    save_attn = False #(sample_step == 1) and (layer_id == 59)
    if save_attn:
        attn_map_cumulated = torch.zeros((sequence_length, img_kv_len + encoder_sequence_length), 
                                       dtype=torch.float32, device='cuda')
    
    # 逐head计算
    for head_idx in range(num_heads):
        current_q = map_q[:, head_idx:head_idx+1].to(dtype=torch.float32)  # [B, 1, S, D]
        current_k = map_k[:, head_idx:head_idx+1].to(dtype=torch.float32)  # [B, 1, S+T, D]

        valid_score_total = 0.0
        all_score_total = 0.0
        
        if save_attn:
            head_attn_map = torch.zeros((sequence_length, img_kv_len + encoder_sequence_length), 
                                      dtype=torch.float32, device='cuda')
        
        # 分块计算
        for i in range(0, current_q.size(2), chunk_size):
            chunk_end = min(i + chunk_size, current_q.size(2))
            q_chunk = current_q[:, :, i:chunk_end]  # [B, 1, chunk_size, D]
            
            scores_32 = torch.matmul(
                q_chunk, 
                current_k.transpose(-2, -1)
            ) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device='cuda'))
            
            attn_weights = F.softmax(scores_32, dim=-1)
            
            chunk_valid_mask = full_valid_mask[i:chunk_end].unsqueeze(0).unsqueeze(0)
            valid_score_sum = (attn_weights * chunk_valid_mask).sum()
            all_score_sum = attn_weights.sum()
            
            valid_score_total += valid_score_sum.item()
            all_score_total += all_score_sum.item()
            
            # For last layer, accumulate attention weights
            if save_attn:
                head_attn_map[i:chunk_end] = attn_weights.squeeze(0).squeeze(0)
            
            del scores_32, attn_weights
            torch.cuda.empty_cache()
        
        recall = valid_score_total / (all_score_total + 1e-9)
        print(f"step{sample_step}_layer{layer_id}_head{head_idx}_window_diff_{mask_t_diff}_{mask_x_diff}_{mask_y_diff}_shape_{shape[0]}_{shape[1]}_{shape[2]}_img{img_mask_density:.4f}_full{full_mask_density:.4f}'s recall:", recall)
        
        if save_attn:
            attn_map_cumulated += head_attn_map
    
    if save_attn:
        attn_map_avg = attn_map_cumulated / num_heads
        torch.save(attn_map_avg, f'logs/attn_map_step{sample_step}_layer{layer_id}.pt')
    
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

def get_block(idx, shape=(16, 30, 30)):
    t_size, x_size, y_size = shape
    xy = x_size * y_size
    t = idx // xy
    r = idx % xy
    x = r // y_size
    y = r % y_size
    return t, x, y
