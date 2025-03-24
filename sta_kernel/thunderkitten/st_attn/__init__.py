import math

import torch
from st_attn_cuda import sta_fwd


def sliding_tile_attention(q_all, k_all, v_all, window_size, text_length, has_text=True):
    seq_length = q_all.shape[2]
    if has_text:
        assert q_all.shape[
            2] == 115456, "STA currently only supports video with latent size (30, 48, 80), which is 117 frames x 768 x 1280 pixels"
        assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
            k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
            v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
    else:
        assert q_all.shape[2] == 82944

    hidden_states = torch.empty_like(q_all)
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1, head_index:head_index + 1],
                                              k_all[batch:batch + 1,
                                                    head_index:head_index + 1], v_all[batch:batch + 1,
                                                                                      head_index:head_index + 1],
                                              hidden_states[batch:batch + 1, head_index:head_index + 1])

            _ = sta_fwd(q_head, k_head, v_head, o_head, t_kernel, h_kernel, w_kernel, text_length, False, has_text)
    if has_text:
        _ = sta_fwd(q_all, k_all, v_all, hidden_states, 3, 3, 3, text_length, True, True)
    return hidden_states[:, :, :seq_length]
