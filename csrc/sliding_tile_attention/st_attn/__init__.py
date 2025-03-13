import math

import torch
from st_attn_cuda import sta_fwd
from torch.nn.attention.flex_attention import flex_attention
from functools import lru_cache
from csrc.sliding_tile_attention.test.flex_sta_ref import get_sliding_tile_attention_mask
@lru_cache(maxsize=32)
def get_compiled_flex_attention(strategy, tile_size, image_size, text_length, device):
    """
    Create and compile flex attention with a specific sliding block mask.
    This function is cached to avoid recompiling for the same parameters.
    
    Args:
        strategy (tuple): A tuple (t, h, w) defining the strategy
        tile_size (tuple): A tuple (ts_t, ts_h, ts_w) defining the tile size
        image_size (tuple): A tuple (n_t, n_h, n_w) defining the image size
        text_length (int): The text length
        device (str): The device to use
        
    Returns:
        function: A compiled flex attention function with the specified mask
    """
    # Convert strategy to the required format (ceil(t*3/2), h*2, w)
    adjusted_strategy = strategy
    
    # Get the sliding block attention mask
    mask = get_sliding_tile_attention_mask(
        adjusted_strategy, 
        tile_size, 
        image_size, 
        text_length, 
        device
    )
    
    def flex_attn_with_mask(q, k, v, scale=None):
        return flex_attention(q, k, v, block_mask=mask, scale=scale)
    
    # Compile the wrapper function
    compiled_flex_attn = torch.compile(flex_attn_with_mask)
    
    return compiled_flex_attn

def flex_sliding_tile_attention(q_all, k_all, v_all, strategy, tile_size, 
                           image_size, text_length, scale=None):
    device = q_all.device
    
    # Get the compiled flex attention function (cached if called with same parameters)
    compiled_flex_attn = get_compiled_flex_attention(
        strategy, 
        tile_size,
        image_size, 
        text_length, 
        device
    )
    
    
    # Apply the compiled flex attention
    output = compiled_flex_attn(q_all, k_all, v_all, scale=scale)

        
    return output

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
