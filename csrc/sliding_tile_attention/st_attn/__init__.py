import math
import subprocess
import torch
from torch.nn.attention.flex_attention import flex_attention
from functools import lru_cache
from typing import Tuple
from torch import BoolTensor, IntTensor
from torch.nn.attention.flex_attention import create_block_mask

# Peiyuan: This is neccesay. Dont know why. see https://github.com/pytorch/pytorch/issues/135028
torch._inductor.config.realize_opcount_threshold = 100


def generate_sta_mask(canvas_twh, kernel_twh, tile_twh, text_length):
    """Generates a 3D NATTEN attention mask with a given kernel size.
    
    Args:
        canvas_t: The time dimension of the canvas.
        canvas_h: The height of the canvas.
        canvas_w: The width of the canvas.
        kernel_t: The time dimension of the kernel.
        kernel_h: The height of the kernel.
        kernel_w: The width of the kernel.
    """
    canvas_t, canvas_h, canvas_w = canvas_twh
    kernel_t, kernel_h, kernel_w = kernel_twh
    tile_t_size, tile_h_size, tile_w_size = tile_twh
    total_tile_size = tile_t_size * tile_h_size * tile_w_size
    canvas_tile_t, canvas_tile_h, canvas_tile_w = canvas_t // tile_t_size, canvas_h // tile_h_size, canvas_w // tile_w_size
    img_seq_len = canvas_t * canvas_h * canvas_w

    def get_tile_t_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        tile_id = idx // total_tile_size
        tile_t = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w = tile_id % canvas_tile_w
        return tile_t, tile_h, tile_w

    def sta_mask_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_t_tile, q_x_tile, q_y_tile = get_tile_t_x_y(q_idx)
        kv_t_tile, kv_x_tile, kv_y_tile = get_tile_t_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t_tile.clamp(kernel_t // 2, (canvas_tile_t - 1) - kernel_t // 2)
        kernel_center_x = q_x_tile.clamp(kernel_h // 2, (canvas_tile_h - 1) - kernel_h // 2)
        kernel_center_y = q_y_tile.clamp(kernel_w // 2, (canvas_tile_w - 1) - kernel_w // 2)
        time_mask = (kernel_center_t - kv_t_tile).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x_tile).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y_tile).abs() <= kernel_w // 2
        image_mask = (q_idx < img_seq_len) & (kv_idx < img_seq_len)
        image_to_text_mask = (q_idx < img_seq_len) & (kv_idx >= img_seq_len) & (kv_idx < img_seq_len + text_length)
        text_to_all_mask = (q_idx >= img_seq_len) & (kv_idx < img_seq_len + text_length)
        return (image_mask & time_mask & hori_mask & vert_mask) | image_to_text_mask | text_to_all_mask

    sta_mask_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return sta_mask_3d


def get_sliding_tile_attention_mask(kernel_size, tile_size, img_size, text_length, device, text_max_len=256):
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    image_mask = generate_sta_mask(img_size, kernel_size, tile_size, text_length)
    mask = create_block_mask(image_mask,
                             B=None,
                             H=None,
                             Q_LEN=img_seq_len + text_max_len,
                             KV_LEN=img_seq_len + text_max_len,
                             device=device,
                             _compile=True)
    return mask


def get_gpu_type():
    try:
        # Run nvidia-smi to get GPU information
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode('utf-8')
        
        # Check if H100 is in any of the GPU names
        gpus = [gpu.strip() for gpu in result.split('\n') if gpu.strip()]
        
        for gpu in gpus:
            if 'H100' in gpu:
                return 'H100'
            if '4090' in gpu:
                return '4090'
                
        return None
    except Exception as e:
        return None

gpu_type = get_gpu_type()

if gpu_type == 'H100':
    from st_attn_cuda import sta_fwd

    
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
    if gpu_type == 'H100':
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
    else:
        assert q_all.shape[
                2] == 46336, "Flex STA currently only supports video with latent size (12, 48, 80), which is 45 frames x 768 x 1280 pixels"
        head_num = q_all.size(1)
        hidden_states = torch.empty_like(q_all)
        strategy_to_heads = {}
        for head_index in range(head_num):
            strategy = tuple(window_size[head_index])  # Convert list to tuple for dict key
            if strategy not in strategy_to_heads:
                strategy_to_heads[strategy] = []
            strategy_to_heads[strategy].append(head_index)
        for strategy, heads in strategy_to_heads.items():                
            # Gather all heads with this strategy
            query_heads = torch.cat([q_all[:, head_idx:head_idx + 1, :, :] for head_idx in heads], dim=1)
            key_heads = torch.cat([k_all[:, head_idx:head_idx + 1, :, :] for head_idx in heads], dim=1)
            value_heads = torch.cat([v_all[:, head_idx:head_idx + 1, :, :] for head_idx in heads], dim=1)
            
            # Process all heads with this strategy at once
            # processed_heads = selected_attn_processor[processor_idx](query_heads, key_heads, value_heads)
            processed_heads = flex_sliding_tile_attention(query_heads, key_heads, value_heads, strategy, (6, 8, 8), (12, 48, 80), text_length)
            
            # Distribute results back to the correct positions
            for i, head_idx in enumerate(heads):
                hidden_states[:, head_idx:head_idx + 1, :, :] = processed_heads[:, i:i + 1, :, :]
            
        return hidden_states
