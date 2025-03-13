# def mask(b, h, q_idx, kv_idx):
#     return kv_idx < text_length + img_seq_len
from torch.nn.attention.flex_attention import create_block_mask, or_masks
from torch import IntTensor, BoolTensor
import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import math
# Peiyuan: This is neccesay. Dont know why. see https://github.com/pytorch/pytorch/issues/135028
torch._inductor.config.realize_opcount_threshold = 100
def generate_sba_mask(
    canvas_twh, 
    kernel_twh, 
    tile_twh,
    text_length
):
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

    def natten_mask_mod_3d(
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

    natten_mask_mod_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return natten_mask_mod_3d



def generate_baseline_3d_window_mask(
    canvas_twh, 
    kernel_twh, 
    img_seq_len,
    text_length
):
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
    def get_t_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        t = idx // (canvas_h * canvas_w)
        x = (idx % (canvas_h * canvas_w)) // canvas_w
        y = idx % canvas_w
        return t, x, y

    def natten_mask_mod_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_t, q_x, q_y = get_t_x_y(q_idx)
        kv_t, kv_x, kv_y = get_t_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t.clamp(kernel_t // 2, (canvas_t - 1) - kernel_t // 2)
        kernel_center_x = q_x.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        kernel_center_y = q_y.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        time_mask = (kernel_center_t - kv_t).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_w // 2
        no_pad_mask = kv_idx < text_length + img_seq_len
        return time_mask & hori_mask & vert_mask & no_pad_mask

    natten_mask_mod_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return natten_mask_mod_3d


def generate_tiled_3d_window_mask(
    canvas_twh, 
    kernel_twh, 
    img_seq_len,
    text_length
):
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
    tile_t, tile_h, tile_w = 4, 8, 8
    n_tile_t, n_tile_h, n_tile_w = canvas_t // tile_t, canvas_h // tile_h, canvas_w // tile_w
    
    def get_t_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        tile_id = idx // (tile_t * tile_h * tile_w)
        t_t, t_x, t_y = tile_id // (n_tile_h * n_tile_w), (tile_id % (n_tile_h * n_tile_w)) // n_tile_w, tile_id % n_tile_w
        t_offset = idx % (tile_t * tile_h * tile_w)
        i_t, i_x, i_y = t_offset // (tile_h * tile_w), (t_offset % (tile_h * tile_w)) // tile_w, t_offset % tile_w
        return t_t * tile_t + i_t, t_x * tile_h + i_x, t_y * tile_w + i_y
    
    def natten_mask_mod_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_t, q_x, q_y = get_t_x_y(q_idx)
        kv_t, kv_x, kv_y = get_t_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t.clamp(kernel_t // 2, (canvas_t - 1) - kernel_t // 2)
        kernel_center_x = q_x.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        kernel_center_y = q_y.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        time_mask = (kernel_center_t - kv_t).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_w // 2
        no_pad_mask = kv_idx < text_length + img_seq_len
        return time_mask & hori_mask & vert_mask & no_pad_mask

    natten_mask_mod_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return natten_mask_mod_3d


def generate_text_mask(img_seq_len, text_length):
    def text_mask(b, h, q_idx, kv_idx):
        mask1 = kv_idx < text_length + img_seq_len
        mask2 = kv_idx >= img_seq_len
        return mask1 & mask2
    return text_mask

 
def get_sliding_block_attention_mask(kernel_size, tile_size, img_size, text_length, device):
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    image_mask = generate_sba_mask(img_size, kernel_size, tile_size, text_length)
    mask = create_block_mask(image_mask, B=None, H=None, Q_LEN=img_seq_len+256 , KV_LEN=img_seq_len+256, device=device,  _compile=True, BLOCK_SIZE=128)
    return mask

def get_baseline_sliding_window_mask(kernel_size, img_seq_len, text_length, device):
    image_mask = generate_baseline_3d_window_mask( (32, 48, 80), kernel_size, img_seq_len, text_length)
    text_mask = generate_text_mask(img_seq_len=img_seq_len, text_length=text_length)
    mask = or_masks(image_mask, text_mask)
    mask = create_block_mask(mask, B=None, H=None, Q_LEN=img_seq_len + 256 , KV_LEN=img_seq_len + 256, device=device,  _compile=True)
    return mask

def get_tiled_sliding_window_mask(kernel_size, img_seq_len, text_length, device):
    image_mask = generate_tiled_3d_window_mask((32, 48, 80), kernel_size, img_seq_len, text_length)
    text_mask = generate_text_mask(img_seq_len=img_seq_len, text_length=text_length)
    mask = or_masks(image_mask, text_mask)
    mask = create_block_mask(mask, B=None, H=None, Q_LEN=img_seq_len + 256 , KV_LEN=img_seq_len + 256, device=device,  _compile=True)
    return mask


def sliding_tile_attention(q_all, k_all, v_all, window_size, text_length, has_text=True):
    seq_length = q_all.shape[2]
    # if has_text:
    #     assert q_all.shape[
    #         2] == 115456, "STA currently only supports video with latent size (30, 48, 80), which is 117 frames x 768 x 1280 pixels"
    #     assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"
    #     target_size = math.ceil(seq_length / 384) * 384
    #     pad_size = target_size - seq_length
    #     if pad_size > 0:
    #         q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
    #         k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
    #         v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
    # else:
    #     assert q_all.shape[2] == 82944

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

if __name__ == "__main__":
    # benchmark speed 
    from torch.nn.attention.flex_attention import flex_attention
    flex_attention = torch.compile(flex_attention)
    import time
    device = torch.device("cuda")
    kernel_size_ls = [(8, 6, 10), (4, 6, 10), (4, 6, 5), (4, 3, 5), (3, 3, 3)]
    tile_size = (4, 8, 8)
    
    # random input
    q = torch.randn(1, 24, 123136, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 24, 123136, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 24, 123136, 128, device=device, dtype=torch.bfloat16)
    
    for kernel_size in kernel_size_ls:
        mask = get_sliding_block_attention_mask(kernel_size, tile_size, (32, 48, 80), 39, device)
        