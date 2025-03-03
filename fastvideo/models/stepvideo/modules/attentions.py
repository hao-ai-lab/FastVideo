import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func

try:
    from st_attn import sliding_tile_attention
except ImportError:
    print("Could not load Sliding Tile Attention.")
    sliding_tile_attention = None

from fastvideo.utils.communications import all_to_all_4D
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        else:
            raise Exception('Not supported attention type...')

    def tile(self, x, sp_size):
        x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=36 // sp_size, h=48, w=48)
        return rearrange(x,
                         "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                         n_t=6,
                         n_h=6,
                         n_w=6,
                         ts_t=6,
                         ts_h=8,
                         ts_w=8)

    def untile(self, x, sp_size):
        x = rearrange(x,
                      "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
                      n_t=6,
                      n_h=6,
                      n_w=6,
                      ts_t=6,
                      ts_h=8,
                      ts_w=8)
        return rearrange(x, "b (t sp h w) head d -> b (sp t h w) head d", sp=sp_size, t=36 // sp_size, h=48, w=48)

    def torch_attn_func(self, q, k, v, attn_mask=None, causal=False, drop_rate=0.0, **kwargs):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        if attn_mask is not None and attn_mask.ndim == 3:  ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = torch.nn.functional.scaled_dot_product_attention(q,
                                                             k,
                                                             v,
                                                             attn_mask=attn_mask,
                                                             dropout_p=drop_rate,
                                                             is_causal=causal)
        x = rearrange(x, 'b h s d -> b s h d')
        return x

    def parallel_attn_func(self, q, k, v, causal=False, mask_strategy=None, **kwargs):
        if get_sequence_parallel_state():
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)

        if mask_strategy[0] is not None:
            q = self.tile(q, nccl_info.sp_size).transpose(1, 2).contiguous()
            k = self.tile(k, nccl_info.sp_size).transpose(1, 2).contiguous()
            v = self.tile(v, nccl_info.sp_size).transpose(1, 2).contiguous()

            head_num = q.size(1)  # 48 // sp_size
            current_rank = nccl_info.rank_within_group

            start_head = current_rank * head_num
            windows = [mask_strategy[head_idx + start_head] for head_idx in range(head_num)]

            x = sliding_tile_attention(q, k, v, windows, 0, False).transpose(1, 2).contiguous()
            x = self.untile(x, nccl_info.sp_size)
        else:
            x = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        if get_sequence_parallel_state():
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        x = x.to(q.dtype)
        return x
