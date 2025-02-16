import torch
import torch.nn as nn
from einops import rearrange
from fastvideo.utils.communications import all_to_all_4D
from flash_attn import flash_attn_func
    
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

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
        k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
        v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)
        x = flash_attn_func(q, k, v, causal=causal)
        x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)
        return x

