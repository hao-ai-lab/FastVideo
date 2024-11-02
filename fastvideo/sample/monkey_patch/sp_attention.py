import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
from diffusers.models.embeddings import apply_rotary_emb

import diffusers
from types import MethodType

from diffusers.models.attention_processor import Attention
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
from fastvideo.utils.communications import all_gather_BHSD, all_to_all_SBH

class NewMochiAttnProcessor2_0:
    """Attention processor used in Mochi."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [b, s, h * d]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # [b, s, h=24, d=128]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # [b, 256, h * d]
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        # [b, 256, h=24, d=128]
        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        freqs_cos, freqs_sin = image_rotary_emb[0], image_rotary_emb[1]
        # shard the head dimension
        if get_sequence_parallel_state():
            # B, S, H, D to (S, B,) H, D
            batch_size, seq_len, attn_heads, head_dim = query.shape
            query = rearrange(query, 'b s h d -> (s b) h d')
            key = rearrange(key, 'b s h d -> (s b) h d')
            value = rearrange(value, 'b s h d -> (s b) h d')
            
            query = all_to_all_SBH(query, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, attn_heads // nccl_info.world_size, head_dim)
            key = all_to_all_SBH(key, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, attn_heads // nccl_info.world_size, head_dim)
            value = all_to_all_SBH(value, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, attn_heads // nccl_info.world_size, head_dim)
            # batch_size, S * world_size, H / world_size, D
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
            
            def shrink_head(encoder_state, dim):
                local_heads = encoder_state.shape[dim] // nccl_info.world_size
                return encoder_state.narrow(dim, nccl_info.rank * local_heads, local_heads)
            encoder_query = shrink_head(encoder_query, dim=2)
            encoder_key = shrink_head(encoder_key, dim=2)
            encoder_value = shrink_head(encoder_value, dim=2)
            
            freqs_cos = shrink_head(freqs_cos, dim=1)
            freqs_sin = shrink_head(freqs_sin, dim=1)
    
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)
    
        if image_rotary_emb is not None:
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)
            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)
            
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )
        # [b, h, s, d]
        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)

        # Hint: please check encoder_query.shape
        query = torch.cat([query, encoder_query], dim=2)
        key = torch.cat([key, encoder_key], dim=2)
        value = torch.cat([value, encoder_value], dim=2)
        
        
                
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=2
        )
        if get_sequence_parallel_state():
            hidden_states = rearrange(hidden_states, 'b h s d -> s b h d').reshape(-1, attn_heads // nccl_info.world_size, head_dim)
            hidden_states = all_to_all_SBH(hidden_states, scatter_dim=0, gather_dim=1).reshape(-1, batch_size, attn_heads, head_dim).transpose(0, 1)
            encoder_hidden_states = all_gather_BHSD(encoder_hidden_states, dim=1).contiguous()
            hidden_states = hidden_states.flatten(2, 3)
        else: 
            hidden_states = hidden_states.transpose(1,2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2).flatten(2, 3)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

