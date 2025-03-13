import torch
import time
from einops import rearrange

def tile(x, sp_size):
    x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=12 // sp_size, h=48, w=80)
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                     n_t=2,
                     n_h=6,
                     n_w=10,
                     ts_t=6,
                     ts_h=8,
                     ts_w=8)

q = torch.load("query.pt")
k = torch.load("key.pt")
v = torch.load("value.pt")
text_mask = torch.load("text_mask.pt")

query, encoder_query = q.split_with_sizes((q.shape[1] - 256, 256), dim=1)
key, encoder_key = k.split_with_sizes((k.shape[1] - 256, 256), dim=1)
value, encoder_value = v.split_with_sizes((v.shape[1] - 256, 256), dim=1)

q = torch.cat([tile(query, 1), encoder_query], dim=1).transpose(1, 2)
k = torch.cat([tile(key, 1), encoder_key], dim=1).transpose(1, 2)
v = torch.cat([tile(value, 1), encoder_value], dim=1).transpose(1, 2)

selected_strategies = [(2, 6, 1), (1, 6, 10), (2, 3, 3), (2, 6, 10), (2, 1, 10), (2, 3, 5)]
text_length = text_mask.sum()
selected_attn_processor = []
from torch.nn.attention.flex_attention import flex_attention
from functools import partial
from csrc.sliding_tile_attention.test.sba import get_sliding_block_attention_mask

for ms in selected_strategies:
    mask = get_sliding_block_attention_mask(ms, (6, 8, 8), (12, 48, 80), text_length, "cuda")
    attn_processor = torch.compile(partial(flex_attention, block_mask=mask), mode="max-autotune-no-cudagraphs")
    selected_attn_processor.append(attn_processor)
    
warmup_time = 1

print(q.shape)

for processor in selected_attn_processor:
    processor(q, k, v)

for processor in selected_attn_processor:
    torch.cuda.synchronize()
    start_time = time.time()
    processor(q, k, v)
    torch.cuda.synchronize()
    actuall_time = time.time()-start_time
    strategy = selected_strategies[selected_attn_processor.index(processor)]
    t, h, w = strategy
    print(f"for startegy {selected_strategies[selected_attn_processor.index(processor)]}")
    print(f"theortical speed up is {(2*6*10)/(t*h*w)}")
    print(f"actual speed up is {0.15808820724487305/actuall_time}")
    
