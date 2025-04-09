(sta-usage)=

# Usage

```python
from st_attn import sliding_tile_attention
# assuming video size (T, H, W) = (30, 48, 80), text tokens = 256 with padding. 
# q, k, v: [batch_size, num_heads, seq_length, head_dim], seq_length = T*H*W + 256
# a tile is a cube of size (6, 8, 8)
# window_size in tiles: [(window_t, window_h, window_w), (..)...]. For example, window size (3, 3, 3) means a query can attend to (3x6, 3x8, 3x8) = (18, 24, 24) tokens out of the total 30x48x80 video.
# text_length: int ranging from 0 to 256
# If your attention contains text token (Hunyuan)
out = sliding_tile_attention(q, k, v, window_size, text_length)
# If your attention does not contain text token (StepVideo)
out = sliding_tile_attention(q, k, v, window_size, 0, False)

```