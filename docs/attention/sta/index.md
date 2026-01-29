# Sliding Tile Attention (STA)

Optimized attention for window-based video generation (e.g., HunyuanVideo).

## Installation

STA is included in the `fastvideo-kernel` package. See the [main Attention page](../index.md) for build instructions.

## Usage

```python
from fastvideo_kernel import sliding_tile_attention

# q, k, v: [batch_size, num_heads, seq_length, head_dim]
# window_size: List of (t, h, w) tiles. Tile size is (6, 8, 8).
# text_length: Number of text tokens (0-256)

out = sliding_tile_attention(
    q, k, v, 
    window_size=[(3, 3, 3)],  # Example window
    text_length=256
)
```

## Citation

If you use Sliding Tile Attention in your research, please cite:

```bibtex
@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}
```
