# Video Sparse Attention (VSA)

Sparse attention mechanism selecting top-k blocks.

## Installation

VSA is included in the `fastvideo-kernel` package. See the [main Attention page](../index.md) for build instructions.

## Usage

```python
from fastvideo_kernel import video_sparse_attn

# q, k, v: [batch_size, num_heads, seq_len, head_dim]
# variable_block_sizes: Number of valid tokens per block
# topk: Number of blocks to attend

output = video_sparse_attn(
    q, k, v, 
    variable_block_sizes=block_sizes, 
    topk=32
)
```

## Citation

If you use Video Sparse Attention in your research, please cite:

```bibtex
@article{zhang2025vsa,
  title={Vsa: Faster video diffusion with trainable sparse attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Huang, Haofeng and Lin, Will and Liu, Zhengzhong and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.13389},
  year={2025}
}
```
