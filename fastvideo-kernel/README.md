# FastVideo Kernel

CUDA kernels for FastVideo video generation.

## Installation

```bash
git submodule update --init --recursive
cd fastvideo-kernel
pip install .
```

## Usage

```python
from fastvideo-kernel import sliding_tile_attention, video_sparse_attn, moba_attn_varlen

# Example: Sliding Tile Attention
out = sliding_tile_attention(q, k, v, window_sizes, text_len)

# Example: Video Sparse Attention (with Triton fallback)
out = video_sparse_attn(q, k, v, block_sizes, topk=5)

# Example: VMoBA
out = moba_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
```

## Requirements

- H100 GPU (sm_90a) for CUDA kernels
- Triton for non-H100 fallback
