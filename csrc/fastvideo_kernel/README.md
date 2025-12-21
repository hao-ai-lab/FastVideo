# FastVideo Kernel

CUDA kernels for FastVideo video generation.

## Installation

```bash
git submodule update --init --recursive
cd csrc/fastvideo_kernel
pip install .
```

## Usage

```python
from fastvideo_kernel import sliding_tile_attention, video_sparse_attn
```

## Requirements

- H100 GPU (sm_90a) for CUDA kernels
- Triton for non-H100 fallback
