# FastVideo Kernel

CUDA kernels for FastVideo video generation.

## Installation

### Standard Installation (Local Development)
This will automatically detect your GPU architecture. If an NVIDIA Hopper (H100/sm_90a) GPU is detected, ThunderKittens kernels will be enabled. Otherwise, they will be skipped, and the package will use Triton fallbacks at runtime.

```bash
git submodule update --init --recursive
cd fastvideo-kernel
./build.sh
```

### Release Build (Force Enable Kernels)
If you are building a release wheel or docker image on a machine without a GPU (e.g., CI/CD), you can force-enable the compilation of Hopper-specific ThunderKittens kernels.

```bash
cd fastvideo-kernel
./build.sh --release
```
*Note: The resulting wheel will contain kernels that require an H100 GPU to run, but can be built on any machine with CUDA 12.3+ toolchain.*

## Usage

```python
from fastvideo_kernel import sliding_tile_attention, video_sparse_attn, moba_attn_varlen

# Example: Sliding Tile Attention
out = sliding_tile_attention(q, k, v, window_sizes, text_len)

# Example: Video Sparse Attention (with Triton fallback)
out = video_sparse_attn(q, k, v, block_sizes, topk=5)

# Example: VMoBA
out = moba_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
```

## Requirements

- **Runtime**:
  - NVIDIA H100 (sm_90a) for C++ optimized kernels.
  - Any CUDA GPU for Triton-based fallbacks.
- **Build**:
  - CUDA Toolkit 12.3+
  - C++20 compatible compiler (GCC 10+, Clang 11+)

## Acknowledgement

This package structure and build system are based on [sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) from the SGLang project.
