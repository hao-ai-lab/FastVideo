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

### Rocm Build
If you are in a rocm environment without the compilation toolchaine of CUDA.

```bash
cd fastvideo-kernel
./build.sh --rocm
```

## Usage

### Sliding Tile Attention (STA) & Video Sparse Attention (VSA)

For detailed usage, please check the [Attention Documentation](../docs/attention/index.md).

```python
from fastvideo_kernel import sliding_tile_attention, video_sparse_attn, moba_attn_varlen

# Example: Sliding Tile Attention
out = sliding_tile_attention(q, k, v, window_sizes, text_len)

# Example: Video Sparse Attention (with Triton fallback)
out = video_sparse_attn(q, k, v, block_sizes, block_sizes, topk=5)

# Example: VMoBA
out = moba_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
```

## Benchmark

### VSA (block-sparse) TFLOPs

After building/installing `fastvideo-kernel`, run:

```bash
cd fastvideo-kernel
python benchmarks/bench_vsa.py --batch_size 1 --num_heads 16 --head_dim 128 --q_seq_lens 49152 --topk 64
```

### TurboDiffusion Kernels

This package also includes kernels from [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion), including INT8 GEMM, Quantization, RMSNorm and LayerNorm.

## Requirements

- **Runtime**:
  - NVIDIA H100 (sm_90a) for C++ optimized kernels.
  - Any CUDA GPU for Triton-based fallbacks.
- **Build**:
  - CUDA Toolkit 12.3+
  - C++20 compatible compiler (GCC 10+, Clang 11+)

## Acknowledgement

This package structure and build system are based on [sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) from the SGLang project.

The implementation of `turbodiffusion` kernels is adapted from [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion). If you use these kernels, please cite:

```bibtex
@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
```
