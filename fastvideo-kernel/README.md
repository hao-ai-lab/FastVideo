# FastVideo Kernel

CUDA kernels for FastVideo video generation.

## Installation

### Standard Installation (Local Development)
This will automatically detect your GPU architecture. If an NVIDIA Hopper (H100/sm_90a) GPU is detected, ThunderKittens kernels will be enabled. Otherwise, they will be skipped, and the package will use Triton fallbacks at runtime.

Before installation, set CUDA toolchain paths:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc
```

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

### Optional: FA4 CuTe block-sparse backend (VSA-256 fastpath)

The VSA-256 fastpath (tile volume 256, on NVIDIA Blackwell / sm_100) routes to the
FlashAttention-4 CuTe-DSL block-sparse kernel exposed as `flash_attn.cute`. This is
an **optional** dependency: it is imported lazily, and `video_sparse_attn`
transparently falls back to the Triton backend when it is absent (so the package is
fully usable without it).

The symbols the fastpath needs (`flash_attn.cute.block_sparsity.BlockSparseTensorsTorch`,
`flash_attn.cute.interface._flash_attn_fwd`) are provided upstream by
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). Pin to
commit `940cd9680f3315f2f06b43ab5bea2c2cf2d96806` — the cutlass-4.5-safe revision
FastVideo itself pins (as the `flash-attn-4` source in the repo-root
`pyproject.toml`). The wrapper targets that revision's `_flash_attn_fwd`
signature (block sizes passed as a `tile_mn` tuple); older revisions that take
`m_block_size` / `n_block_size` (upstream `c19cd20e`, the `fa4-compile` fork)
are not compatible.

```bash
pip install "nvidia-cutlass-dsl>=4.5.0" torchvision
pip install "git+https://github.com/Dao-AILab/flash-attention.git@940cd9680f3315f2f06b43ab5bea2c2cf2d96806#subdirectory=flash_attn/cute"
```

The CuTe kernel JIT-compiles on first use. Verified on Blackwell (sm_100) against
`tests/test_vsa256_forward*.py`.

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
  - `CUDA_HOME` must be set (for example, `/usr/local/cuda`)
  - `CUDACXX` must be set (for example, `$CUDA_HOME/bin/nvcc`)
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
