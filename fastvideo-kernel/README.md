# FastVideo Kernel

CUDA kernels for FastVideo video generation.

## Kernel inventory

Compiled CUDA extensions (CMake, see the build summary printed at the end of every configure):

| Extension | Kernels | Sources | GPU arch | Build gate |
|---|---|---|---|---|
| `fastvideo_kernel._C.fastvideo_kernel_ops` | TurboDiffusion INT8 GEMM, quant, RMSNorm, LayerNorm | `csrc/turbodiffusion/` | every arch in `TORCH_CUDA_ARCH_LIST` | always built |
| same extension, optional part | ThunderKittens sliding-tile attention (`sta_fwd`) and VSA block-sparse (`block_sparse_fwd/bwd`) | `csrc/attention/*_h100.cu` | Hopper `sm_90a` only | `FASTVIDEO_KERNEL_BUILD_TK` (AUTO = ON iff `9.0a` is in the arch list; always OFF on aarch64 hosts — TK headers don't compile there) |
| `fp4attn_cuda`, `fp4quant_cuda` | FP4 attention + quantization ("attn_qat_infer", modified SageAttention3) | `attn_qat_infer/` | consumer Blackwell `sm_120a` only, CUDA ≥ 12.8 | `FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER` (AUTO = ON iff `12.0a` is in the arch list) |

Runtime-JIT kernels (no build step, ship in every wheel/image):

| Kernels | Where | Used when |
|---|---|---|
| Triton: STA, VSA block-sparse, SLA, fused compress+topk, FP4 QAT training, quant/norm utils | `python/fastvideo_kernel/triton_kernels/` | automatic fallback when the matching C++ op is absent (`ops.py`, `turbodiffusion_ops.py`) |
| FA4 CuTe-DSL block-sparse (VSA-256 fastpath on `sm_100`) | `block_sparse_attn_cute_fwd.py` | optional `flash_attn.cute` dependency, see below |
| VMoBA `moba_attn_varlen` | `vmoba.py` | wraps flash-attn varlen |

## What gets built where, and when

| Surface | Trigger | Leg | `TORCH_CUDA_ARCH_LIST` | TK | FP4 |
|---|---|---|---|---|---|
| PyPI wheels (`.github/workflows/publish-kernel.yml`) | version bump in `fastvideo-kernel/pyproject.toml` on main, or manual dispatch | x86_64 cu126 | `9.0a` | ON | — (CUDA < 12.8) |
| | | x86_64 cu130 | `9.0a;12.0a` | ON | ON |
| | | aarch64 cu130 | `10.0a;12.0a` | — | ON |
| Docker images `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev` (`.github/workflows/infra-build-image.yml`) | `docker/Dockerfile` changes on main, or manual dispatch | amd64 cuda12.6.3 + cuda13.0.0 | `9.0a` | ON | — |
| | | arm64 cuda12.6.3 (GH200) | `9.0a` | — (aarch64) | — |
| | | arm64 cuda13.0.0 (GB10 / DGX Spark) | `12.1` | — | — |
| Local `./build.sh` | manual | probes the visible GPU via torch | detected | ON iff sm_90 (non-aarch64 host) | ON iff sm_120 |

Notes:

- No Docker image ships the FP4 kernels; only the x86_64/aarch64 cu130 wheels do.
- On arm64 images (GH200 included) STA/VSA run on the Triton fallbacks, since TK never builds on aarch64.
- Kernel tests run on Buildkite GPU CI for PRs touching `fastvideo-kernel/**` (see `.buildkite/pipeline.yml`).

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

### Optional: FA4 CuTe block-sparse backend

The NVIDIA Blackwell / sm_100 VSA fastpath routes to FastVideo's in-tree
FlashAttention-4 CuTe-DSL source fork, exposed as `flash_attn.cute`. This is an
**optional** dependency: it is imported lazily, and `video_sparse_attn`
transparently falls back to the Triton backend when it is absent (so the package
is fully usable without it).

The fork lives in `fastvideo-kernel/fa4`, retains the `flash-attn-4` distribution
name, and records its exact upstream revision and refresh procedure in
`fa4/UPSTREAM.md`. Install it editable while developing the kernels:

```bash
uv pip install --no-deps --editable fastvideo-kernel/fa4
```

The production VSA-256 entrypoint keeps its existing logical Q256/KV256 mask.
When the CuTe backend is enabled, an optional shape override expands that mask
to the finer physical schedules without changing selected token pairs:

```bash
FASTVIDEO_VSA_CUTEDSL=1 FASTVIDEO_VSA_FA4_BLOCK_SHAPE=128x64 ...
# At the original Q256 API boundary, logical Q64 children are safely paired
# onto the optimized physical Q128/KV64 schedule:
FASTVIDEO_VSA_CUTEDSL=1 FASTVIDEO_VSA_FA4_BLOCK_SHAPE=64x64 ...
```

The default `FASTVIDEO_VSA_FA4_BLOCK_SHAPE=256x256` preserves the historical
route (its logical KV256 edges are implemented with physical KV128 tiles).
Direct callers with arbitrary, distinct Q64 masks still use the native
Q64/KV64 kernel. The Q128/KV64 path enables score/probability double buffering
by default; set `FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER=0` only to run its retained
legacy schedule for comparison or rollback.

The CuTe kernel JIT-compiles on first use. Verified on Blackwell (sm_100) against
`tests/test_fa4_vsa_block_shapes.py` and `tests/test_vsa256_forward*.py`.

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

### Attn-QAT training

The default shape matches one sequence-parallel rank of the 4-GPU
Wan2.1-T2V-1.3B MixKit recipe (`B=1, H=3, L=31200, D=128`):

```bash
cd fastvideo-kernel
python benchmarks/benchmark_attn_qat_train.py
```

The benchmark reports both conventional attention FLOPs and the extra matrix
multiplications executed by the QAT straight-through path. Override
`--peak-tflops` when running on a GPU other than RTX 5090.

The QAT kernel is entirely Triton and routes by architecture at runtime. SM100
uses a large-tile forward and split 64x64 backward for the production
non-causal, head-dimension-128 configuration with a 16-aligned KV length. SM120
(including RTX 5090) keeps the previous tiling but joins the quantized and STE
P@V operations and uses a shallower backward pipeline for long sequences. Set
`FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV=0` to compare against the split P@V path.
Unsupported configurations retain the previous implementation. Set
`FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED=0` to benchmark that previous path on SM100. Forward tuning is available through
`FASTVIDEO_ATTN_QAT_FWD_MODE=fast|balanced|reference`; exact reference-order
softmax statistics are controlled by `FASTVIDEO_ATTN_QAT_FWD_EXACT_M` and are
disabled by default for maximum throughput. Set it to `1` for reference-order
statistics and bitwise-compatible `dV`.

### VSA (block-sparse) TFLOPs

For the GB200 FA4 comparison, install the in-tree source fork and run the
issue-#4554-derived harness. Its default `exact256` mask mode preserves the
same selected token pairs across Q×KV block shapes, reports sparse-aware
TFLOP/s, MFU against the 2.5-PFLOP/s dense-BF16 GB200 peak, and efficiency
relative to the raw FA4 256×256 path:

```bash
uv pip install --no-deps --editable fastvideo-kernel/fa4
CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ENABLE_TVM_FFI=1 \
  uv run --no-sync python fastvideo-kernel/benchmarks/bench_vsa_blackwell.py \
  --seq_lens 32768 --sparsities dense 90 \
  --block_shapes 256x256 128x64 64x64
```

To exercise every GPU in a four-GB200 tray, launch one independent replica per
device. Each rank writes its own suffixed JSON (for example,
`/tmp/fa4_tray.rank0.json`):

```bash
CUTE_DSL_ENABLE_TVM_FFI=1 uv run --no-sync torchrun --standalone --nproc-per-node=4 \
  fastvideo-kernel/benchmarks/bench_vsa_blackwell.py \
  --seq_lens 32768 --sparsities dense 90 \
  --block_shapes 256x256 128x64 64x64 --out /tmp/fa4_tray.json
```

The older generic benchmark below measures FastVideo's 64-token block-sparse
wrapper; it does not exercise the FA4 fine-grained source fork:

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
