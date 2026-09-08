---
license: apache-2.0
---

# FastVideo Kernel

Hub-compatible FastVideo CUDA kernels packaged with Hugging Face `kernel-builder`.

## Usage

```python
from kernels import get_kernel

fastvideo_kernel = get_kernel("hao-ai-lab/fastvideo-kernel", version=1)
```

The module exposes:

- `sta_fwd`
- `block_sparse_fwd`
- `block_sparse_bwd`
- `rms_norm`
- `layer_norm`
- `int8_quant`
- `int8_gemm`

These APIs are thin wrappers around the native FastVideo kernels. Higher-level
fallback logic remains in the regular `fastvideo-kernel` Python package.

## Requirements

This first Hub build targets CUDA Hopper (`sm_90a`) because the native attention
kernels rely on Hopper/ThunderKittens features.

## Source

Source repository: https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo-kernel
