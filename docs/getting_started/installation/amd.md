# AMD and ROCm

FastVideo has a ROCm platform implementation for AMD GPUs. This page describes
the checked-in support paths; a complete device-by-model validation matrix and
end-to-end installation walkthrough are not currently documented.

## What is implemented

- ROCm platform detection uses AMD SMI or a HIP-enabled PyTorch build with
  access to `/dev/kfd` and DRM render devices.
- The platform supports PyTorch SDPA and selects FlashAttention when its
  installation, dtype and head size are compatible, with SDPA fallback.
- `VIDEO_SPARSE_ATTN` and `VIDEO_SPARSE_ATTN_H3` use the kernel package's Triton
  routes. CUDA CuTe and ThunderKittens paths are not ROCm implementations.
- The optimization guide documents a CDNA4-specific
  [FP8 weight path](../../inference/optimizations.md#fp8-weight-quantization).
  This does not establish FP8 support on every AMD device.

These paths are defined in the
[ROCm platform source](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/platforms/rocm.py)
and [platform detection](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/platforms/__init__.py).

## Setup references

The repository includes a
[ROCm Dockerfile](https://github.com/hao-ai-lab/FastVideo/blob/main/docker/rocm7_1.Dockerfile)
and [kernel build instructions](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/README.md#rocm-build).
The kernel build supports `./build.sh --rocm` and a Python/Triton-only mode when
the compiled HIP extension is unavailable.

Use a matching ROCm-enabled PyTorch environment. The ordinary CUDA installation
command is not an AMD installation command. Check the
[support matrix](../../inference/support_matrix.md) and the selected model's
[cookbook](../../cookbook/index.md) for its runtime requirements. A CUDA recipe
or released NVFP4 checkpoint alone is not evidence of an AMD-compatible run.

## Documentation still needed

A maintained ROCm walkthrough should record the GPU architecture, ROCm/PyTorch
versions, model revision, attention backend, runnable command, output, and measured
memory. Until that evidence is published, device and model combinations beyond
the linked source paths remain unverified here.
