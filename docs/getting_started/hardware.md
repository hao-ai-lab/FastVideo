# Hardware and runtimes

Choose your runtime first, then check the model and attention backend. A model
being listed does not mean every precision, kernel, or device supports it.

| Hardware | Runtime | Start here | What changes |
| --- | --- | --- | --- |
| NVIDIA RTX / workstation GPUs | PyTorch + CUDA | [NVIDIA GPU setup](installation/gpu.md), then the model's cookbook | Memory budget and GPU architecture determine which precision and attention kernels are available. The [RTX PRO 6000 FastH3 guide](../inference/fasth3_rtx_pro_6000.md) is a specific Blackwell recipe, not a requirement for all NVIDIA users. |
| NVIDIA data center GPUs | PyTorch + CUDA | [NVIDIA GPU setup](installation/gpu.md), then [multi-GPU configuration](../inference/configuration.md#multi-gpu-setup) | Use the configuration and optimization guides for parallelism, offloading and backend selection. Keep benchmark results tied to the measured GPU. |
| NVIDIA DGX Spark / GB10 | PyTorch + CUDA on ARM64 | [Spark installation](installation/spark.md) | This is a separate installation path. [Pairing two Sparks](installation/spark_pair.md) and [tuning](installation/spark_performance.md) are advanced steps after one device works. |
| Apple Silicon | Native MLX | [MLX installation](installation/mlx.md) | Use MLX-compatible checkpoints and examples. FastMetal and FastH3 have different conversion and generation paths. The older MPS example is a separate PyTorch demo. |
| AMD GPUs | PyTorch + ROCm | [ROCm support and current limits](installation/amd.md) | The repository has a ROCm platform and Triton attention routes. CUDA-specific kernels and quantized exports do not automatically apply. |

## Then choose a model

[Fast Models](https://haoailab.com/FastVideo/fast-models/) highlights FastVideo releases.
The [support matrix](../inference/support_matrix.md) covers the wider model set.
The [cookbook](../cookbook/index.md) provides model-specific recipes and runtime
choices. Follow a recipe's recorded requirements rather than inferring memory
fit from a model name or parameter count.

## Shared guides

- [Configuration](../inference/configuration.md) owns generation settings and parallelism.
- [Offloading](../inference/offloading.md) explains memory tradeoffs.
- [Optimizations](../inference/optimizations.md) lists attention and precision constraints.
- [ComfyUI](../inference/comfyui.md) is an integration, not a hardware platform.
