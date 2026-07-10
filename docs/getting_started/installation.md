
# 🔧 Installation

FastVideo supports the following hardware platforms:

- [NVIDIA CUDA](installation/gpu.md)
- [NVIDIA DGX Spark / GB10 (ARM64 + CUDA 13)](installation/spark.md)
- [Apple silicon](installation/mps.md)

## Quick Installation

### Using uv (recommended)

Use uv as the default environment manager for faster and more stable installs.
The commands below target NVIDIA CUDA 12; use `UV_TORCH_BACKEND=cu130` on
CUDA 13. Apple silicon users should follow the [MPS guide](installation/mps.md).

```bash
# Create and activate a new uv environment
uv venv --python 3.12 --seed
source .venv/bin/activate

UV_TORCH_BACKEND=cu126 uv pip install fastvideo
```

### Using Conda (alternative)

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12 -y
conda activate fastvideo

UV_TORCH_BACKEND=cu126 uv pip install fastvideo
```

### From source

FastVideo pins PyTorch 2.12.0. Select its CUDA build explicitly with
`UV_TORCH_BACKEND=cu126` for CUDA 12 or `UV_TORCH_BACKEND=cu130` for CUDA 13.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
UV_TORCH_BACKEND=cu126 uv pip install -e .

# optional: install flash-attn
uv pip install flash-attn --no-build-isolation -v
```

Alternative with Conda environment (still drives installs through `uv`):

```bash
UV_TORCH_BACKEND=cu126 uv pip install -e .
uv pip install flash-attn --no-build-isolation -v
```

## Hardware Requirements

- **NVIDIA GPUs**: CUDA 12.6+ with compute capability 7.0+
- **Apple Silicon**: macOS 14.0+ with M1/M2/M3/M4 chips
- **CPU**: x86_64 architecture (for CPU-only inference)

## Next Steps

- [Quick Start Guide](quick_start.md) - Get started with your first video generation
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore example scripts and notebooks
