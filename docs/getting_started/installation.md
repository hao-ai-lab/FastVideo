
# 🔧 Installation

FastVideo supports the following hardware platforms:

- [NVIDIA CUDA](installation/gpu.md)
- [Apple silicon](installation/mps.md)

## Quick Installation

### Using uv (recommended)

Use uv as the default environment manager for faster and more stable installs.

```bash
# Create and activate a new uv environment
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install fastvideo
```

### Using Conda (alternative)

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12 -y
conda activate fastvideo

pip install fastvideo
```

### From source

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv pip install -e .

# optional: install flash-attn
uv pip install flash-attn --no-build-isolation -v
```

Alternative with Conda environment:

```bash
pip install -e .
pip install flash-attn --no-build-isolation -v
```

## Hardware Requirements

- **NVIDIA GPUs**: CUDA 11.8+ with compute capability 7.0+
- **Apple Silicon**: macOS 12.0+ with M1/M2/M3 chips
- **CPU**: x86_64 architecture (for CPU-only inference)

## Next Steps

- [Quick Start Guide](quick_start.md) - Get started with your first video generation
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore example scripts and notebooks
