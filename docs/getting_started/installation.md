
# 🔧 Installation

FastVideo supports the following hardware platforms:

- [NVIDIA CUDA](installation/gpu.md)
- [Apple silicon](installation/mps.md)

## Quick Installation

### Using pip

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12
conda activate fastvideo

pip install fastvideo
```

### Using uv

```bash
# Create and activate a new uv environment
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install fastvideo
```

### From source

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
pip install -e .

# or if you are using uv
uv pip install -e .
```

Also optionally install flash-attn:

```bash
pip install flash-attn --no-build-isolation
```

## Hardware Requirements

- **NVIDIA GPUs**: CUDA 11.8+ with compute capability 7.0+
- **Apple Silicon**: macOS 12.0+ with M1/M2/M3 chips
- **CPU**: x86_64 architecture (for CPU-only inference)

## Next Steps

- [Quick Start Guide](quick_start.md) - Get started with your first video generation
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore example scripts and notebooks
