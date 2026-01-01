
# 🔧 Installation

FastVideo supports the following hardware platforms:

- [NVIDIA CUDA](installation/gpu.md)
- [Apple silicon](installation/mps.md)

## Quick Installation

### Using pip

```bash
pip install fastvideo
```

### Using conda

```bash
conda install -c conda-forge fastvideo
```

### From source

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
pip install -e .
```

## Hardware Requirements

- **NVIDIA GPUs**: CUDA 11.8+ with compute capability 7.0+
- **Apple Silicon**: macOS 12.0+ with M1/M2/M3 chips
- **CPU**: x86_64 architecture (for CPU-only inference)

## Next Steps

- [Quick Start Guide](quick_start.md) - Get started with your first video generation
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore example scripts and notebooks
