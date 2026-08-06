# 🚀 Quick Start

Get up and running with FastVideo in minutes!

## Installation

First, install FastVideo:

```bash
# If you previously used Conda, use uv instead for a faster, more stable setup
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install FastVideo on NVIDIA CUDA 12
UV_TORCH_BACKEND=cu126 uv pip install fastvideo
```

Use `UV_TORCH_BACKEND=cu130` instead on CUDA 13.

Also optionally install flash-attn:

```bash
uv pip install flash-attn --no-build-isolation -v
```

## ⚡ Generate Your Command

Select your task, model profile, and GPU count to get a ready-to-run command:

<div class="quick-start-guide-wrap">
  <iframe
    class="quick-start-guide-frame"
    src="../../config-generator/"
    title="FastVideo Guided Config Generator"
    data-config-generator
    loading="lazy"
  ></iframe>
</div>

!!! tip "Need more control?"
    Use the [Advanced Tuning Guide](advanced_tuning_guide.md) to tune all parameters — resolution, attention backend, memory offloading, and more.

## Next Steps

- [Advanced Tuning Guide](advanced_tuning_guide.md) - Fine-grained parameter tuning
- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore more
  examples
- [Optimizations](../inference/optimizations.md) - Performance optimization tips
