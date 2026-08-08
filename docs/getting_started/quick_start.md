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

## Choose a maintained recipe

The cookbook selects complete, checked-in recipes instead of mixing model,
parallelism, offload, and attention settings independently.

[Open the inference cookbook](../cookbook/index.md){ .md-button .md-button--primary }

!!! tip "Need more control?"
    Start from a maintained recipe, then use the
    [configuration](../inference/configuration.md) and
    [optimization](../inference/optimizations.md) guides for supported changes.

## Next Steps

- [Inference Cookbook](../cookbook/index.md) - Choose a maintained recipe
- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore more
  examples
- [Optimizations](../inference/optimizations.md) - Performance optimization tips
