# Quick start

Install FastVideo, generate one video, then start from a maintained recipe.

## Install

Requires Python 3.10 to 3.12 and CUDA 12.6 or newer.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# CUDA 12. Use cu130 on CUDA 13.
UV_TORCH_BACKEND=cu126 uv pip install fastvideo
```

For the fastest attention kernels, install flash-attn:

```bash
uv pip install flash-attn --no-build-isolation -v
```

## Generate a video

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    text_encoder_cpu_offload=True,
)
generator.generate_video(
    "A raccoon peers through a field of sunflowers.",
    output_path="videos/",
    save_video=True,
)
```

The first run downloads the Wan2.1 1.3B checkpoint. The call writes an MP4 under `videos/`. The full example is `examples/inference/basic/basic.py`.

## Choose a maintained recipe

The cookbook selects complete, checked-in recipes instead of mixing model,
parallelism, offload, and attention settings independently.

[Open the inference cookbook](../cookbook/index.md)

:::tip[Need more control?]
Start from a maintained recipe, then use the
[configuration](../inference/configuration.md) and
[optimization](../inference/optimizations.md) guides for supported changes.
:::

## Next Steps

- [Inference Cookbook](../cookbook/index.md) - Choose a maintained recipe
- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore more
  examples
- [Optimizations](../inference/optimizations.md) - Performance optimization tips
