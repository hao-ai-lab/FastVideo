# Waypoint-1-Small Pipeline

This directory contains the FastVideo implementation of the [Overworld Waypoint-1-Small](https://huggingface.co/Overworld/Waypoint-1-Small) interactive world model.

## Overview

Waypoint-1-Small is a 2.3 billion parameter control-and-text-conditioned causal diffusion model designed for real-time world generation. It generates video frames at 60 FPS at 360p resolution, conditioned on:

- **Text prompts**: Describes the scene or world to generate
- **Controller inputs**: Mouse velocity, keyboard buttons, and scroll wheel

## Architecture

The model consists of:

1. **Transformer (WaypointWorldModel)**: A 22-layer DiT with:
   - 2560 hidden dimension
   - 40 attention heads (20 KV heads with GQA)
   - Causal attention for autoregressive generation
   - Local/global attention windows for efficiency
   - Control conditioning via MLP fusion

2. **VAE (WorldEngineVAE)**: DCAE-based encoder/decoder
   - 16 latent channels
   - 16x spatial downsampling

3. **Text Encoder**: UMT5-XL from Google
   - 2048-dimensional embeddings
   - Multilingual support

## Usage

### Basic Streaming Inference

```python
from fastvideo.pipelines.basic.waypoint import WaypointPipeline
from fastvideo.pipelines.basic.waypoint.waypoint_pipeline import CtrlInput

# Initialize pipeline
pipeline = WaypointPipeline.from_pretrained(
    "Overworld/Waypoint-1-Small",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# Reset with a prompt
batch = ForwardBatch(prompt="A medieval castle on a hill")
pipeline.streaming_reset(batch, fastvideo_args)

# Generate frames with control inputs
ctrl = CtrlInput(
    button={17, 32},  # W + D keys
    mouse=(0.5, 0.0),  # Move mouse right
    scroll=0,
)
result = pipeline.streaming_step(ctrl, fastvideo_args)
frames = result.output

# Clean up
pipeline.streaming_clear()
```

### Control Input Format

The `CtrlInput` dataclass accepts:

- `button`: Set of pressed button IDs (0-255) using Owl-Control keycodes
- `mouse`: Tuple of (x, y) mouse velocity as floats
- `scroll`: Scroll wheel value (-1, 0, or 1)

Common keycodes:
| Key | Keycode |
|-----|---------|
| W   | 17      |
| A   | 30      |
| S   | 31      |
| D   | 32      |
| Space | 57    |
| Left Mouse | 0 |
| Right Mouse | 1 |

## Configuration

The pipeline is configured via `WaypointT2VConfig`:

```python
from fastvideo.configs.pipelines.waypoint import WaypointT2VConfig

config = WaypointT2VConfig(
    is_causal=True,        # Autoregressive generation
    base_fps=60,           # Native frame rate
    n_buttons=256,         # Number of button inputs
    scheduler_sigmas=[     # Fixed denoising schedule
        1.0, 0.861, 0.729, 0.321, 0.0
    ],
)
```

## Files

- `waypoint_pipeline.py`: Main pipeline implementation with streaming interface
- `__init__.py`: Module exports

## Related Files

- `fastvideo/models/dits/waypoint_transformer.py`: Transformer implementation
- `fastvideo/configs/models/dits/waypoint_transformer.py`: Transformer config
- `fastvideo/configs/pipelines/waypoint.py`: Pipeline config
- `fastvideo/configs/sample/waypoint.py`: Sampling parameters
- `tests/local_tests/transformers/test_waypoint_transformer.py`: Parity tests
- `tests/local_tests/pipelines/test_waypoint_pipeline_smoke.py`: Pipeline tests
- `examples/inference/basic/basic_waypoint_streaming.py`: Example script

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 24GB VRAM (e.g., RTX 4090)
- **Recommended**: NVIDIA GPU with 80GB VRAM (e.g., A100/H100) for full-speed inference

## References

- [Hugging Face Model Card](https://huggingface.co/Overworld/Waypoint-1-Small)
- [World Engine Repository](https://github.com/Overworldai/world_engine)
- [Overworld Website](https://www.overworld.ai/)

