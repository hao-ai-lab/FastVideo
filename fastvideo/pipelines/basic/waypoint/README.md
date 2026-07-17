# Waypoint-1-Small

FastVideo supports [Overworld Waypoint-1-Small](https://huggingface.co/Overworld/Waypoint-1-Small), a text- and control-conditioned autoregressive world model. It produces 360×640 frames for 60 FPS playback.

## Basic inference

Run [`examples/inference/basic/basic_waypoint.py`](../../../../examples/inference/basic/basic_waypoint.py) for one-shot generation through `VideoGenerator`.

## Streaming inference

```python
import torch

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator

generator = StreamingVideoGenerator.from_pretrained(
    "FastVideo/Waypoint-1-Small-Diffusers",
    num_gpus=1,
    use_fsdp_inference=False,
    dit_cpu_offload=True,
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=True,
)
generator.reset(
    prompt="A first-person view walking through a medieval village.",
    num_frames=240,
    height=360,
    width=640,
    output_path="waypoint.mp4",
)

keyboard = torch.zeros(1, 16, 256)
keyboard[:, :, 17] = 1  # W
mouse = torch.zeros(1, 16, 2)
scroll = torch.zeros(1, 16)

frames, _ = generator.step(keyboard, mouse, scroll)
generator.finalize()
generator.shutdown()
```

The control shapes are:

- `keyboard_cond`: `[batch, frames, 256]`
- `mouse_cond`: `[batch, frames, 2]`
- `scroll_cond`: `[batch, frames]` or `[batch, frames, 1]`; values are reduced to their sign

An optional `image_path` passed to `reset` seeds the first cache frame. Waypoint uses the fixed sigma schedule `[1.0, 0.8611363, 0.7293324, 0.3207127, 0.0]`.

## Validation

- `tests/local_tests/waypoint/test_waypoint_transformer.py`: official checkpoint and autoregressive trajectory parity
- `tests/local_tests/waypoint/test_waypoint_kv_cache_parity.py`: local/global ring-cache parity
- `tests/local_tests/vaes/test_world_engine_vae_parity.py`: VAE parity
- `tests/local_tests/pipelines/test_waypoint_pipeline_smoke.py`: pipeline and control routing
