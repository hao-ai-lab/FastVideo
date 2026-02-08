# Stable Audio

FastVideo supports text-to-audio (T2A) generation via the **Stable Audio Open** model from Stability AI. This page describes supported models, installation, usage, and known limitations.

## Supported Models and Weights

| Model | HuggingFace ID | Local Path |
|-------|----------------|------------|
| Stable Audio Open 1.0 | `stabilityai/stable-audio-open-1.0` | `official_weights/stable-audio-open-1.0` |

**Weight format**:

- **HuggingFace**: Pass the model ID (e.g. `stabilityai/stable-audio-open-1.0`) to `VideoGenerator.from_pretrained()`. FastVideo will download and cache the model on first use.
- **Local**: Place a unified checkpoint (`model.safetensors` or `model.ckpt`) and `model_config.json` at the model root. Use the directory path as `model_path`.

## Installation and Dependencies

### Conflict with `stable-audio-tools` (Python 3.12)

The `stable-audio-tools` PyPI package has dependencies that **fail to build on Python 3.12** (e.g. PyWavelets). Do **not** run `pip install stable-audio-tools` directly.

Use the following two-step install:

```bash
# 1. Install stable-audio-tools without its dependencies
pip install stable-audio-tools --no-deps

# 2. Install compatible inference dependencies
pip install .[stable-audio]
# or: pip install k-diffusion v-diffusion-pytorch prefigure ema-pytorch local-attention alias-free-torch
```

If FastVideo is already installed:

```bash
pip install stable-audio-tools --no-deps
pip install fastvideo[stable-audio]
```

### Dependencies Installed by `[stable-audio]`

- `k-diffusion>=0.1.1`
- `v-diffusion-pytorch>=0.0.2`
- `prefigure>=0.0.9`
- `ema-pytorch>=0.2.3`
- `local-attention>=1.8.6`
- `alias-free-torch>=0.0.6`

These versions are compatible with FastVideo. `stable-audio-tools` declares stricter pins; the `--no-deps` install avoids pulling in conflicting packages (PyWavelets, encodec, etc.) that are not required for inference.

## Running the Example

### Basic usage

```bash
python examples/inference/basic/stable_audio_basic.py
```

### With custom parameters

```bash
python examples/inference/basic/stable_audio_basic.py \
  --prompt "A gentle rain on a wooden roof" \
  --duration 10 \
  --steps 250 \
  --output my_audio.wav
```

### Main parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `stabilityai/stable-audio-open-1.0` | Model path or HuggingFace model ID |
| `--prompt` | `A beautiful piano arpeggio` | Text description of the audio to generate |
| `--duration` | `10.0` | Output duration in seconds |
| `--output` | `outputs_audio/stable_audio_output.wav` | Output WAV file path |
| `--steps` | `250` | Number of denoising steps (`num_inference_steps`) |
| `--guidance-scale` | `6.0` | Classifier-free guidance scale |
| `--seed` | `42` | Random seed |
| `--no-cpu-offload` | (flag) | Disable CPU offload for higher GPU utilization (requires more VRAM) |

### Programmatic usage

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    num_gpus=1,
)

result = generator.generate_audio(
    prompt="A beautiful piano arpeggio",
    duration_seconds=10.0,
    num_inference_steps=250,
    guidance_scale=6.0,
    seed=42,
)

# result["audio"]: torch.Tensor (B, C, T)
# result["sample_rate"]: 44100
generator.shutdown()
```

### Sampling parameters (`generate_audio` kwargs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration_seconds` | `10.0` | Output duration (seconds) |
| `num_inference_steps` | `250` | Denoising steps |
| `guidance_scale` | `6.0` | CFG scale |
| `seed` | `42` | Random seed |
| `seconds_start` | `0.0` | Conditioning start offset |
| `seconds_total` | Same as `duration_seconds` | Conditioning total duration |

`sample_rate` is fixed at **44.1 kHz** and comes from the pipeline config.

## Known Limitations

| Item | Description |
|------|-------------|
| **T2A only** | Only text-to-audio is supported. Audio-to-audio, stem separation, and other stable-audio-tools features are not implemented. |
| **Single model** | Only Stable Audio Open 1.0 is supported. |
| **VRAM** | ~6–8 GB for typical generation (10 s, 250 steps). Use `--no-cpu-offload` for higher GPU utilization; this increases VRAM use. |
| **Max duration** | ~47.5 s at 44.1 kHz (model `sample_size` limit). |
| **Differences from official** | Uses FastVideo’s pipeline layout and executor; sampling logic matches stable-audio-tools (k-diffusion v-prediction, DPM++ 2M SDE). Minor numerical differences may occur due to implementation details. |
