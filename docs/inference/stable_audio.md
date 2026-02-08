# Stable Audio

Text-to-audio with **Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`). Weights are loaded from HuggingFace or a local path (unified `model.safetensors` + `model_config.json`).

## Setup

**1. HuggingFace login** (accept model terms on the Hub if required):

```bash
hf auth login
```

**2. Install deps** (k-diffusion, alias-free-torch, einops-exts):

```bash
pip install .[stable-audio]
```

## Usage

**CLI:**

```bash
python examples/inference/basic/stable_audio_basic.py
python examples/inference/basic/stable_audio_basic.py --prompt "A gentle rain" --duration 8 --output out.wav
```

**Python:**

```python
from fastvideo import VideoGenerator

gen = VideoGenerator.from_pretrained("stabilityai/stable-audio-open-1.0", num_gpus=1)
out = gen.generate_audio(prompt="A beautiful piano arpeggio", duration_seconds=10.0)
# out["audio"]: (B, C, T), out["sample_rate"]: 44100
gen.shutdown()
```

Optional: `--no-cpu-offload` for higher GPU use (more VRAM). Max duration ~47 s at 44.1 kHz.
