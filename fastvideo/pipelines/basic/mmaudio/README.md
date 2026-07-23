# MMAudio Video-to-Audio

This directory contains FastVideo's native `MMAudioPipeline` implementation for
video-to-audio (V2A) and text-to-audio (T2A) generation. The first supported
checkpoint is `large_44k_v2`, which produces mono audio at 44.1 kHz.

The production pipeline is implemented with FastVideo components and does not
import the upstream `mmaudio` Python package. It loads the transformer, DFN5B
CLIP text/vision encoders, Synchformer visual encoder, audio VAE, BigVGAN-v2,
tokenizer, and scheduler from a standard FastVideo/Diffusers-style checkpoint
directory.

## Status

- `large_44k_v2` V2A inference: supported
- T2A pipeline routing: supported; the provided example currently targets V2A
- Output: mono WAV, 44.1 kHz
- Default inference duration: 8 seconds
- Variable-duration inference: supported
- Single-GPU inference with FastVideo's default offloading: supported
- `small_44k`, `medium_44k`, and `large_44k` v1 training: supported through
  FastVideo's modular trainer and official-compatible precomputed features
- Source-video/audio muxing: not yet part of the pipeline output
- `large_44k_v2` training: intentionally rejected because the upstream recipe
  does not support training the v2 checkpoint
- 16 kHz training/inference: not included in this port

The native pipeline has passed exact official-vs-FastVideo real-weight parity
for a 25-step two-second waveform and a real ten-second V2A inference smoke
test.

## Requirements

Follow FastVideo's main NVIDIA installation guide. The relevant baseline is:

- Linux or Windows WSL
- Python 3.10-3.12
- CUDA 12.6 or CUDA 13.0
- PyTorch 2.12.0
- One NVIDIA GPU

The port was validated with Python 3.12, PyTorch 2.12.0+cu126, and an RTX 6000
Ada 48 GB. This is a validated configuration, not a minimum VRAM claim. Default
layerwise/component offloading is enabled. FlashAttention is optional; the
pipeline falls back to Torch SDPA when FlashAttention is unavailable.

Install FastVideo from this source tree with `uv`:

```bash
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu126 uv pip install -e .
```

Training additionally needs TensorDict for the memory-mapped feature cache:

```bash
UV_TORCH_BACKEND=cu126 uv pip install -e '.[mmaudio-train]'
```

Use `UV_TORCH_BACKEND=cu130` instead on CUDA 13. Conda is not required.

## Checkpoint Layout

Model weights are intentionally excluded from the FastVideo Git repository.
For local development, the example expects a converted checkpoint at:

```text
converted_weights/mmaudio/large_44k_v2/
├── model_index.json
├── transformer/
├── text_encoder/
├── tokenizer/
├── image_encoder/
├── image_encoder_2/
├── audio_vae/
├── vocoder/
└── scheduler/
```

`official_weights/`, `converted_weights/`, and inference outputs are ignored by
Git. Cloning a code branch therefore does not clone the model.

### Pre-converted checkpoint

FastVideo accepts either a local directory or a Hugging Face model ID. When a
complete converted checkpoint is published, select it with:

```bash
export MMAUDIO_MODEL_PATH=ORG/MMAudio-large-44k-v2-Diffusers
```

FastVideo will then download the complete snapshot on first use and reuse the
Hugging Face cache on later runs. At the time of this port, the registered
`FastVideo/MMAudio-large-44k-v2-Diffusers` name is reserved but is not yet a
public checkpoint, so use the local conversion below.

### Convert the official weights locally

Only checkpoint conversion requires `open_clip_torch`; native FastVideo
inference does not depend on the upstream MMAudio package.

```bash
uv pip install open_clip_torch

mkdir -p official_weights/mmaudio/raw/weights
mkdir -p official_weights/mmaudio/raw/ext_weights
mkdir -p official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384
mkdir -p official_weights/mmaudio/bigvgan_v2_44khz_128band_512x
```

Download the three MMAudio assets:

```bash
curl -L --continue-at - \
  https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth \
  -o official_weights/mmaudio/raw/weights/mmaudio_large_44k_v2.pth

curl -L --continue-at - \
  https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth \
  -o official_weights/mmaudio/raw/ext_weights/v1-44.pth

curl -L --continue-at - \
  https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth \
  -o official_weights/mmaudio/raw/ext_weights/synchformer_state_dict.pth
```

Download only the DFN5B and BigVGAN files used by the converter:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="apple/DFN5B-CLIP-ViT-H-14-384",
    local_dir="official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384",
    allow_patterns=["open_clip_config.json", "open_clip_pytorch_model.bin"],
)
snapshot_download(
    repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
    local_dir="official_weights/mmaudio/bigvgan_v2_44khz_128band_512x",
    allow_patterns=["config.json", "bigvgan_generator.pt"],
)
PY
```

Convert the source assets into the component tree consumed by FastVideo:

```bash
python scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py \
  --transformer-checkpoint official_weights/mmaudio/raw/weights/mmaudio_large_44k_v2.pth \
  --audio-vae-checkpoint official_weights/mmaudio/raw/ext_weights/v1-44.pth \
  --synchformer-checkpoint official_weights/mmaudio/raw/ext_weights/synchformer_state_dict.pth \
  --dfn5b-dir official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384 \
  --bigvgan-dir official_weights/mmaudio/bigvgan_v2_44khz_128band_512x \
  --output converted_weights/mmaudio/large_44k_v2
```

The converted checkpoint is approximately 9 GB.

## Train MMAudio v1

Training uses the native `fastvideo/train/` stack. FastVideo owns the training
loop, HSDP/FSDP wrapping, optimizer and LR scheduler, gradient accumulation,
gradient clipping, tracking, and distributed checkpoints. The reusable
`FlowMatchingFineTuneMethod` implements shape-agnostic velocity supervision;
`MMAudioModel` supplies MMAudio's audio posterior sampling, latent
normalization, independent video/text CFG dropout, and multimodal forward.

The first smoke-test target should be `small_44k`. The upstream documentation
does not support training checkpoints ending in `_v2`, so the inference
checkpoint `large_44k_v2` must not be used here.

### Convert a trainable v1 transformer

Download the official v1 checkpoint:

```bash
mkdir -p official_weights/mmaudio/raw/weights
curl -L --continue-at - \
  https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_44k.pth \
  -o official_weights/mmaudio/raw/weights/mmaudio_small_44k.pth
```

Training only loads the transformer, so it does not need another copy of the
VAE, DFN5B, Synchformer, or vocoder components:

```bash
python scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py \
  --variant small_44k \
  --transformer-only \
  --transformer-checkpoint official_weights/mmaudio/raw/weights/mmaudio_small_44k.pth \
  --output converted_weights/mmaudio/small_44k
```

Replace `small_44k` with `medium_44k` or `large_44k` in both arguments to train
a larger v1 model.

### Precomputed feature contract

The online training step deliberately does not run the audio VAE, DFN5B, or
Synchformer. Point `training.data.data_path` at a TensorDict memory-mapped
directory with the same tensors produced by upstream MMAudio preprocessing:

```text
video + audio cache
├── mean             [N, 345, 40]
├── std              [N, 345, 40]
├── clip_features    [N,  64, 1024]
├── sync_features    [N, 192, 768]
└── text_features    [N,  77, 1024]

audio-only cache
├── mean             [N, 345, 40]
├── std              [N, 345, 40]
└── text_features    [N,  77, 1024]
```

Video and audio-only caches can be mixed by using a mapping of cache paths to
repeat counts in YAML. Missing video conditions are replaced by MMAudio's
learned null-video tokens. The cache loader has no runtime import dependency on
the upstream `mmaudio` package.

The example config intentionally contains an empty data path until a dataset is
prepared:

```text
examples/train/configs/fine_tuning/mmaudio/small_44k.yaml
```

Start a single-GPU run while supplying the cache path on the command line:

```bash
NUM_GPUS=1 bash examples/train/run.sh \
  examples/train/configs/fine_tuning/mmaudio/small_44k.yaml \
  --training.data.data_path /path/to/mmaudio-feature-cache
```

Leaving the path empty produces an explicit error instead of silently starting
with the wrong data. The single-GPU FSDP-wrapped path and distributed
checkpoint/resume have been validated with the real `small_44k` weights.
Multi-GPU data parallel remains to be validated; sequence and tensor
parallelism are not implemented, so keep `sp_size: 1` and `tp_size: 1`. The
example preserves the official learning rate, AdamW betas/epsilon, weight
decay, 1,000-step warmup, CFG dropout, logit-normal timestep sampling, and
300,000-step duration. Its per-GPU batch size is intentionally one for initial
smoke testing and should be scaled with gradient accumulation or additional
data-parallel GPUs.

## Run V2A Inference

The runnable example is
[`examples/inference/basic/basic_mmaudio.py`](../../../../examples/inference/basic/basic_mmaudio.py).
Run all commands from the FastVideo repository root.

```bash
export MMAUDIO_MODEL_PATH=converted_weights/mmaudio/large_44k_v2

python examples/inference/basic/basic_mmaudio.py \
  --video-path /path/to/input.mp4 \
  --duration-seconds 8 \
  --prompt "A skateboarder rolls over concrete and lands on a metal rail." \
  --negative-prompt "music, speech" \
  --output-path outputs_audio/mmaudio.wav
```

To select a GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 \
MMAUDIO_MODEL_PATH=converted_weights/mmaudio/large_44k_v2 \
python examples/inference/basic/basic_mmaudio.py \
  --video-path /path/to/input.mp4 \
  --duration-seconds 10 \
  --output-path outputs_audio/mmaudio_10s.wav
```

The text prompt is optional, but a short description of audible events usually
provides better control. The negative prompt can suppress unwanted categories
such as music or speech.

Eight seconds is the published training/default duration, not a hard inference
limit. Shorter and longer clips use dynamic sequence lengths. As in the
official MMAudio demo, quality may decrease when the requested duration is far
from eight seconds. If the source video is shorter than the requested duration,
the pipeline uses the available decoded duration. Synchformer requires at least
16 frames at 25 FPS, so V2A input must cover at least 0.64 seconds.

## Listen with the Source Video

The pipeline currently writes a WAV file. To create a preview MP4 that replaces
the source audio with the generated waveform:

```bash
ffmpeg -y \
  -i /path/to/input.mp4 \
  -i outputs_audio/mmaudio.wav \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -shortest \
  outputs_audio/mmaudio_preview.mp4
```

## Troubleshooting

- **Model download fails:** verify that `MMAUDIO_MODEL_PATH` is an existing
  local converted directory or a public Hugging Face repository containing
  `model_index.json` and all eight components.
- **`FlashAttention-2 ... not found`:** this is informational. Torch SDPA is a
  supported fallback and was used for exact parity validation.
- **The original video already has audio:** V2A conditioning reads video frames
  only. The source audio is not passed into MMAudio.
- **No MP4 is returned:** the native result is audio-only by design; use the
  `ffmpeg` command above for a preview mux.
- **Do not commit checkpoints:** never force-add `official_weights/` or
  `converted_weights/` to the FastVideo Git repository.

## License and Attribution

The upstream MMAudio checkpoint is distributed as CC-BY-NC 4.0. The native
audio VAE implementation also contains EDM2-derived code marked
CC-BY-NC-SA-4.0. Review and preserve the upstream licenses and attribution
before redistributing raw or converted checkpoints.
