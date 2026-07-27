# MMAudio Video-to-Audio

This directory contains FastVideo's native `MMAudioPipeline` implementation for
video-to-audio (V2A) and text-to-audio (T2A) generation. All five published
MMAudio model architectures are registered independently so their sequence
length, latent width, sample rate, VAE, and vocoder cannot be mixed silently.

The production pipeline is implemented with FastVideo components and does not
import the upstream `mmaudio` Python package. It loads the transformer, DFN5B
CLIP text/vision encoders, Synchformer visual encoder, audio VAE, BigVGAN-v2,
tokenizer, and scheduler from a standard FastVideo/Diffusers-style checkpoint
directory.

## Official model variants

| Variant | Audio | Hidden / layers / heads | Inference | Official training recipe |
| --- | ---: | --- | --- | --- |
| `small_16k` | 16 kHz | 448 / 12 / 7 | implemented | supported |
| `small_44k` | 44.1 kHz | 448 / 12 / 7 | implemented | supported |
| `medium_44k` | 44.1 kHz | 896 / 12 / 14 | implemented | supported |
| `large_44k` | 44.1 kHz | 896 / 21 / 14 | implemented | supported |
| `large_44k_v2` | 44.1 kHz | 896 / 21 / 14 | implemented and validated | not supported upstream |

`small_16k` uses latent shape `[250, 20]`, the 16 kHz VAE, and the original
80-mel BigVGAN. The 44.1 kHz variants use latent shape `[345, 40]`, the 44 kHz
VAE, and 128-mel BigVGAN-v2. `_v2` changes transformer activations and timestep
conditioning but not the large model dimensions.

## Status

- Five variant-specific presets, pipeline configs, registry entries, and
  converter choices: supported
- T2A pipeline routing: supported; the provided example currently targets V2A
- Output: mono WAV at 16 kHz or 44.1 kHz, according to the checkpoint
- Default inference duration: 8 seconds
- Variable-duration inference: supported
- Single-GPU inference with FastVideo's default offloading: supported
- `small_16k`, `small_44k`, `medium_44k`, and `large_44k` from-scratch training:
  supported through FastVideo's modular trainer and official-compatible features
- Source-video/audio muxing: not yet part of the pipeline output
- `large_44k_v2` training: intentionally rejected because the upstream recipe
  does not support training the v2 checkpoint

The native pipeline has passed exact official-vs-FastVideo real-weight parity
for a 25-step two-second waveform and a real ten-second V2A inference smoke
test with `large_44k_v2`. The other four variants have architecture/config
coverage; real-weight waveform parity still requires converting their official
checkpoints.

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
a larger v1 model. `small_16k` is also supported, but requires 16 kHz features.

### Train the DiT from scratch

From scratch means all trainable MMAudio transformer weights are randomly
initialized. The VAE, DFN5B CLIP, and Synchformer remain pretrained feature
extractors, as in the official recipe. The fixed CLIP embedding for an empty
string is loaded either from upstream `empty_string.pth` or a converted official
checkpoint, but is not trainable and does not initialize any DiT layer. To avoid
downloading a DiT checkpoint for a pure scratch run:

```bash
curl -L --continue-at - \
  https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth \
  -o official_weights/mmaudio/raw/ext_weights/empty_string.pth
```

Four configs mirror the official v1 architectures:

```text
examples/train/configs/fine_tuning/mmaudio/small_16k_from_scratch.yaml
examples/train/configs/fine_tuning/mmaudio/small_44k_from_scratch.yaml
examples/train/configs/fine_tuning/mmaudio/medium_44k_from_scratch.yaml
examples/train/configs/fine_tuning/mmaudio/large_44k_from_scratch.yaml
```

They use AdamW (`1e-4`, betas `0.9/0.95`, epsilon `1e-6`, weight decay
`1e-6`), 1,000 warmup steps, MultiStepLR drops at 240k and 270k, gradient clip
1.0, and 300k steps. The FSDP-aware `PostHocEMACallback` mirrors upstream
MMAudio's `nitrous_ema` settings: sigma profiles `[0.05, 0.1]`, an update every
step, snapshots every 5,000 steps, and final synthesis at sigma `0.05`. It is an
additional callback; FastVideo's existing conventional EMA remains unchanged
for other models.

Edit dataset paths, variant, batch size, learning rate, validation intervals,
output directory, resume path, and tracker at the top of the launcher. No
shell environment variables or command-line overrides are required. The
launcher also downloads the official standalone `empty_string.pth` if it is
missing. Then run the current 44.1 kHz VGGSound cache on four GPUs:

```bash
bash examples/training/finetune/mmaudio/run_train_vggsound_from_scratch.sh
```

### Optional full-replica DDP for `small_44k`

MMAudio's upstream runner uses native PyTorch DDP with one complete model per
GPU and `broadcast_buffers=False`. FastVideo exposes that behavior as the
opt-in `training.distributed.strategy: ddp`; the default remains `fsdp`, so
existing model configs and the launcher above are unchanged. The initial DDP
plugin is intentionally limited to MMAudio from-scratch training.

The independent four-GPU config and launcher are:

```text
examples/train/configs/fine_tuning/mmaudio/small_44k_ddp_from_scratch.yaml
examples/training/finetune/mmaudio/run_train_vggsound_ddp_from_scratch.sh
```

Run it only after the current GPU job has finished:

```bash
bash examples/training/finetune/mmaudio/run_train_vggsound_ddp_from_scratch.sh
```

The script uses port `29502` and writes to
`outputs/mmaudio_small_44k_ddp_from_scratch`, so it does not reuse the FSDP
job's rendezvous or output directory. DDP can be faster for this small model
when every GPU can hold the full model, optimizer, activations, and rank 0's
two EMA copies. The DDP config also enables upstream's fused AdamW. It uses
more per-GPU model memory than FSDP.

The DDP config also enables `training.model.compile_train_fn: true`.
Following upstream MMAudio, compilation happens after DDP wrapping. For
PyTorch 2.12 compatibility, FastVideo keeps explicit-Generator posterior,
flow-time, prior, and CFG-mask sampling in eager code, applies masks with
broadcasted `torch.where`, and compiles the inner transformer's tensor-only
forward with `fullgraph: true`. The outer DDP reducer/control flow and the
lightweight MSE loss remain eager. This preserves the official RNG order while
making any graph break inside the expensive transformer graph a hard error
instead of a silent fallback.

Set `COMPILE_TRAIN_FN=false` near the top of the DDP launcher to disable it.
The typed default is `false`, and the option is currently rejected for FSDP,
so existing FastVideo pipelines are unaffected. Validation loss reuses the
compiled transformer forward; periodic generation keeps its separate eager
`guided_flow` path. Optional `torch.compile` arguments can be placed under
`training.model.torch_compile_kwargs`.

For parity with upstream MMAudio, rank 0 runs the exact
`nitrous_ema.PostHocEMA` implementation with sigma profiles `[0.05, 0.1]`,
updates every optimizer step, and snapshots every 5,000 steps. FastVideo's
distributed checkpoint interval is 10,000 steps so every resumable training
checkpoint has a matching EMA snapshot. Periodic validation/inference still
uses online weights, and the final sigma-0.05 synthesis is saved under
`posthoc_ema/official_ddp/`.

Set `VARIANT` to `medium_44k` or `large_44k` without changing the feature
cache. The launcher computes official latent normalization statistics once,
saves `latent_statistics_44k.pt` inside the feature directory, and reuses it
across those three variants. `small_16k` needs a separately preprocessed 16 kHz
cache because its latent shape is `[250, 20]`. `large_44k_v2` is rejected before
weights or data are loaded because the published training implementation does
not support v2.

### Validation and training-time inference

`MMAudioValidationCallback` evaluates the cached validation split every 5,000
steps by default. It follows the official validation loss construction:
posterior latent sampling, logit-normal flow time, prior noise, and independent
video/text CFG masks. The validation RNG is reset on every pass so losses are
comparable between checkpoints. Set `VALIDATION_MAX_BATCHES=0` in the launcher
to use the full split, or a positive value for a faster subset.

Every 20,000 steps the same callback runs the native FastVideo MMAudio pipeline
with the live training transformer and cached CLIP, Synchformer, and text
features. It selects a fixed global set of 16 validation samples (four per rank
on a four-GPU job), not the complete validation split. The validation seed,
sampler, and per-sample noise seeds stay fixed across training steps.
`INFERENCE_MODEL_PATH` supplies only the frozen scheduler, audio VAE, and
vocoder needed to decode a waveform; its transformer weights and feature
encoders are not loaded. Samples are written to:

```text
outputs/mmaudio_<variant>_from_scratch/validation_audio/step_<step>/
```

Both a standalone WAV and an MP4 containing the original validation video plus
generated audio are saved. MP4 composition mirrors official MMAudio demo's
`reencode_with_audio`: decode source frames through the requested endpoint,
re-encode H.264 at the source's guessed frame rate and 10 Mbps in `yuv420p`,
then feed the mono generated waveform to the AAC encoder at the model sample
rate. It uses PyAV and does not import Torio. As in the official 44.1 kHz path,
the encoded container can be slightly longer than 8 seconds because 353,280
audio samples and AAC frame padding do not end at exactly 8.0 seconds. If
composition or source-video loading fails, the WAV is retained and training
continues.

Post-hoc EMA rank-local profiles and the final synthesized sigma-0.05 shards
are written under `posthoc_ema/rank_<rank>/`. Periodic validation uses online
weights by default, matching the official training loop; official MMAudio only
synthesizes sigma-0.05 EMA weights after training. Set `use_ema: true` in the
YAML only when intentionally evaluating an available post-hoc snapshot. The
current callback reports flow-matching validation loss and generates media
samples; dataset-level FAD/CLAP/synchronization benchmark metrics remain a
separate evaluation job.

Set `TRACKER="wandb"` near the top of the launcher after `wandb login` to log
`total_loss`, `flow_matching_loss`, gradient norm, step time, validation loss,
and the 16 composed validation MP4 files. Set
`INFERENCE_LOG_TO_TRACKER=false` to keep media local while retaining scalar
logging.

Each YAML's `training.model_path` names a matching converted component tree.
Training never reads its DiT weights; the path is retained so FastVideo can
copy the component configs and replace the transformer when exporting a DCP
checkpoint:

```bash
python scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py \
  --variant small_44k \
  --transformer-config-only \
  --output converted_weights/mmaudio/small_44k
```

This export skeleton contains no weights and is sufficient for transformer-only
checkpoint export. Then run:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint outputs/mmaudio_small_44k_from_scratch/checkpoint-10000 \
  --output-dir converted_weights/mmaudio/small_44k_trained \
  --overwrite
```

For a directly runnable inference export, the template at
`converted_weights/mmaudio/small_44k` must contain all pipeline components.
A transformer-only template produces a transformer-only export, which can be
combined with the shared 44 kHz encoders/VAE/vocoder component tree.

### Preprocess VGGSound with FastVideo

Feature extraction is a native FastVideo preprocessing workflow. The upstream
MMAudio repository is useful as a numerical reference, but is not imported by
the preprocessing or training command.

The reference training loader imports the now-removed
`torio.io.StreamingMediaDecoder`. Current FastVideo uses `torchaudio.load` for
audio and PyAV timestamp sampling for the two video frame rates, while retaining
the reference normalization, resampling, transforms, mel, and encoder contracts.

First create a preprocessing-only component tree. It contains the 44.1 kHz
VAE encoder, DFN5B CLIP text/vision encoders, Synchformer, and tokenizer; it
does not duplicate the trainable DiT or inference vocoder:

```bash
python scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py \
  --preprocessor-only \
  --audio-vae-checkpoint official_weights/mmaudio/raw/ext_weights/v1-44.pth \
  --synchformer-checkpoint official_weights/mmaudio/raw/ext_weights/synchformer_state_dict.pth \
  --dfn5b-dir official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384 \
  --output converted_weights/mmaudio/preprocess_44k
```

For the `Loie/VGGSound` release, extract the gzip shards once for random-access
decoding. The archives contain a seven-component directory prefix:

```bash
mkdir -p /path/to/VGGSound/videos
for shard in /path/to/VGGSound/vggsound_*.tar.gz; do
  tar -xzf "$shard" -C /path/to/VGGSound/videos --strip-components=7
done
```

Run the shared FastVideo preprocessing entrypoint. Each rank gets disjoint
samples and writes resumable TensorDict shards. Existing sample IDs are skipped
when the same output directory is resumed. A filtered caption manifest must
have the same `id<TAB>label` schema used by MMAudio; `label` is the full text
description encoded by DFN5B, not the original VGGSound class label.

```bash
torchrun --standalone --nproc_per_node=4 \
  -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
  --model-path converted_weights/mmaudio/preprocess_44k \
  --mode preprocess \
  --workload-type v2a \
  --preprocess.dataset-type vggsound \
  --preprocess.dataset-path /path/to/VGGSound \
  --preprocess.dataset-metadata-path /path/to/VGGSound/sets/filtered_caption/vgg-train-filtered-caption.tsv \
  --preprocess.dataset-split train \
  --preprocess.dataset-output-dir /path/to/VGGSound/mmaudio_features/train \
  --preprocess.preprocess-video-batch-size 1 \
  --preprocess.dataloader-num-workers 2 \
  --preprocess.samples-per-file 256
```

The configurable launcher defaults to four GPUs and derives the manifest from
`SPLIT`:

```bash
DATASET_PATH=/path/to/VGGSound SPLIT=train \
  bash examples/training/finetune/mmaudio/preprocess_vggsound.sh
```

Use the same pipeline for every split by changing only `SPLIT`; write each one
to its own cache directory:

```bash
for split in train val test; do
  DATASET_PATH=/path/to/VGGSound SPLIT="$split" GPU_NUM=4 \
    bash examples/training/finetune/mmaudio/preprocess_vggsound.sh
done
```

MMAudio's split contract peak-normalizes train audio to 0.95 and leaves val/test
audio amplitudes unchanged. Corrupt, silent, or short train inputs and invalid
val/test inputs are skipped and recorded in `failures_rank_*.jsonl`.

### Precomputed feature contract

The online training step deliberately does not run the audio VAE, DFN5B, or
Synchformer. Point `training.data.data_path` at either one TensorDict
memory-mapped directory or the parent directory of FastVideo's feature shards:

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
learned null-video tokens. The `text_features` values are normalized per-token
DFN5B CLIP hidden states computed from the dataset captions. The cache loader
has no runtime import dependency on the upstream `mmaudio` package.

The example config intentionally contains an empty data path until a dataset is
prepared:

```text
examples/train/configs/fine_tuning/mmaudio/small_44k.yaml
```

Start a single-GPU run while supplying the cache path on the command line:

```bash
NUM_GPUS=1 bash examples/train/run.sh \
  examples/train/configs/fine_tuning/mmaudio/small_44k.yaml \
  --training.data.data_path /path/to/VGGSound/mmaudio_features/train
```

Leaving the path empty produces an explicit error instead of silently starting
with the wrong data. The pretrained single-GPU FSDP path and
scratch-initialized four-GPU FSDP path have both been validated. Sequence and
tensor parallelism are not implemented, so keep `sp_size: 1` and `tp_size: 1`. The
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
