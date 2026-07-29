# HunyuanVideo 1.5 T2V Training

This guide walks through training HunyuanVideo 1.5 (480p, text-to-video) in
FastVideo's modular training stack: preprocessing one clip, running a
single-GPU overfit to verify the pipeline end to end, and scaling out to
full-parameter multi-GPU fine-tuning.

The preprocessing step below relies on the HY1.5 data-side pipeline — the
dual text-embedding parquet schema, its collate path, and
`preprocess_hunyuan15_overfit.py` — which ships separately. Training itself
only needs the parquet that step produces.

## What you need

| | |
|---|---|
| GPU | 48GB or more for the LoRA overfit; multi-GPU for full-parameter runs |
| Disk | ~40GB for the checkpoint (DiT 17GB + Qwen2.5-VL 16GB + ByT5/VAE) |
| Checkpoint | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` |

Install as documented in the repository README:

```bash
UV_TORCH_BACKEND=cu126 uv pip install -e ".[dev]"   # cu130 on CUDA 13
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`fastvideo-kernel` pulls in a CUTLASS submodule. If the build fails with
`cutlass/cutlass.h: No such file or directory`, run
`git submodule update --init --recursive` — or skip the kernel package
entirely, since this guide uses the PyTorch SDPA attention backend.

## 1. Prepare a clip

The preprocess script reads a caption manifest plus a `videos/`
subdirectory:

```
data/hunyuan15_overfit/
├── videos2caption.json
└── videos/
    └── robot_pouring.mp4
```

The repository already ships a suitable clip (5.8s, 16fps, 1280x704):

```bash
mkdir -p data/hunyuan15_overfit/videos
cp assets/videos/robot_pouring.mp4 data/hunyuan15_overfit/videos/

cat > data/hunyuan15_overfit/videos2caption.json <<'EOF'
[
  {
    "path": "robot_pouring.mp4",
    "cap": ["A robotic arm carefully pours liquid from a bottle into a glass on a kitchen counter, steady camera, soft indoor lighting."]
  }
]
EOF
```

Clips must supply at least 81 frames at 16fps (≈5.1s) and be a single
continuous shot; the script resizes to 480x832.

## 2. Preprocess

```bash
CUDA_VISIBLE_DEVICES=0 python fastvideo/pipelines/preprocess/preprocess_hunyuan15_overfit.py
```

This encodes the video with the HY1.5 VAE and the caption with both text
encoders, then writes a parquet shard plus a validation prompt file.
Expected output:

```
[Qwen 1/1] (29, 3584)          # caption tokens x Qwen hidden size
[ByT5 1/1] (0, 1472)           # zero-length: the caption has no quoted glyph text
[VAE  1/1] (32, 21, 30, 52)    # 32 channels, 21 latent frames, 480/16 x 832/16
Wrote 1 records to data/hunyuan15_overfit_preprocessed/data_00000.parquet
```

A `(0, 1472)` ByT5 shape is normal and expected: ByT5 only receives the
glyph text extracted from quotes in the caption, so most captions produce
zero tokens. The training plugin trims and forwards that zero-length
stream unchanged.

Note that the parquet stores raw `latent_dist.mode()` outputs — the
training side multiplies by the VAE `scaling_factor` (1.03682) in
`normalize_dit_input("hunyuan15", ...)`.

## 3. Overfit on a single GPU

```bash
export WANDB_MODE=offline      # or `wandb login` for cloud tracking
bash examples/train/run.sh examples/train/configs/overfit_hunyuan15_t2v.yaml
```

The config trains LoRA adapters on the attention projections, which keeps
the run at roughly 30GB. Loss is logged through the configured tracker,
not to stdout.

For a quick "does it run at all" check, cap the step count:

```bash
bash examples/train/run.sh examples/train/configs/overfit_hunyuan15_t2v.yaml \
  --training.loop.max_train_steps 50
```

**Run long jobs under `tmux` or `screen`.** A dropped SSH session takes the
training process with it, and a checkpoint interrupted mid-write is not
resumable (the DCP `.metadata` file is written last).

### Reading the loss

Flow-matching loss is noisy: every step samples a random timestep, and the
denoising task is far easier at low noise levels than at high ones. Two
adjacent steps can differ by an order of magnitude with no bearing on
whether the model is learning. Compare segment means rather than
individual steps:

```bash
RUN=$(ls -dt outputs/hunyuan15_overfit/tracker/wandb/offline-run-* | head -1)
strings -n 1 $RUN/run-*.wandb | grep -A 2 -x "total_loss" \
  | grep -E "^[0-9]+\.[0-9]" | uniq > /tmp/losses.txt

python3 -c "
import re
vals = [float(m.group(1)) for line in open('/tmp/losses.txt')
        if (m := re.match(r'^([0-9]+\.[0-9]+)', line))]
n = len(vals)
for i in range(6):
    seg = vals[i * n // 6:(i + 1) * n // 6]
    print(f'segment {i + 1}/6: {sum(seg) / len(seg):.4f}')
"
```

A steadily falling segment mean means the data → forward → loss → backward
→ optimizer chain is numerically sound. A flat series, or NaNs, points at
a real problem — stop and investigate rather than burning GPU hours.

## 4. Scale out

`examples/train/configs/fine_tuning/hunyuan15/t2v.yaml` runs
full-parameter training across 8 GPUs with FSDP. Full-parameter training
does not fit one 80GB card:

| | bf16 |
|---|---|
| Weights | 16.7 GB |
| Gradients | 16.7 GB |
| AdamW `exp_avg` + `exp_avg_sq` | 33.4 GB |
| **Optimizer + weights subtotal** | **66.8 GB** |
| Activations, gradient-checkpoint boundaries, VAE, allocator overhead | ~13 GB |
| **Total** | **~80 GB** |

That subtotal is independent of sequence length, so shortening the clip
(`--training.data.num_latent_t`) barely helps — measured peak memory moved
by 0.07GB going from 21 latent frames down to 9. Options that do move the
needle:

- **LoRA** (what the overfit config uses) — gradients and optimizer state
  shrink to the adapter parameters, ~30GB total.
- **FSDP sharding across GPUs** — the 66.8GB subtotal divides by the number
  of shards.
- **A memory-light optimizer** — SGD with momentum keeps a single state
  tensor instead of two. `torch.optim.Adafactor` is smaller still, but is
  not DTensor-safe: its in-place `.square_()` fails under FSDP2 with
  `aten.pow_.Scalar: in-place operations that require placement changes
  are not supported`.

Set `training.dit_precision: bf16` for any single-device run. The default
is fp32, which doubles weights, gradients, and optimizer state.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `wandb.errors.UsageError: No API key configured` | Setting `tracker.project_name` enables W&B automatically. Use `export WANDB_MODE=offline` or `wandb login`. |
| `RuntimeError: Given normalized_shape=[1472] ... got [1, 0, 512]` during validation | `TextEncodingStage` sizes the zero-length ByT5 placeholder from the static T5 config default rather than HY1.5's 1472, so captions without glyph text fail. The overfit config leaves validation disabled for this reason. |
| `OutOfMemoryError` at `optimizer.step()` | Optimizer state does not fit. Use the LoRA config, shard across GPUs, or switch optimizers — see the table above. |
| `No LoRA-compatible layers were found for the requested target modules` | HY1.5 names its attention projections `img_attn_qkv` / `txt_attn_qkv` / `img_attn_proj` / `txt_attn_proj`, which the default target list does not cover. The shipped config sets them explicitly. |
| `ValueError: No frames could be decoded from ...` | The clip must live under `data/hunyuan15_overfit/videos/`, not the manifest directory itself. |
| Training dies with SIGTERM and no traceback | The SSH session ended, or another user on a shared machine reclaimed memory. Use `tmux`, and coordinate before taking most of a shared box. |
| Hangs at "Downloading model snapshot from HF Hub" | Rate-limited Hub request. `export HF_HUB_OFFLINE=1` once the weights are cached, or set `HF_TOKEN`. |

## Notes on other platforms

The stack runs on aarch64 + Blackwell (DGX Spark, GB10, CUDA 13.0):
preprocessing and training both complete there. Unified memory makes
checkpoint loading much faster than on a discrete GPU — 17GB of weights
load in about 1.5s versus 30s over PCIe — while each training step runs
roughly 1.5x slower than an A100 80GB. `nvidia-smi` reports
`Memory-Usage: Not Supported` on that platform; use `free -m` instead,
since GPU allocations come out of the same pool as system memory.
