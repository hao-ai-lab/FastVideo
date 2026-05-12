# Cosmos 2.5 T2W Fine-tuning

End-to-end fine-tuning of Cosmos-Predict2.5-2B for text-to-world generation.

## Quick Start

Run all commands from the `FastVideo/` root directory.

### 1. Prepare your dataset

Create a directory with your videos and captions:

```
data/cosmos2_5_raw/
├── merge.txt          # single line: "videos,annotation.json"
├── videos/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
└── annotation.json
```

**merge.txt** (one line):
```
videos,annotation.json
```

**annotation.json** format:
```json
[
  {
    "path": "clip_001.mp4",
    "cap": ["A robot arm picks up a red cube from a table."],
    "resolution": {"width": 1280, "height": 704},
    "fps": 24.0,
    "duration": 3.2,
    "num_frames": 77
  }
]
```

Videos should be at least 704x1280 and ~3+ seconds at 24fps (77 frames).

### 2. Preprocess (encode latents + text embeddings)

```bash
bash examples/training/finetune/cosmos2_5/preprocess_cosmos2_5_t2w.sh
```

This encodes videos with Cosmos25WanVAE (16-ch latents, normalized internally)
and captions with Reason1/Qwen2.5-VL (100352-dim full_concat embeddings).
Output goes to `data/cosmos2_5_processed_t2w/combined_parquet_dataset/`.

Requires 1 GPU with >= 24GB VRAM (Reason1 ~15GB + VAE ~2GB).

### 3. Fine-tune

**LoRA (1 GPU, ~20GB VRAM):**
```bash
bash examples/training/finetune/cosmos2_5/finetune_t2w_lora.sh
```

**Full fine-tune (4 GPUs, ~40GB VRAM each):**
```bash
bash examples/training/finetune/cosmos2_5/finetune_t2w.sh
```

### 4. Edit paths

Before running, update in the shell scripts:
- `DATA_DIR` — point to your preprocessed parquet directory
- `MODEL_PATH` — HuggingFace model ID or local path to Cosmos 2.5 weights
- `DATA_MERGE_PATH` — (preprocessing only) path to your `merge.txt`

## Key differences from WAN T2V training

| | WAN T2V | Cosmos 2.5 T2W |
|---|---------|---------------|
| Text encoder | T5-XXL (4096-dim) | Reason1/Qwen2.5-VL (100352-dim) |
| VAE | WanVAE | Cosmos25WanVAE (normalized latents) |
| Resolution | 480x832 | 704x1280 |
| FPS | 16 | 24 |
| Scheduler | Flow (shift=3.0) | Flow (shift=5.0) |
| Latent normalization | During training | During VAE encoding (skip in training) |

## Troubleshooting

**OOM during training**: Reduce resolution (`--num_height 480 --num_width 832`)
or disable validation (`remove --log_validation`).

**OOM during preprocessing**: Use `--preprocess_video_batch_size 1`.

**Validation crashes**: The validation pipeline loads the full model (text encoder +
VAE + DiT) a second time. Disable with `--log_validation False` for initial runs.
