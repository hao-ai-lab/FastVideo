# 🧱 Data Preprocessing

To save GPU memory during training, FastVideo precomputes text embeddings and VAE latents. This eliminates the need to load the text encoder and VAE during training.

## Quick Start

Download the sample dataset and run preprocessing:

```bash
# Download the crush-smol dataset
python scripts/huggingface/download_hf.py \
    --repo_id "wlsaidhi/crush-smol-merged" \
    --local_dir "data/crush-smol" \
    --repo_type "dataset"

# Run preprocessing
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v_new.sh
```

## Preprocessing Pipeline

The new preprocessing pipeline supports multiple dataset formats and video loaders:

```bash
GPU_NUM=2
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATASET_PATH="data/crush-smol/"
OUTPUT_DIR="data/crush-smol_processed_t2v/"

torchrun --nproc_per_node=$GPU_NUM \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 2 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 5
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--workload_type` | Task type: `t2v` (text-to-video) or `i2v` (image-to-video) |
| `--preprocess.dataset_type` | Input format: `hf` (HuggingFace) or `merged` (local folder) |
| `--preprocess.dataset_path` | Path to dataset (HF repo ID or local folder) |
| `--preprocess.dataset_output_dir` | Output directory for Parquet files |
| `--preprocess.video_loader_type` | Video decoder: `torchcodec` or `torchvision` |
| `--preprocess.max_height` / `max_width` | Target resolution for videos |
| `--preprocess.num_frames` | Number of frames to extract per video |
| `--preprocess.train_fps` | Target FPS for frame extraction |

## Dataset Formats

### Merged Dataset (Local Folder)

Structure your dataset as follows:

```
your_dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── videos2caption.json
```

The `videos2caption.json` maps video filenames to captions:

```json
[
  {"path": "video_001.mp4", "cap": "A cat playing with yarn..."},
  {"path": "video_002.mp4", "cap": "Ocean waves at sunset..."}
]
```

### HuggingFace Dataset

Use `--preprocess.dataset_type hf` and point `--preprocess.dataset_path` to a HuggingFace dataset with `video` and `caption` columns.

## Creating Your Own Dataset

If you have raw videos and captions in separate files, generate the `videos2caption.json`:

```bash
python scripts/dataset_preparation/prepare_json_file.py \
    --data_folder path/to/your_raw_data/ \
    --output path/to/output_folder
```

Your raw data folder should contain:

```
your_raw_data/
├── videos/
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── videos.txt    # list of video filenames
└── prompt.txt    # corresponding captions (one per line)
```

## Output Format

Preprocessing outputs Parquet files in the `combined_parquet_dataset/` subdirectory containing:

- `vae_latent_bytes` — VAE-encoded video latent
- `text_embedding_bytes` — text encoder output
- `clip_feature_bytes` — CLIP image features (I2V only)
- `first_frame_latent_bytes` — first frame latent (I2V only)
- Metadata: shapes, dtypes, and sample identifiers

## Examples

See ready-to-run preprocessing scripts in the training examples:

- **T2V**: `examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v_new.sh`
- **I2V**: `examples/training/finetune/wan_i2v_14B_480p/crush_smol/preprocess_wan_data_i2v_new.sh`

**→ [Browse all training examples](examples/examples_training_index.md)**
