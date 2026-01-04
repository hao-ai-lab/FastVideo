# Training Overview

FastVideo supports finetuning video diffusion models on custom datasets. This page explains what data you need and how to get started.

## Data Requirements

To save GPU memory during training, FastVideo precomputes embeddings and latents ahead of time. This eliminates the need to load the text encoder and VAE during training, significantly reducing memory usage.

### Text-to-Video (T2V) Finetuning

For T2V models, you need:

| Component | Description |
|-----------|-------------|
| **Text embeddings** | Precomputed embeddings from the model's text encoder (e.g., T5 or LLaMA). Stored as numpy arrays in Parquet files. |
| **Video latents** | VAE-encoded representations of your training videos. Each video is encoded into a compressed latent tensor. |

### Image-to-Video (I2V) Finetuning

For I2V models, you need everything from T2V plus additional image conditioning. Note that not all I2V architectures require encoded images—this depends on how the model conditions on the input frame. Wan2.1 and Wan2.2 A14B I2V models do require these additional components:

| Component | Description |
|-----------|-------------|
| **Text embeddings** | Same as T2V—precomputed from the text encoder. |
| **Video latents** | Same as T2V—VAE-encoded video representations. |
| **First frame latent** | VAE-encoded representation of the first frame, used as the conditioning image. |
| **CLIP features** | Image embeddings from a CLIP vision encoder for the conditioning frame. |

## Input Dataset Formats

FastVideo supports two dataset formats for preprocessing:

### HuggingFace Datasets (`--preprocess.dataset_type hf`)

Load datasets directly from the HuggingFace Hub or local HF datasets. The dataset should have:

- A `video` column containing video file paths or binary data
- A `caption` column containing text prompts

```bash
# Example: preprocessing a HuggingFace dataset
torchrun --nproc_per_node=2 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.dataset_type hf \
    --preprocess.dataset_path "your-org/your-video-dataset" \
    --preprocess.dataset_output_dir "data/output_processed/"
```

### Merged Dataset (`--preprocess.dataset_type merged`)

A local folder-based format with videos and a JSON metadata file:

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

## Preprocessing Pipeline

The new preprocessing pipeline (`v1_preprocessing_new`) encodes raw videos and captions into training-ready Parquet files. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `--workload_type` | Task type: `t2v` or `i2v` |
| `--preprocess.dataset_type` | Input format: `hf` or `merged` |
| `--preprocess.dataset_path` | Path to dataset (HF repo ID or local folder) |
| `--preprocess.dataset_output_dir` | Output directory for Parquet files |
| `--preprocess.video_loader_type` | Video decoder: `torchcodec` or `torchvision` |

### T2V Preprocessing Example

```bash
torchrun --nproc_per_node=2 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path "data/your_dataset/" \
    --preprocess.dataset_output_dir "data/your_dataset_processed/" \
    --preprocess.video_loader_type torchvision \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16
```

### I2V Preprocessing Example

```bash
torchrun --nproc_per_node=2 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" \
    --mode preprocess \
    --workload_type i2v \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path "data/your_dataset/" \
    --preprocess.dataset_output_dir "data/your_dataset_processed/" \
    --preprocess.video_loader_type torchvision \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16
```

For more details, see [Data Preprocessing](data_preprocess.md).

## Output Format

Preprocessing outputs a Parquet dataset containing:

- `vae_latent_bytes` — encoded video latent
- `text_embedding_bytes` — text encoder output
- `clip_feature_bytes` — CLIP image features (I2V only)
- `first_frame_latent_bytes` — first frame latent (I2V only)
- Metadata: shapes, dtypes, and sample identifiers

## Training Examples

Ready-to-run examples with preprocessing scripts, training launchers, and validation configs are available for multiple models and datasets:

**→ [Browse all training examples](examples/examples_training_index.md)**

Each example includes:

- `download_dataset.sh` — download sample data
- `preprocess_*.sh` — run preprocessing
- `finetune_*.sh` — launch training (full finetune or LoRA)
- `validation.json` — validation prompts for checkpoints

## Training Methods

FastVideo supports several training approaches:

| Method | Use Case |
|--------|----------|
| **Full finetune** | Adapt entire model to a new domain or style |
| **LoRA finetune** | Lightweight adaptation with frozen base weights |
| **VSA finetune** | Finetune with Variable Sparse Attention for efficiency |

## Next Steps

1. **Get started**: Pick an example from the [training examples index](examples/examples_training_index.md)
2. **Prepare data**: Follow [data preprocessing](data_preprocess.md) for your own dataset
3. **Run inference**: After training, see [inference examples](../inference/examples/examples_inference_index.md)
