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

## Preprocessing

Before training, you need to preprocess your raw videos and captions into Parquet files containing precomputed latents and embeddings.

FastVideo supports two input formats:

- **HuggingFace datasets** — load directly from HF Hub or local HF datasets
- **Merged datasets** — local folder with videos and a `videos2caption.json` metadata file

**→ See [Data Preprocessing](data_preprocess.md) for full details and examples.**

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
