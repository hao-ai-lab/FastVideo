# MixKit training data (QAD 5090 recipe)

The QAD 5090 models are distilled from Wan2.1-T2V-1.3B on a MixKit subset at
**480×832, 77 frames, 16 fps**. FastVideo training consumes **Parquet** shards of
precomputed VAE latents + text embeddings (no text encoder / VAE needed at train
time).

## Option A — download the preprocessed data (recommended)

The encoded dataset is published on the Hugging Face Hub, ready to train:

```bash
# from the repo root
bash examples/training/finetune/wan_t2v_1.3B/mixkit/download_mixkit_data.sh
```

This pulls [`weizhou03/HD-Mixkit-Finetune-Wan`](https://huggingface.co/datasets/weizhou03/HD-Mixkit-Finetune-Wan)
into `data/HD-Mixkit-Finetune-Wan/`:

```
data/HD-Mixkit-Finetune-Wan/
├── combined_parquet_dataset/      # training shards  -> point --data_path here
│   └── worker_0/data_chunk_*.parquet
└── validation_parquet_dataset/    # validation shards
    └── worker_0/data_chunk_0.parquet
```

Each Parquet row holds the VAE latent bytes + text-embedding bytes (plus
shape/dtype metadata), matching FastVideo's standard preprocessing output.

## Option B — build the Parquet from raw videos

If you want to reproduce the encoding from your own MixKit videos, arrange them as
a `merged` dataset (videos + a captions JSON), then run FastVideo's standard
preprocessing to VAE-encode and text-embed them into Parquet:

```bash
GPU_NUM=2
torchrun --nproc_per_node=$GPU_NUM \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path "data/mixkit_raw/" \
    --preprocess.dataset_output_dir "data/HD-Mixkit-Finetune-Wan/" \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8
```

The raw videos are full-HD MixKit clips (≈1080p/30fps); preprocessing resizes to
480×832, resamples to 16 fps, and extracts 77 frames per clip. See
[`docs/training/data_preprocess.md`](../../../../../docs/training/data_preprocess.md)
for the full parameter reference.

## Next

With the data in place, run the QAD finetune / DMD-distillation scripts in this
directory (added alongside the recipe). The 4-bit attention path is selected via
`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN` and the NVFP4 linear layers via
`transformer_quant="nvfp4_qat"` — see the QAD recipe docs.
