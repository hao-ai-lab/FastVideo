# 🎯 Distillation

We introduce a new finetuning strategy - **Sparse-distill**, which jointly integrates **[DMD](https://arxiv.org/abs/2405.14867)** and **[VSA](https://arxiv.org/abs/2505.13389)** in a single training process. This approach combines the benefits of both distillation to shorten diffusion steps and sparse attention to reduce attention computations, enabling much faster video generation.

## 📊 Model Overview

We provide two distilled models:

- **[FastWan2.1-T2V-1.3B-Diffusers](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)**: 3-step inference, up to **16 FPS** on H100 GPU
- **[FastWan2.1-T2V-14B-480P-Diffusers](https://huggingface.co/FastVideo/FastWan2.1-T2V-14B-480P-Diffusers)**: 3-step inference, up to **60x speed up** at 480P, **90x speed up** at 720P for denoising loop
- **[FastWan2.2-TI2V-5B-FullAttn-Diffusers](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers)**: 3-step inference, up to **50x speed up** at 720P for denoising loop

Both models are trained on **61×448×832** resolution but support generating videos with **any resolution** (1.3B  model mainly support 480P, 14B model support 480P and 720P, quality may degrade for different resolutions).

## ⚙️ Inference
First install [VSA](https://hao-ai-lab.github.io/FastVideo/video_sparse_attention/installation.html). Set `MODEL_BASE` to your own model path and run:

```bash
bash scripts/inference/v1_inference_wan_dmd.sh
```

## 🗂️ Dataset

We use the **FastVideo 480P Synthetic Wan dataset** ([FastVideo/Wan-Syn_77x448x832_600k](https://huggingface.co/datasets/FastVideo/Wan-Syn_77x448x832_600k)) for distillation, which contains 600k synthetic latents.

### Download Dataset

```bash
# Download the preprocessed dataset
python scripts/huggingface/download_hf.py \
    --repo_id "FastVideo/Wan-Syn_77x448x832_600k" \
    --local_dir "FastVideo/Wan-Syn_77x448x832_600k" \
    --repo_type "dataset"
```

## 🚀 Training Scripts

### Wan2.1 1.3B Model Sparse-Distill

For the 1.3B model, we use **4 nodes with 32 H200 GPUs** (8 GPUs per node):

```bash
# Multi-node training (8 nodes, 64 GPUs total)
sbatch examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/distill_dmd_VSA_t2v_1.3B.slurm
```

**Key Configuration:**
- Global batch size: 64
- Gradient accumulation steps: 2
- Learning rate: 1e-5
- VSA attention sparsity: 0.8
- Training steps: 4000 (~12 hours)

### Wan2.1 14B Model Sparse-Distill

For the 14B model, we use **8 nodes with 64 H200 GPUs** (8 GPUs per node):

```bash
# Multi-node training (8 nodes, 64 GPUs total)
sbatch examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/distill_dmd_VSA_t2v_14B.slurm
```

**Key Configuration:**
- Global batch size: 64
- Sequence parallel size: 4
- Gradient accumulation steps: 4
- Learning rate: 1e-5
- VSA attention sparsity: 0.9
- Training steps: 3000 (~52 hours)
- HSDP shard dim: 8

### Wan2.2 5B Model Sparse-Distill

For the 5B model, we use **8 nodes with 64 H200 GPUs** (8 GPUs per node):

```bash
# Multi-node training (8 nodes, 64 GPUs total)
sbatch examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/distill_dmd_t2v_5B.sh 
```

**Key Configuration:**
- Global batch size: 64
- Sequence parallel size: 1
- Gradient accumulation steps: 1
- Learning rate: 2e-5
- Training steps: 3000 (~12 hours)
- HSDP shard dim: 1
