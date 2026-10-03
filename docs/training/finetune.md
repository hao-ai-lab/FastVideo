# 🧠 Finetuning

This guide covers finetuning video diffusion models with FastVideo, including full finetuning and LoRA.

## Training Arguments

Each training launcher passes a `TrainingRunConfig` YAML file to its entry point with `--config` (the schema is in
`fastvideo/api/training_schema.py`). A dotted override after `--config` sets one field, for example
`--training.optimizer.learning_rate 1e-5` or `--engine.num_gpus "$NUM_GPUS"`:

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/training/wan_training_pipeline.py \
    --config finetune_t2v.yaml \
    --engine.num_gpus 4
```

The settings are grouped as follows:

### Training Arguments

| Config path                                            | Description                                       |
| ------------------------------------------------------ | ------------------------------------------------- |
| `training.loop.max_train_steps`                        | Total training steps                              |
| `training.data.train_batch_size`                       | Batch size per GPU                                |
| `training.loop.gradient_accumulation_steps`            | Steps to accumulate before optimizer update       |
| `training.data.num_latent_t`                           | Temporal latent dimension (reduce to save memory) |
| `training.data.num_height` / `training.data.num_width` | Video resolution                                  |
| `training.data.num_frames`                             | Number of frames per video                        |
| `training.checkpoint.output_dir`                       | Directory for checkpoints                         |

### Parallelism Arguments

| Config path                             | Description                                                |
| --------------------------------------- | ---------------------------------------------------------- |
| `engine.num_gpus`                       | Total number of GPUs                                       |
| `engine.parallelism.sp_size`            | Sequence parallel size (increase to reduce memory per GPU) |
| `engine.parallelism.tp_size`            | Tensor parallel size                                       |
| `engine.parallelism.hsdp_replicate_dim` | HSDP replication dimension                                 |
| `engine.parallelism.hsdp_shard_dim`     | HSDP sharding dimension                                    |

### Optimizer Arguments

| Config path                        | Description                     |
| ---------------------------------- | ------------------------------- |
| `training.optimizer.learning_rate` | Base learning rate              |
| `training.optimizer.weight_decay`  | Weight decay for regularization |
| `training.optimizer.max_grad_norm` | Gradient clipping threshold     |

### Validation Arguments

| Config path                          | Description                                                 |
| ------------------------------------ | ----------------------------------------------------------- |
| `training.validation.enabled`        | Enable validation logging                                   |
| `training.validation.dataset_file`   | JSON file with validation prompts                           |
| `training.validation.every_steps`    | Run validation every N steps                                |
| `training.validation.sampling_steps` | Inference steps for validation (a list, for example `[50]`) |
| `training.validation.guidance_scale` | CFG scale for validation                                    |

## Full Finetuning

Full finetuning updates all model weights. This provides the best quality but requires more GPU memory.

```bash
# Example: Wan2.1 T2V 1.3B full finetune (4 GPUs)
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh
```

**Typical settings:**

- Learning rate: `1e-5` to `5e-5`
- Gradient checkpointing: `training.model.enable_gradient_checkpointing_type: full`
- Memory scaling: Increase `engine.parallelism.sp_size` or reduce `training.data.num_latent_t` to fit in memory

## Attention Quantization-Aware Training

Attn-QAT fine-tunes a model while simulating low-bit attention in the forward
and backward passes. The modular trainer can select the backend per model role,
so a later DMD2 stage can keep fake quantization on the student while the
teacher and critic use Flash Attention.

The ready-to-run Wan2.1 MixKit workflow includes supervised fine-tuning,
checkpoint export, and three-step DMD2 distillation:

**→ [Follow the Attn-QAT training guide](attn_qat.md)**

## LoRA Finetuning

LoRA (Low-Rank Adaptation) trains lightweight adapters while keeping the base model frozen. This significantly reduces memory usage and training time.

### LoRA-Specific Arguments

| Config path                   | Description                             |
| ----------------------------- | --------------------------------------- |
| `training.lora.enabled: true` | Enable LoRA mode                        |
| `training.lora.rank`          | Rank of LoRA adapters (16, 32, 64, 128) |

### Learning Rate for LoRA

**Important:** LoRA typically requires a **10–20× higher learning rate** than full finetuning because only the low-rank adapters are being trained while the base model is frozen.

| Training Mode | Recommended Learning Rate |
|---------------|---------------------------|
| Full finetune | `1e-5` to `5e-5` |
| LoRA | `1e-4` to `2e-4` |

### Example LoRA Training

```bash
# Example: Wan2.1 T2V 1.3B LoRA finetune (1 GPU)
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v_lora.sh
```

Key differences from full finetune:

- Set `training.lora.enabled: true` and `training.lora.rank: 32`
- Use higher learning rate (10–20× full finetune)
- Can run on fewer GPUs (even single GPU)
- Outputs adapter weights instead of full model

## LoRA Extraction and Merging

FastVideo provides tools to extract LoRA adapters from finetuned models and merge them back.

### Extract LoRA Adapter

Extract a LoRA adapter by comparing a finetuned model to its base:

```bash
python scripts/lora_extraction/extract_lora.py \
  --base Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --ft path/to/your/finetuned_model \
  --out adapter_r32.safetensors \
  --rank 32
```

| Argument | Description |
|----------|-------------|
| `--base` | Base model (HuggingFace ID or local path) |
| `--ft` | Finetuned model path |
| `--out` | Output adapter file (.safetensors) |
| `--rank` | LoRA rank (16, 32, 64, 128) |
| `--full-rank` | Extract full-rank adapter (optional) |

### Merge LoRA Adapter

Merge an adapter back into a base model:

```bash
python scripts/lora_extraction/merge_lora.py \
  --base Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --adapter adapter_r32.safetensors \
  --ft path/to/your/finetuned_model \
  --output merged_model
```

| Argument | Description |
|----------|-------------|
| `--base` | Base model path |
| `--adapter` | LoRA adapter file |
| `--ft` | Finetuned model (for config reference) |
| `--output` | Output directory for merged model |

### Validate Merged Model

Compare the merged model against the original finetuned model:

```bash
python scripts/lora_extraction/lora_inference_comparison.py \
  --base merged_model \
  --ft path/to/your/finetuned_model \
  --adapter NONE \
  --output-dir results \
  --prompt "A cat sitting on a windowsill" \
  --compute-ssim \
  --compute-lpips
```

## Training Examples

Ready-to-run training scripts are available for multiple models:

**→ [Browse all training examples](examples/examples_training_index.md)**

| Model | Type | Example |
|-------|------|---------|
| Wan2.1 T2V 1.3B | T2V | `examples/training/finetune/wan_t2v_1.3B/crush_smol/` |
| Wan2.1 I2V 14B | I2V | `examples/training/finetune/wan_i2v_14B_480p/crush_smol/` |
| Wan2.1-Fun 1.3B InP | I2V | `examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/` |
| Wan2.1 VSA | T2V/I2V | `examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/` |
| Wan2.1 T2V 1.3B Attn-QAT | QAT SFT + DMD2 | `examples/train/scenario/qad_wan2_1_mixkit/` |

Each example includes:

- a README pointing at the matching download script under `examples/datasets/`
- `preprocess_*.sh` — run preprocessing
- `finetune_*.sh` — full finetune launcher
- `finetune_*_lora.sh` — LoRA finetune launcher
- a YAML file next to each launcher — the `TrainingRunConfig` that the launcher passes with `--config`
- `validation.json` — validation prompts
