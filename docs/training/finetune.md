# 🧠 Finetuning

This guide covers finetuning video diffusion models with FastVideo, including full finetuning and LoRA.

## Training Arguments

FastVideo training scripts use several argument groups:

### Training Arguments

| Argument | Description |
|----------|-------------|
| `--max_train_steps` | Total training steps |
| `--train_batch_size` | Batch size per GPU |
| `--gradient_accumulation_steps` | Steps to accumulate before optimizer update |
| `--num_latent_t` | Temporal latent dimension (reduce to save memory) |
| `--num_height` / `--num_width` | Video resolution |
| `--num_frames` | Number of frames per video |
| `--output_dir` | Directory for checkpoints |

### Parallelism Arguments

| Argument | Description |
|----------|-------------|
| `--num_gpus` | Total number of GPUs |
| `--sp_size` | Sequence parallel size (increase to reduce memory per GPU) |
| `--tp_size` | Tensor parallel size |
| `--hsdp_replicate_dim` | HSDP replication dimension |
| `--hsdp_shard_dim` | HSDP sharding dimension |

### Optimizer Arguments

| Argument | Description |
|----------|-------------|
| `--learning_rate` | Base learning rate |
| `--mixed_precision` | Precision mode (`bf16` recommended) |
| `--weight_decay` | Weight decay for regularization |
| `--max_grad_norm` | Gradient clipping threshold |

### Validation Arguments

| Argument | Description |
|----------|-------------|
| `--log_validation` | Enable validation logging |
| `--validation_dataset_file` | JSON file with validation prompts |
| `--validation_steps` | Run validation every N steps |
| `--validation_sampling_steps` | Inference steps for validation |
| `--validation_guidance_scale` | CFG scale for validation |

## Full Finetuning

Full finetuning updates all model weights. This provides the best quality but requires more GPU memory.

```bash
# Example: Wan2.1 T2V 1.3B full finetune (4 GPUs)
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh
```

**Typical settings:**

- Learning rate: `1e-5` to `5e-5`
- Gradient checkpointing: `--enable_gradient_checkpointing_type "full"`
- Memory scaling: Increase `--sp_size` or reduce `--num_latent_t` to fit in memory

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

| Argument | Description |
|----------|-------------|
| `--lora_training True` | Enable LoRA mode |
| `--lora_rank` | Rank of LoRA adapters (16, 32, 64, 128) |

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

- Add `--lora_training True --lora_rank 32`
- Use higher learning rate (10–20× full finetune)
- Can run on fewer GPUs (even single GPU)
- Outputs adapter weights instead of full model

## LoRA Extraction and Merging

FastVideo provides generic runtime adapter extraction and retains a legacy merger for its previously supported adapter layouts.

### Extract LoRA Adapter

Extract a LoRA adapter by comparing a finetuned model to its base:

```bash
python scripts/lora_extraction/extract_lora.py \
  --base Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --ft path/to/your/finetuned_model \
  --out adapter_r32.safetensors \
  --rank 32 \
  --exact-tensor-pattern '^condition_embedder\.' \
  --exact-tensor-pattern '^proj_out\.weight$'
```

| Argument | Description |
|----------|-------------|
| `--base` | Base model (Hugging Face ID or local path) |
| `--ft` | Fine-tuned model (Hugging Face ID or local path) |
| `--out` | Output adapter file (`.safetensors`) |
| `--rank` | Requested LoRA rank (for example, 16, 32, 64, or 128) |
| `--full-rank` | Extract full-rank adapter (optional) |
| `--min-delta` | Omit tensors whose maximum absolute FP32 delta is at or below the threshold (default: `1e-8`) |
| `--load-mode` | `auto` (indexed, then pipeline fallback), `indexed`, or `pipeline` |
| `--device` | SVD device, such as `cpu` or `cuda:0` |
| `--svd-method` | Exact or randomized SVD |
| `--factor-dtype` | Storage dtype for the low-rank factors |
| `--dense-dtype` | Storage dtype for exact `.diff`/`.diff_b`/`.diff_param` payloads (default: `float32`) |
| `--exact-tensor-pattern` | Repeatable regex for matrices the target runtime cannot load as LoRA factors |

For large checkpoints, indexed loading streams one transformer tensor pair at a time and downloads only `transformer/*`. The extractor also preserves changed norms, biases, and standalone parameters as exact deltas, and fine-tuned-only parameters as `.set_weight` or `.set_param`. Matrix selection is runtime-agnostic, so full-finetune extraction must keep runtime-unsupported matrices exact, as the Wan example does. See the [LoRA utilities](../utilities/lora.md) for the GPU/randomized-SVD command, resume options, and accuracy controls.

Mixed low-rank/dense adapters from the generic extractor must be supplied when constructing FastVideo through
`ComponentConfig(lora_path=...)`; their dense payload cannot be swapped later with `set_lora_adapter`. The legacy
offline merger below is not part of this extraction workflow.

### Legacy Merge LoRA Adapter

The command below documents the pre-existing merger for adapters it already supports. Do not pass a mixed adapter from
the generic extractor to it: the legacy merger does not apply exact dense or replacement payloads.

```bash
python scripts/lora_extraction/merge_lora.py \
  --base Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --adapter legacy_factor_only_adapter.safetensors \
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
- `validation.json` — validation prompts
