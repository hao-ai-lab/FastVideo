(v0-data-preprocess)=

# 🧱 Data Preprocess for Distillation

For distillation, we use the same data preprocessing pipeline as training. Please refer to the [Training Data Preprocess](../training/data_preprocess.md) for general preprocessing steps.

## Distillation-Specific Datasets

### FastVideo 480P Synthetic Wan Dataset

For Wan2.1 T2V distillation, we use the **FastVideo 480P Synthetic Wan dataset** ([FastVideo/Wan-Syn_77x448x832_600k](https://huggingface.co/datasets/FastVideo/Wan-Syn_77x448x832_600k)) which contains 600k synthetic latents.

```bash
# Download the preprocessed dataset
python scripts/huggingface/download_hf.py \
    --repo_id "FastVideo/Wan-Syn_77x448x832_600k" \
    --local_dir "FastVideo/Wan-Syn_77x448x832_600k" \
    --repo_type "dataset"
```

### Crush Smol Dataset

For Wan2.2 TI2V distillation, we use the crush_smol dataset which includes both raw videos and preprocessed latents.

```bash
# Download dataset
python scripts/huggingface/download_hf.py \
    --repo_id=FastVideo/mini_i2v_dataset \
    --local_dir=data/mini_i2v_dataset \
    --repo_type=dataset
```

## Preprocessing for Distillation

The preprocessing steps are identical to training. Run the appropriate preprocessing script based on your model:

```bash
# For Wan2.1 T2V
bash scripts/preprocess/v1_preprocess_wan_data_t2v

# For Wan2.2 TI2V  
bash examples/distill/Wan2.2-TI2V-5B-Diffusers/crush_smol/preprocess_wan_data_ti2v_5b.sh
```
