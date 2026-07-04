# Wan2.1-T2V Distill Example
These are end-to-end example scripts for distilling Wan2.1 T2V models using DMD-only and DMD+VSA methods.

### 0. Make sure you have installed VSA

```bash
uv pip install vsa
```

### 1. Download dataset:
```bash
bash examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/download_dataset.sh
```

### 2. Configure validation:

The example scripts log validation samples by default. Set `VALIDATION_DATASET_FILE` to a real validation JSON, such as:

```bash
VALIDATION_DATASET_FILE="examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/validation_64.json"
```

### 3. Configure and run distillation:

#### For DMD-only distillation:
```bash
sbatch examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/distill_dmd_t2v_1.3B.slurm
```

#### For DMD+VSA distillation on Wan2.1-T2V-1.3B:
```bash
sbatch examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/distill_dmd_VSA_t2v_1.3B.slurm
```

#### For DMD+VSA distillation on Wan2.1-T2V-14B:
```bash
sbatch examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/distill_dmd_VSA_t2v_14B.slurm
```
