# Wan2.1-T2V-1.3B Distill Example
These are end-to-end example scripts for distilling Wan2.1 T2V 1.3B model using DMD-only and DMD+VSA methods.

### 0. Make sure you have installed VSA

```bash
pip install vsa
```

### 1. Download dataset:
```bash
bash examples/distill/Wan-Syn-480P/download_dataset.sh
```

### 2. Configure and run distillation:

#### For DMD-only distillation:
```bash
sbatch examples/distill/Wan-Syn-480P/distill_dmd_t2v.slurm
```

#### For DMD+VSA distillation:
```bash
sbatch examples/distill/Wan-Syn-480P/distill_dmd_VSA_t2v.slurm
```
