# Self-Forcing Distillation for SFWan2.1 T2V 1.3B

These scripts demonstrate self-forcing distillation (SFwan) for the causal Wan2.1 T2V 1.3B model. The workflow mirrors DMD2 while injecting self-forcing blocks so the student can autoregressively refine later frames.

## Run the recipe
1. Download the preprocessed text-video dataset:
   ```bash
   bash examples/distill/SFWan2.1-T2V/download_dataset.sh
   ```
2. (Optional) Regenerate parquet shards locally:
   ```bash
   bash examples/distill/SFWan2.1-T2V/preprocess_data.sh
   ```
3. Launch self-forcing distillation with your cluster settings:
   ```bash
   sbatch examples/distill/SFWan2.1-T2V/distill_dmd_t2v_1.3B.sh
   ```

Update the dataset paths and wandb credentials inside the script before running on your environment.
