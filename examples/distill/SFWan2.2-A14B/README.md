# Self-Forcing Distillation for SFWan2.2 T2V A14B

This example distills the Wan2.2 T2V A14B model with the self-forcing (SFwan) objective. It keeps the DMD2 teacher/critic setup but adds causal self-forcing windows for stronger temporal consistency.

## Run the recipe
1. Point the script at your training parquet and validation JSON paths.
2. Submit the SLURM job:
   ```bash
   sbatch examples/distill/SFWan2.2-A14B/distill_dmd.sh
   ```

The script defaults to Wan2.1 checkpoints for teacher/critic slots—swap in Wan2.2 paths if you have them available.
