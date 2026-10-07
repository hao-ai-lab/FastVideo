# Finetuning Wan2.1 to make it work with VSA to accelerate 

These are e2e example scripts for finetuning Wan2.1 T2V with VSA to accelerate inference.

## Execute the following commands from `FastVideo/` to run training:

## Make sure you have installed VSA

Go to fastvideo-kernel/README.md for instructions.

### Download the synthetic dataset:

```bash
bash examples/datasets/wan-syn/download_dataset_480p.sh
bash examples/datasets/wan-syn/download_dataset_720p.sh
```
### Slurm script to train the model
```bash
sbatch examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/T2V-14B-VSA.slurm
```
```bash
sbatch examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/I2V-14B-VSA.slurm
```