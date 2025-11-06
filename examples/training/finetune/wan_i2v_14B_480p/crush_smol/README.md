# Wan2.1-I2V-14B-480P Crush-Smol Example
These are e2e examples scripts for finetuning Wan2.1 I2V 14B 480P on the crush-smol dataset.

## Execute the following commands from `FastVideo/` to run training:

### Download crush-smol dataset:

`bash examples/training/finetune/wan_i2v_14b_480p/crush_smol/download_dataset.sh`

### Preprocess the videos and captions into latents:

`bash examples/training/finetune/wan_i2v_14b_480p/crush_smol/preprocess_wan_data_i2v.sh`

### Edit the following file and run finetuning:

`bash examples/training/finetune/wan_i2v_14b_480p/crush_smol/finetune_i2v.sh`