# Wan2.1-T2V-1.3B Crush-Smol Example
These are e2e example scripts for finetuning Wan2.1 T2V 1.3B on the crush-smol dataset.

## Execute the following commands from `FastVideo/` to run training:

### Download crush-smol dataset:

`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/download_dataset.sh`

### Preprocess the videos and captions into latents:

`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/preprocess_wan_data_t2v.sh`

### Edit the following file and run finetuning:

`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/finetune_t2v.sh`
