This directory contain e2e examples scripts for finetuning Wan2.1 T2v. 

Execute the following commands from `FastVideo/` to run training:

- Download crush-smol dataset:
`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/download_dataset.sh`
- Preprocess the videos and captions into latents:
`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/preprocess_wan_data_t2v.sh`
- Edit the following file and run finetuning:
`bash examples/training/finetune/wan_t2v_1_3b/crush_smol/finetune_t2v.sh`
