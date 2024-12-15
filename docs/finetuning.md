
## ⚡ Finetune

We provide full weight finetune for both Mochi and Yunyuan model, we also provide Image-Video mixture finetune to enhance the multi-style finetune for users.

To launch finetuning, you will need to prepare data in the according to formats described in section [Data Preprocess](#-data-preprocess). 

If you are doing image-video mixture finetuning, make sure `--group_frame` is in your script.

Then run the finetune with:
```
bash scripts/finetune/finetune_mochi.sh # for mochi
bash scripts/finetune/finetune_hunyuan.sh # for hunyuan
```

## Lora Finetune

Currently, we only provide Lora Finetune for Mochi model, the command for Lora Finetune is
```
bash scripts/finetune/finetune_mochi_lora.sh
```

### 💰Hardware requirement

- 72G VRAM is required for finetuning 10B mochi model.

