
## ⚡ Finetune

We support full fine-tuning for both the Mochi and Hunyuan models. Additionally, we provide Image-Video Mix finetuning.


Ensure your data is prepared and preprocessed in the format specified in the [Data Preprocess](#-data-preprocess). 
Download the original model weights with:
```bash
python scripts/huggingface/download_hf.py --repo_id=genmo/mochi-1-preview --local_dir=data/mochi --repo_type=model
python scripts/huggingface/download_hf.py --repo_id=FastVideo/hunyuan --local_dir=data/hunyuan --repo_type=model
```


FastVideo/BLACK-MYTH-YQ
Then run the finetune with:
```
bash scripts/finetune/finetune_mochi.sh # for mochi
bash scripts/finetune/finetune_hunyuan.sh # for hunyuan
```
For Image-Video Mixture Fine-tuning, make sure to enable the --group_frame option in your script.


## Lora Finetune

Currently, we only provide Lora Finetune for Mochi model, the command for Lora Finetune is
```
bash scripts/finetune/finetune_mochi_lora.sh
```

### 💰Hardware requirement

- 72G VRAM is required for finetuning 10B mochi model.

