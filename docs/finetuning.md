
## ⚡ Full Finetune

Ensure your data is prepared and preprocessed in the format specified in [data_preprocess.md](docs/data_preprocess.md). For convenience, we also provide a mochi preprocessed Black Myth Wukong data that can be downloaded directly:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Mochi-Black-Myth --local_dir=data/Mochi-Black-Myth --repo_type=dataset
```
Download the original model weights with:
```bash
python scripts/huggingface/download_hf.py --repo_id=genmo/mochi-1-preview --local_dir=data/mochi --repo_type=model
python scripts/huggingface/download_hf.py --repo_id=FastVideo/hunyuan --local_dir=data/hunyuan --repo_type=model
```

Then you can run the finetune with:
```
bash scripts/finetune/finetune_mochi.sh # for mochi
```
**Note that we did not tune the hyperparameters in the provided script**

## ⚡ Lora Finetune

Currently, we only provide Lora Finetune for Mochi model, the command for Lora Finetune is
```
bash scripts/finetune/finetune_mochi_lora.sh
```
### Minimum Hardware Requirement
- 40 GB GPU memory each for 2 GPUs with lora
- 30 GB GPU memory each for 2 GPUs with CPU offload and lora.

## Finetune with Both Image and Video
Our codebase support finetuning with both image and video. 

```bash
bash scripts/finetune/finetune_hunyuan.sh
bash scripts/finetune/finetune_mochi_lora_mix.sh
```
For Image-Video Mixture Fine-tuning, make sure to enable the --group_frame option in your script.




