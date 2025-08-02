(v0-finetune)=
# 🧠 Finetune
## ⚡ Full Finetune
Ensure your data is prepared and preprocessed in the format specified in [data_preprocess.md](#v0-data-preprocess). For convenience, we also provide a mochi preprocessed Black Myth Wukong data that can be downloaded directly:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Mochi-Black-Myth --local_dir=data/Mochi-Black-Myth --repo_type=dataset
```

Download the original model weights as specified in the [Distillation Section](../distillation/dmd.md):

Then you can run the finetune with:

```
bash scripts/finetune/finetune_mochi.sh # for mochi
```

**Note that for finetuning, we did not tune the hyperparameters in the provided script.**
## ⚡ Finetune with VSA
Follow [data_preprocess.md](#v0-data-preprocess) to get parquet files for preproccessed latent, and then run:

```bash
bash scripts/finetune/finetune_v1_VSA.sh
```

## ⚡ Lora Finetune

Hunyuan supports Lora fine-tuning of videos up to 720p. Demos and prompts of Black-Myth-Wukong can be found in [here](https://huggingface.co/FastVideo/Hunyuan-Black-Myth-Wukong-lora-weight). You can download the Lora weight through:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Hunyuan-Black-Myth-Wukong-lora-weight --local_dir=data/Hunyuan-Black-Myth-Wukong-lora-weight --repo_type=model
```

### Minimum Hardware Requirement
- 40 GB GPU memory each for 2 GPUs with lora.
- 30 GB GPU memory each for 2 GPUs with CPU offload and lora.

Currently, both Mochi and Hunyuan models support Lora finetuning through diffusers. To generate personalized videos from your own dataset, you'll need to follow three main steps: dataset preparation, finetuning, and inference.

### Dataset Preparation
We provide scripts to better help you get started to train on your own characters!
You can run this to organize your dataset to get the videos2caption.json before preprocess. Specify your video folder and corresponding caption folder (caption files should be .txt files and have the same name with its video):

```
python scripts/dataset_preparation/prepare_json_file.py --video_dir data/input_videos/ --prompt_dir data/captions/ --output_path data/output_folder/videos2caption.json --verbose
```

Also, we provide script to resize your videos:

```
python scripts/data_preprocess/resize_videos.py
```

### Finetuning
After basic dataset preparation and preprocess, you can start to finetune your model using Lora:

```
bash scripts/finetune/finetune_hunyuan_hf_lora.sh
```

### Inference
For inference with Lora checkpoint, you can run the following scripts with additional parameter `--lora_checkpoint_dir`:

```
bash scripts/inference/inference_hunyuan_hf.sh
```

**We also provide scripts for Mochi in the same directory.**

### Finetune with Both Image and Video
Our codebase support finetuning with both image and video.

```bash
bash scripts/finetune/finetune_hunyuan.sh
bash scripts/finetune/finetune_mochi_lora_mix.sh
```

For Image-Video Mixture Fine-tuning, make sure to enable the `--group_frame` option in your script.
