# FastVideo

<div align="center">
<img src=assets/logo.jpg width="30%"/>
</div>

FastVideo is an open framework for distilling, training, and inferencing large video diffusion model.
### What is this?

As state-of-the-art video diffusion models grow in size and sequence length, their become harder to use. For instance, sampling a 5-second 480p video with Mochi takes 20 minutes on an A100 80G GPU, making both inference and fine-tuning prohibitively slow for many users.

To make those wonderful video diffusion model more **accessible**, FastVideo is designed to address these challenges by making large video diffusion models efficient to train and fast to infer. It is an open framework for distilling, training, and inferencing large-scale video diffusion models.

We introduce FastMochi and FastHunyuan, distilled versions of the Mochi and Hunyuan video diffusion models. FastMochi achieves high-quality sampling with just 8 inference steps. FastHunyuan maintains sampling quality with only 4 inference steps.

### What can I do with FastVideo?
FastVideo provides a comprehensive pipeline for training, distilling, and inferencing video diffusion models. Key capabilities include:

- Support for FSDP, sequence parallelism, and selective gradient checkpointing for efficient training. Our code scales to 64 GPUs without any code changes.
- LoRA finetuning coupled with precomputed latents and text embeddings for minimal memory usage.
- Finetuning with both image and videos.

<table style="margin-left: auto; margin-right: auto; border: none;">
  <tr>
    <td>
      <img src="assets/8steps/mochi-demo.gif" width="640" alt="Mochi Demo">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Get 8X diffusion boost for Mochi with FastVideo
    </td>
  </tr>
</table>


## Change Log

- ```2024/12/16```: `FastVideo` v0.1 is released.


## 🔧 Installation

- Python >= 3.10.0
- Cuda >= 12.1

```
./env_setup.sh fastvideo
conda activate fastvideo
```



## 🚀 Inference

### FastMochi

First, download the model weight:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastMochi-diffusers --local_dir=data/FastMochi-diffusers --repo_type=model
```
For a gradio web demo, use:

```
python demo/gradio_web_demo.py --model_path data/FastMochi-diffusers --guidance_scale 1.5 --num_frames 163
```

For CLI inference, use:

```
bash scripts/inference/inference_mochi_sp.sh
```

### FastHunyuan


## 🧱 Data Preprocess

To reduce the memory cost and time consumption caused by VAE and T5 during distillation and finetuning, we offload the VAE and T5 preprocess media part to the Data Preprocess section.
For data preprocessing, we need to prepare a source folder for the media we wish to use and a JSON file for the source information of these media.

### Sample for Data Preprocess

We provide a small sample dataset for you to start with, download the source media with command:
```
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Image-Vid-Finetune-Src --local_dir=data/Image-Vid-Finetune-Src --repo_type=dataset
```
To preprocess dataset for finetune/distill run:

```
bash scripts/preprocess/preprocess_mochi_data.sh # for mochi
bash scripts/preprocess/preprocess_hunyuan_data.sh # for hunyuan
```

The preprocessed dataset will be stored in `Image-Vid-Finetune-Mochi` or `Image-Vid-Finetune-HunYuan` correspondingly.

### Create Custom Dataset

If you wish to create your own dataset for finetuning or distillation, please pay attention to the following format:

Use a txt file to contain the source folder for media and the JSON file for meta information:

```
path_to_media_source_foder,path_to_json_file
```
The content of the JSON file is a list with each item corresponding to a media source.

For image media, the JSON item needs to follow this format:
```
{
    "path": "0.jpg",
    "cap": ["captions"]
}
```
For video media, the JSON item needs to follow this format:
```
{
    "path": "1.mp4",
    "resolution": {
      "width": 848,
      "height": 480
    },
    "fps": 30.0,
    "duration": 6.033333333333333,
    "cap": [
      "caption"
    ]
  }
```
Adjust the `DATA_MERGE_PATH` and `OUTPUT_DIR` in `scripts/preprocess/preprocess_****_data.sh` accordingly and run:
```
bash scripts/preprocess/preprocess_****_data.sh
```
The preprocessed data will be put into the `OUTPUT_DIR` and the `videos2caption.json` can be used in finetune and distill scripts.

## 🎯 Distill

We provide a dataset example here. First download testing data. Use [scripts/huggingface/download_hf.py](scripts/huggingface/download_hf.py) to download the data to a local directory. Use it like this:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Mochi-425-Data --local_dir=data/Mochi-425-Data --repo_type=dataset
python scripts/huggingface/download_hf.py --repo_id=FastVideo/validation_embeddings --local_dir=data/validation_embeddings --repo_type=dataset
```

Then the distillation can be launched by:

```
bash scripts/distill/distill_mochi.sh # for mochi
bash scripts/distill/distill_hunyuan.sh # for hunyuan
```


## ⚡ Finetune


### 💰Hardware requirement

- 72G VRAM is required for finetuning 10B mochi model.

To launch finetuning, you will need to prepare data in the according to formats described in section [Data Preprocess](#-data-preprocess). 

If you are doing image-video mixture finetuning, make sure `--group_frame` is in your script.

Then run the finetune with:
```
bash scripts/finetune/finetune_mochi.sh # for mochi
bash scripts/finetune/finetune_hunyuan.sh # for hunyuan
```

## Acknowledgement
We learned and reused code from the following projects: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), and [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).
