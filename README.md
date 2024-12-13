# FastVideo

<div align="center">
  <a href=""><img src="https://img.shields.io/static/v1?label=API:H100&message=Replicate&color=pink"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>
<br>
<div align="center">
<img src=assets/logo.png width="50%"/>
</div>

FastVideo is a scalable framework for post-training video diffusion models, addressing the growing challenges of fine-tuning, distillation, and inference as model sizes and sequence lengths increase. As a first step, it provides an efficient script for distilling and fine-tuning the 10B Mochi model, with plans to expand features and support for more models.

### Features

- FastMochi, a distilled Mochi model that can generate videos with merely 8 sampling steps.
- Finetuning with FSDP (both master weight and ema weight), sequence parallelism, and selective gradient checkpointing.
- LoRA coupled with pecomputed the latents and text embedding for minumum memory consumption.
- Finetuning with both image and videos.

## Change Log


- ```2024/12/13```: `FastVideo` v0.0.1 is released.


## Fast and High-Quality Text-to-video Generation

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



## Table of Contents

Jump to a specific section:

- [🔧 Installation](#-installation)
- [🚀 Inference](#-inference)
- [🧱 Data Preprocess](#-data-preprocess)
- [🎯 Distill](#-distill)
- [⚡ Finetune](#-lora-finetune)


## 🔧 Installation

```
conda create -n fastvideo python=3.10.0 -y && conda activate fastvideo
pip3 install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 
pip install "git+https://github.com/huggingface/diffusers.git@bf64b32652a63a1865a0528a73a13652b201698b"
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo && pip install -e .
```




## 🚀 Inference

Use [scripts/huggingface/download_hf.py](scripts/huggingface/download_hf.py) to download the hugging-face style model to a local directory. Use it like this:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastMochi --local_dir=data/FastMochi --repo_type=model
```


Start the gradio UI with
```
python fastvideo/demo/gradio_web_demo.py --model_path data/FastMochi
```

We also provide CLI inference script featured with sequence parallelism.

```
export NUM_GPUS=4

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/FastMochi \
    --prompt_path assets/prompt.txt \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 8 \
    --guidance_scale 1.5 \
    --output_path outputs_video/demo_video \
    --seed 12345 \
    --scheduler_type "pcm_linear_quadratic" \
    --linear_threshold 0.1 \
    --linear_range 0.75
```

## 🧱 Data Preprocess

To reduce the memory cost and time consumption caused by VAE and T5 during the distillation and finetune, we offload the VAE and T5 preprocess media part to Data Preprocess section.
For data preprocess, we need to prepare a source folder for the media we wish to use and a json file for the source information of these media. 

### Sample for data preprocess
We provide a small sample dataset for you to start with, download the source media with command:
```
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Image-Vid-Finetune-Src --local_dir=data/Image-Vid-Finetune-Src --repo_type=dataset
```
For preprocess dataset for mochi finetune/distill run:
```
bash scripts/preprocess/preprocess_mochi_data.sh
```
For preprocess dataset for hunyuan finetune/distill run:

```
bash scripts/preprocess/preprocess_hunyuan_data.sh
```
The preprocessed dataset will be stored in `Image-Vid-Finetune-Mochi` or `Image-Vid-Finetune-HunYuan` correspondingly.

### Create personal dataset
If you wish to creat you own dataset for finetune or distill, please pay attention to the following format: 

Use a txt file to contain the source folder for media and the json file for meta information

```
path_to_media_source_foder,path_to_json_file
```
The content of the json file is a list with each item corresponding to a media source.

For image media, the json item needs to follow the following format:
```
{
    "path": "0.jpg",
    "cap": ["captions"]
}
```
For video media, the json item needs to follow the following format:
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
Adjust the `DATA_MERGE_PATH` and `OUTPUT_DIR` in `scripts/preprocess/preprocess_****_data.sh` correspondingly and run
```
bash scripts/preprocess/preprocess_****_data.sh
```
The preprocessed data will be put into the `OUTPUT_DIR` and the `videos2caption.json` can be used in finetune and distill scripts.

## 🎯 Distill

## 💰Hardware requirement

-  VRAM is required for both distill 10B mochi model


We provide a dataset example here. First download testing data. Use [scripts/huggingface/download_hf.py](scripts/huggingface/download_hf.py) to download the data to a local directory. Use it like this:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Mochi-425-Data --local_dir=data/Mochi-425-Data --repo_type=dataset
python scripts/huggingface/download_hf.py --repo_id=FastVideo/validation_embeddings --local_dir=data/validation_embeddings --repo_type=dataset
```

Then the distillation can be launched by:

```
bash scripts/distill_t2v.sh
```


## ⚡ Finetune
To launch finetuning, you will need to prepare data in the according to formats described in section [Data Preprocess](#-data-preprocess). 

If you are doing image-video mixture finetuning, make sure `--group_frame` is in your script.

Then run the finetune with
```
bash scripts/finetune/finetune_mochi.sh # for mochi
bash scripts/finetune/finetune_hunyuan.sh # for hunyuan
```


## 💰Hardware requirement

-  VRAM is required for both distill 10B mochi model

Then the finetuning can be launched by:

```
bash scripts/lora_finetune.sh
```

## Acknowledgement
We learned from and reused code from the following projects: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), and [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).
