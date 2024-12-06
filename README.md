# FastVideo

<div align="center">
  <a href=""><img src="https://img.shields.io/static/v1?label=API:H100&message=Replicate&color=pink"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>
<br>
<div align="center">
<img src=assets/logo.png width="50%"/>
</div>

As the model size and sequence length of video diffusion model continues to increase, FastVideo aims to build an efficient fintuning and distillation platform for current and future big video generation models. Fastvideo use FSDP, sequence parallel, and selective activation checkpoint to shard the model and data to multiple GPUs. Fastvideo also provides Lora for low-memory finetuning. We currently support mochi and will add more models in the future.


## News

- 🔥 **News**: ```2024/12/08```:Added Replicate Demo and API for Fastvideo [![Replicate]()]().

- 🔥 **News**: ```2024/12/06```: We have open-sourced `FastMochi model` and its training script.



## Fast and High-Quality Text-to-video Generation

### 8-Step Results of T2V-Turbo

<table class="center">
  <td><img src=assets/8steps/1.gif width="320"></td></td>
  <td><img src=assets/8steps/2.gif width="320"></td></td></td>
  <tr>
  <td style="text-align:center;" width="320">tmp</td>
  <td style="text-align:center;" width="320">tmp</td>
  <tr>
</table >


## Table of Contents

Jump to a specific section:

- [🔧 Installation](#-installation)
- [⬇️ Download Weights](#download-weights)
- [🚀 Inference](#-inference)
- [🎯 Distill](#-distill)
- [⚡ Lora Finetune](#-lora-finetune)
- [📚 Citation](#-citation)

## 🔧 Installation

```
conda create -n fastmochi python=3.10.0 -y && conda activate fastmochi
pip3 install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo && pip install -e .
```

## ⬇️ Download Weights

Use [scripts/download_hf.py](scripts/download_hf.py) to download the hugging-face style model to a local directory. Use it like this:
```bash
python scripts/download_hf.py --repo_id=genmo/mochi-1-preview --local_dir=data/mochi --repo_type=model
```


## 🚀 Inference

Start the gradio UI with
```
python3 ./demos/gradio_ui.py --model_dir weights/ --cpu_offload
```
We also provide CLI inference script featured with sequence parallelism.

```
export NUM_GPUS=4

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_path data/prompt.txt \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 8 \
    --guidance_scale 4.5 \
    --output_path outputs_video/demo_video \
    --shift 8 \
    --seed 42 \
    --scheduler_type "pcm_linear_quadratic"
```

For the mochi style, simply following the scripts list in mochi repo.

```
git clone https://github.com/genmoai/mochi.git
cd mochi

# install env
...

python3 ./demos/cli.py --model_dir weights/ --cpu_offload
```


## 🎯 Distill

## 💰Hardware requirement

-  VRAM is required for both distill 10B mochi model

To launch distillation, you will first need to prepare data in the following formats

```bash
asset/example_data
├── AAA.txt
├── AAA.png
├── BCC.txt
├── BCC.png
├── ......
├── CCC.txt
└── CCC.png
```

We provide a dataset example here. First download testing data. Use [scripts/download_hf.py](scripts/download_hf.py) to download the data to a local directory. Use it like this:
```bash
python scripts/download_hf.py --repo_id=Stealths-Video/Merge-425-Data --local_dir=data/Merge-425-Data --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/validation_embeddings --local_dir=data/validation_embeddings --repo_type=dataset
```

Then the distillation can be launched by:

```
bash scripts/distill_t2v.sh
```


## ⚡ Lora Finetune


## 💰Hardware requirement

-  VRAM is required for both distill 10B mochi model

To launch finetuning, you will first need to prepare data in the following formats.



Then the finetuning can be launched by:

```
bash scripts/lora_finetune.sh
```

## Acknowledgement
We learned from and reused code from the following projects: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), and [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).