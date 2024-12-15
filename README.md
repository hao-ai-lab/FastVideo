<div align="center">
<img src=assets/logo.jpg width="30%"/>
</div>

FastVideo is an open framework for distilling, training, and inferencing large video diffusion model.
<div align="center">
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
  </div>

### What is this?

As state-of-the-art video diffusion models grow in size and sequence length, their become prohibitive to use. For instance, sampling a 5-second 720P video with Hunyuan takes 13 minutes on 4 X A100. FastVideo aim to make large video diffusion models fast to infer and efficient to train, and thus making them more **accessible**. 

We introduce FastMochi and FastHunyuan, distilled versions of the Mochi and Hunyuan video diffusion models. FastMochi achieves high-quality sampling with just 8 inference steps. FastHunyuan maintains sampling quality with only 4 inference steps.



### What can I do with FastVideo?
Other than the distilled weight, FastVideo provides a pipeline for training, distilling, and inferencing video diffusion models. Key capabilities include:

- **Scalable**: FastVideo supports FSDP, sequence parallelism, and selective gradient checkpointing. Our code seamlessly scales to 64 GPUs in our test.
- **Memory Efficient**: FastVideo supports LoRA finetuning coupled with precomputed latents and text embeddings for minimal memory usage.
- **Variable Sequence length**: You can finetuning with both image and videos.

## Change Log

- ```2024/12/16```: `FastVideo` v0.1 is released.


## 🔧 Installation
The code is tested on Python 3.10.0 and CUDA 12.1.

```
./env_setup.sh fastvideo
conda activate fastvideo
```

## 🚀 Inference
We recommend using a GPU with 80GB of memory. To run the inference, use the following command:
### FastHunyuan
```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan --local_dir=data/FastHunyuan --repo_type=model
# change the gpu count inside the script
sh scripts/inference/inference_hunyuan.sh
```

### FastMochi
You can use FastMochi

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastMochi-diffusers --local_dir=data/FastMochi-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_mochi_sp.sh
# Gradio web dem
python demo/gradio_web_demo.py --model_path data/FastMochi-diffusers --guidance_scale 1.5 --num_frames 163
```

## Distillation
Please refer to the [distillation guide](docs/distilation.md).

## Finetuning
Please refer to the [finetuning guide](docs/finetuning.md).

## Development Plan


## Acknowledgement
We learned and reused code from the following projects: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), and [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).
