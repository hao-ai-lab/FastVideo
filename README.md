<div align="center">
<img src=assets/logo.jpg width="30%"/>
</div>

FastVideo is an open-source framework for accelerating large video diffusion model.
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

<p align="center">
    🤗 <a href="https://huggingface.co/FastVideo/FastMochi-diffuser" target="_blank">FastMochi</a> | 🤗 <a href="https://huggingface.co/FastVideo/FastHunyuan"  target="_blank">FastHunyuan</a> 
</p>

FastVideo currently offers: (with more to come)

- FastHunyuan and FastMochi: consistency distilled video diffusion models for 8x inference speedup.
- First open video DiT distillation recipes based on [PCM](https://github.com/G-U-N/Phased-Consistency-Model).
- Scalable training with FSDP, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.
- Memory efficient finetuning with LoRA, precomputed latents, and precomputed text embeddings.


## Change Log

- ```2024/12/17```: `FastVideo` v0.1 is released.


## 🔧 Installation
The code is tested on Python 3.10.0, CUDA 12.1 and H100.
```
./env_setup.sh fastvideo
```

## 🚀 Inference
We recommend using a GPU with 80GB of memory. To run the inference, use the following command:
### FastHunyuan
```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan --local_dir=data/FastHunyuan --repo_type=model
# CLI inference
sh scripts/inference/inference_hunyuan.sh
```
You can also inference FastHunyuan in the [official Hunyuan github](https://github.com/Tencent/HunyuanVideo).
### FastMochi

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastMochi-diffusers --local_dir=data/FastMochi-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_mochi_sp.sh
```

## Distillation
Please refer to the [distillation guide](docs/distillation.md).

## Finetuning
Please refer to the [finetuning guide](docs/finetuning.md).

## Development Plan


## Acknowledgement
We learned and reused code from the following projects: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), and [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).
