<div align="center">
<img src=assets/logo.jpg width="30%"/>
</div>

FastVideo is a lightweight framework for accelerating large video diffusion models.

<p align="center">
    | <a href="https://hao-ai-lab.github.io/FastVideo"><b>Documentation</b></a> | 🤗 <a href="https://huggingface.co/FastVideo/FastHunyuan"  target="_blank"><b>FastHunyuan</b></a>  | 🤗 <a href="https://huggingface.co/FastVideo/FastMochi-diffusers" target="_blank"><b>FastMochi</b></a> | 🟣💬 <a href="https://join.slack.com/t/fastvideo/shared_invite/zt-2zf6ru791-sRwI9lPIUJQq1mIeB_yjJg" target="_blank"> <b>Slack</b> </a> |
</p>

https://github.com/user-attachments/assets/79af5fb8-707c-4263-b153-9ab2a01d3ac1

FastVideo currently offers: (with more to come)

- [NEW!] V1 inference API available. Full announcement coming soon!
- [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).
- FastHunyuan and FastMochi: consistency distilled video diffusion models for 8x inference speedup.
- First open distillation recipes for video DiT, based on [PCM](https://github.com/G-U-N/Phased-Consistency-Model).
- Support distilling/finetuning/inferencing state-of-the-art open video DiTs: 1. Mochi 2. Hunyuan.
- Scalable training with FSDP, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.
- Memory efficient finetuning with LoRA, precomputed latent, and precomputed text embeddings.

Dev in progress and highly experimental.

## Change Log
- ```2025/02/20```: FastVideo now supports STA on [StepVideo](https://github.com/stepfun-ai/Step-Video-T2V) with 3.4X speedup!
- ```2025/02/18```: Release the inference code and kernel for [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).
- ```2025/01/13```: Support Lora finetuning for HunyuanVideo.
- ```2024/12/25```: Enable single 4090 inference for `FastHunyuan`, please rerun the installation steps to update the environment.
- ```2024/12/17```: `FastVideo` v1.0 is released.

## 🔧 Installation from source
The code is tested on Python 3.10-3.12, CUDA 12.4 and H100.

```
# Clone FastVideo
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

# Install FastVideo
pip install -e .

# Install Flash Attention (optional)
pip install flash-attn==2.7.0.post2
```

To try Sliding Tile Attention (optional), please follow the instruction in [csrc/sliding_tile_attention/README.md](csrc/sliding_tile_attention/README.md) to install STA.

You can also install the Sliding Tile Attention package using

```
pip install st_attn==0.0.4
```

## 🚀 Inference
### Inference StepVideo with Sliding Tile Attention
First, download the model:

```
python scripts/huggingface/download_hf.py --repo_id=stepfun-ai/stepvideo-t2v --local_dir=data/stepvideo-t2v --repo_type=model
```

Use the following scripts to run inference for StepVideo. When using STA for inference, the generated videos will have dimensions of 204×768×768 (currently, this is the only supported shape).

```bash
sh scripts/inference/inference_stepvideo_STA.sh # Inference stepvideo with STA
sh scripts/inference/inference_stepvideo.sh # Inference original stepvideo
```

### Inference HunyuanVideo with Sliding Tile Attention
First, download the model:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/hunyuan --local_dir=data/hunyuan --repo_type=model
```

We provide two examples in the following script to run inference with STA + [TeaCache](https://github.com/ali-vilab/TeaCache) and STA only.

```bash
sh scripts/inference/inference_hunyuan_STA.sh
```

### Video Demos using STA + Teacache
Visit our [demo website](https://fast-video.github.io/) to explore our complete collection of examples. We shorten a single video generation process from 945s to 317s on H100.

### Inference FastHunyuan on single RTX4090
We now support NF4 and LLM-INT8 quantized inference using BitsAndBytes for FastHunyuan. With NF4 quantization, inference can be performed on a single RTX 4090 GPU, requiring just 20GB of VRAM.

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan-diffusers --local_dir=data/FastHunyuan-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_hunyuan_hf_quantization.sh
```

For more information about the VRAM requirements for BitsAndBytes quantization, please refer to the table below (timing measured on an H100 GPU):

| Configuration                  | Memory to Init Transformer | Peak Memory After Init Pipeline (Denoise) | Diffusion Time | End-to-End Time |
|--------------------------------|----------------------------|--------------------------------------------|----------------|-----------------|
| BF16 + Pipeline CPU Offload    | 23.883G                   | 33.744G                                    | 81s            | 121.5s          |
| INT8 + Pipeline CPU Offload    | 13.911G                   | 27.979G                                    | 88s            | 116.7s          |
| NF4 + Pipeline CPU Offload     | 9.453G                    | 19.26G                                     | 78s            | 114.5s          |

For improved quality in generated videos, we recommend using a GPU with 80GB of memory to run the BF16 model with the original Hunyuan pipeline. To execute the inference, use the following section:

### FastHunyuan

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan --local_dir=data/FastHunyuan --repo_type=model
# CLI inference
bash scripts/inference/inference_hunyuan.sh
```

You can also inference FastHunyuan in the [official Hunyuan github](https://github.com/Tencent/HunyuanVideo).

### FastMochi

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastMochi-diffusers --local_dir=data/FastMochi-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_mochi_sp.sh
```

## 📑 Development Plan

- More distillation methods
  - [ ] Add Distribution Matching Distillation
- More models support
  - [ ] Add CogvideoX model
- Code update
  - [ ] fp8 support
  - [ ] faster load model and save model support

## 🤝 Contributing

We welcome all contributions. Please run `bash format.sh --all` before submitting a pull request.

## 🔧 Testing
Run `pytest` to verify the data preprocessing, checkpoint saving, and sequence parallel pipelines. We recommend adding corresponding test cases in the `test` folder to support your contribution.

## Acknowledgement
We learned and reused code from the following projects: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), and [xDiT](https://github.com/xdit-project/xDiT).

We thank MBZUAI and Anyscale for their support throughout this project.

## Citation
If you use FastVideo for your research, please cite our paper:

```bibtex
@misc{zhang2025fastvideogenerationsliding,
      title={Fast Video Generation with Sliding Tile Attention},
      author={Peiyuan Zhang and Yongqi Chen and Runlong Su and Hangliang Ding and Ion Stoica and Zhenghong Liu and Hao Zhang},
      year={2025},
      eprint={2502.04507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.04507},
}
@misc{ding2025efficientvditefficientvideodiffusion,
      title={Efficient-vDiT: Efficient Video Diffusion Transformers With Attention Tile},
      author={Hangliang Ding and Dacheng Li and Runlong Su and Peiyuan Zhang and Zhijie Deng and Ion Stoica and Hao Zhang},
      year={2025},
      eprint={2502.06155},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.06155},
}
```
