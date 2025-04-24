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
- ```2024/12/17```: `FastVideo` v0.0.1 is released.

## Getting Started

- [Install FastVideo](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html)
- [Design Overview](https://hao-ai-lab.github.io/FastVideo/design/overview.html)
- [Contribution Guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html)

### Inference
- [Quick Start](https://hao-ai-lab.github.io/FastVideo/inference/examples/basic.html)
- V1 Inference API Guide (Coming soon!)

### Distillation and Finetuning
- [Distillation Guide](https://hao-ai-lab.github.io/FastVideo/training/distillation.html)
- [Finetuning Guide](https://hao-ai-lab.github.io/FastVideo/training/finetuning.html)

### Deprecated APIs
- [V0 Inference (Deprecated)](https://hao-ai-lab.github.io/FastVideo/inference/v0_inference.html)

## 📑 Development Plan

<!-- - More distillation methods -->
  <!-- - [ ] Add Distribution Matching Distillation -->
- More models support
  <!-- - [ ] Add CogvideoX model -->
  - [ ] Add StepVideo to V1
- Optimization features
  - [ ] Teacache in V1
  - [ ] SageAttention in V1
- Code updates
  - [ ] V1 Configuration API
  - [ ] Support Training in V1
  <!-- - [ ] fp8 support -->
  <!-- - [ ] faster load model and save model support -->

## 🤝 Contributing

We welcome all contributions. Please check out our guide [here](https://hao-ai-lab.github.io/FastVideo/developer_guide/overview.html)

## Acknowledgement
We learned and reused code from the following projects:
- [PCM](https://github.com/G-U-N/Phased-Consistency-Model)
- [diffusers](https://github.com/huggingface/diffusers)
- [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)
- [xDiT](https://github.com/xdit-project/xDiT)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

We thank MBZUAI and [Anyscale](https://www.anyscale.com/) for their support throughout this project.

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
