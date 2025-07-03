<div align="center">
<img src=assets/logo.jpg width="30%"/>
</div>

**FastVideo is a unified framework for accelerated video generation.**

It features a clean, consistent API that works across popular video models, making it easier for developers to author new models and incorporate system- or kernel-level optimizations.
With FastVideo's optimizations, you can achieve more than 3x inference improvement compared to other systems.

<p align="center">
    | <a href="https://hao-ai-lab.github.io/FastVideo"><b>Documentation</b></a> | <a href="https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html"><b> Quick Start</b></a> | 🤗 <a href="https://huggingface.co/FastVideo/FastHunyuan"  target="_blank"><b>FastHunyuan</b></a>  | 🤗 <a href="https://huggingface.co/FastVideo/FastMochi-diffusers" target="_blank"><b>FastMochi</b></a> | 🟣💬 <a href="https://join.slack.com/t/fastvideo/shared_invite/zt-2zf6ru791-sRwI9lPIUJQq1mIeB_yjJg" target="_blank"> <b>Slack</b> </a> |
</p>

<div align="center">
<img src=assets/perf.png width="90%"/>
</div>

## NEWS
- ```2025/06/14```: Release finetuning and inference code for [VSA](https://arxiv.org/pdf/2505.13389)
- ```2025/04/24```: [FastVideo V1](https://hao-ai-lab.github.io/blogs/fastvideo/) is released!
- ```2025/02/18```: Release the inference code for [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).

## Key Features

FastVideo has the following features:
- State-of-the-art performance optimizations for inference
  - [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507)
  - [TeaCache](https://arxiv.org/pdf/2411.19108)
  - [Sage Attention](https://arxiv.org/abs/2410.02367)
- Cutting edge models
  - Wan2.1 T2V, I2V
  - HunyuanVideo
  - FastHunyuan: consistency distilled video diffusion models for 8x inference speedup.
  - StepVideo T2V
- Distillation support
  - Recipes for video DiT, based on [PCM](https://github.com/G-U-N/Phased-Consistency-Model).
  - Support distilling/finetuning/inferencing state-of-the-art open video DiTs: 1. Mochi 2. Hunyuan.
- Scalable training with FSDP, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.
- Memory efficient finetuning with LoRA, precomputed latent, and precomputed text embeddings.

## Getting Started
We recommend using an environment manager such as `Conda` to create a clean environment:

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12
conda activate fastvideo

# Install FastVideo
pip install fastvideo
```

Please see our [docs](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html) for more detailed installation instructions.

## Inference
### Generating Your First Video
Here's a minimal example to generate a video using the default settings. Create a file called `example.py` with the following code:

```python
from fastvideo import VideoGenerator

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
```

Run the script with:

```bash
python example.py
```

For a more detailed guide, please see our [inference quick start](https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html).

### Other docs:

- [Design Overview](https://hao-ai-lab.github.io/FastVideo/design/overview.html)
- [Contribution Guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html)

## Distillation and Finetuning
- [Distillation Guide](https://hao-ai-lab.github.io/FastVideo/training/distillation.html)
- [Finetuning Guide](https://hao-ai-lab.github.io/FastVideo/training/finetune.html)

## 📑 Development Plan

<!-- - More distillation methods -->
  <!-- - [ ] Add Distribution Matching Distillation -->
- More models support
  <!-- - [ ] Add CogvideoX model -->
  - [x] Add StepVideo to V1
- Optimization features
  - [x] Teacache in V1
  - [x] SageAttention in V1
- Code updates
  - [x] V1 Configuration API
  - [ ] Support Training in V1
  <!-- - [ ] fp8 support -->
  <!-- - [ ] faster load model and save model support -->

## 🤝 Contributing

We welcome all contributions. Please check out our guide [here](https://hao-ai-lab.github.io/FastVideo/contributing/overview.html)

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
@misc{zhang2025vsafastervideodiffusion,
      title={VSA: Faster Video Diffusion with Trainable Sparse Attention}, 
      author={Peiyuan Zhang and Haofeng Huang and Yongqi Chen and Will Lin and Zhengzhong Liu and Ion Stoica and Eric Xing and Hao Zhang},
      year={2025},
      eprint={2505.13389},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.13389}, 
}
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
