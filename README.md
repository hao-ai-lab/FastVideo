<div align="center">
<img src=assets/logos/logo.svg width="30%"/>
</div>

<p align="center">
     | <a href="https://hao-ai-lab.github.io/FastVideo"><b>Documentation</b></a> | <a href="https://haoailab.com/FastVideo/cookbook/"><b>Cookbook</b></a> | <a href="https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start/"><b> Quick Start</b></a> | <a href="https://github.com/hao-ai-lab/FastVideo/discussions/982"  target="_blank"><b>Weekly Dev Meeting</b></a>  | 🟣💬 <a href="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" target="_blank"> <b>Slack</b> </a> |  🟣💬 <a href="https://github.com/hao-ai-lab/FastVideo/discussions/1097" target="_blank"> <b> WeChat </b> </a> |
</p>

**FastVideo is a unified post-training and real-time inference framework for accelerated video generation.**

## Recent news
- `2026/10/21`: Release [FastH3 Omni](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) (Ref2VA): reference-conditioned video-and-audio generation from a prompt plus ordered image, video and audio references, in eight fused forwards. It runs on consumer Blackwell (RTX 5090, RTX PRO 6000, DGX Spark — [NVFP4 QAD](https://huggingface.co/FastVideo/FastH3-OmniRef-NVFP4-QAD)), on RTX 4090 (FP8) and on Apple Silicon (MLX). See the [cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/) and the [inference guide](https://hao-ai-lab.github.io/FastVideo/inference/fasth3_omniref/).
- `2026/10/06`: FastVideo now supports [Kandinsky 6](https://x.com/kandinskylab_ai/status/2107374635218055345) from Kandinsky Lab: text- and image-to-video with synchronized audio (base and 10-step distilled pi-Flow checkpoints) plus video super-resolution up to 4x. See the [Kandinsky 6 recipes](https://haoailab.com/FastVideo/cookbook/kandinsky6/).
- `2026/10/06`: FastH3 V2 now runs on a single consumer machine: NVIDIA RTX 5090, RTX 4090 and RTX PRO 6000 GPUs, DGX Spark and Apple Silicon. We also release [FastH3 Trim](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4), an experimental pruned model that is 4.2× smaller than base H3 and runs in as little as 8 GB of GPU memory. Get the [models](https://huggingface.co/collections/FastVideo/fastvideo-fasth3) and read the [Blog](https://haoailab.com/blogs/fasth3-rtx/).
- `2026/09/15`: Release [FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2), an eight-forward data-free DMD2 checkpoint distilled from MiniMax-H3 with 80% Video Sparse Attention. Run it with `examples/inference/basic/basic_fasth3_8step.py` or the [FastH3 8-Step V2 recipe](https://haoailab.com/FastVideo/cookbook/minimax-h3/).
- `2026/09/01`: FastH3 now runs locally on Apple Silicon through MLX and on NVIDIA DGX Spark through CUDA 13, including two-Spark inference. Follow the [FastH3 recipes](https://haoailab.com/FastVideo/cookbook/minimax-h3/) and read the [Blog](https://haoailab.com/blogs/fasth3-local/).
- `2026/08/27`: [FastH3 Preview v1](https://haoailab.com/blogs/fasth3-preview/) is an open-weight 4-step sparse-distilled MiniMax-H3 model for synchronized video-and-audio generation, developed in collaboration with [Nuva Lab](https://nuvalab.ai/) and the [NVIDIA FastGen team](https://github.com/NVlabs/FastGen). Download the recommended [VSA / Data-Free weights](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), or see the [full FastH3 collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3).
- `2026/08/19`: FastVideo now supports MLX on Apple Silicon with [FastMetal-QAD](https://huggingface.co/collections/FastVideo/fastmetal), a family of 1.3B, 5B, and 14B models optimized for Mac. Follow the [MLX install guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/mlx/) and read the [Blog](https://haoailab.com/blogs/fastmetal/).

### More News

- [All announcements and earlier milestones](https://hao-ai-lab.github.io/FastVideo/news/).

## Key Features

FastVideo has the following features:

- End-to-end post-training support for bidirectional and autoregressive models:
  - Support full finetuning and LoRA finetuning for state-of-the-art open video DiTs
  - Data preprocessing pipeline for video, image, and text data
  - Distribution Matching Distillation (DMD2) stepwise distillation.
  - Sparse attention with [Video Sparse Attention](https://arxiv.org/pdf/2505.13389)
  - [Sparse distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) to achieve >50x denoising speedup
  - Scalable training with FSDP2, sequence parallelism, and selective activation checkpointing.
  - Causal distillation through Self-Forcing
  - See this [page](https://hao-ai-lab.github.io/FastVideo/training/overview/) for the supported training workflows, and the [support matrix](https://hao-ai-lab.github.io/FastVideo/inference/support_matrix/) for supported models.
- State-of-the-art performance optimizations for inference
  - Sequence Parallelism for distributed inference
  - Multiple state-of-the-art attention backends
  - User-friendly CLI and Python API
  - See this [page](https://hao-ai-lab.github.io/FastVideo/inference/optimizations/) for full list of supported optimizations.
- Diverse hardware and OS support
  - Support H100, A100, 4090
  - Support Linux, Windows, MacOS
  - See this [page](https://hao-ai-lab.github.io/FastVideo/inference/support_matrix/) for full list of supported models, hardware assumptions, and optimization compatibility.
- Realtime video generation & editing
  - [Dreamverse](apps/dreamverse/README.md): stream and "vibe direct" video in realtime ([live demo](https://dreamverse.fastvideo.org/)), deployable on local GPU, a self-hosted B200 server, Docker, or serverless Modal

## Getting Started

Install FastVideo with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# NVIDIA CUDA 12; use cu130 on CUDA 13
UV_TORCH_BACKEND=cu126 uv pip install fastvideo
```

Platform guides: [NVIDIA GPUs](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/gpu/), [DGX Spark / GB10](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/spark/), [Apple Silicon MLX](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/mlx/), [AMD / ROCm](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/amd/) — or the [full installation guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/).

To have a coding agent do the install, give it the [agent install prompt](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/#install-with-an-ai-coding-agent).

## Sparse Distillation

Sparse distillation reaches >50× denoising speedup. See the [distillation guide](https://hao-ai-lab.github.io/FastVideo/distillation/dmd/) and the [blog](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/); the training recipes live under [`examples/distill/`](examples/distill/) with the [FastVideo Synthetic datasets](https://huggingface.co/FastVideo).

## Dreamverse — Realtime Video Generation & Editing

[Dreamverse](apps/dreamverse/README.md) is FastVideo's realtime video generation
and editing platform — "vibe directing" a video as it streams. It lives in the
monorepo under [`apps/dreamverse/`](apps/dreamverse/) and ships its own backend
(`dreamverse-server`) plus a web UI.

Try the [live demo](https://dreamverse.fastvideo.org/), read the
[blog](https://haoailab.com/blogs/dreamverse/), or run it yourself. Dreamverse
deploys on a local GPU, a self-hosted B200 server over SSH, Docker, or
serverless [Modal](apps/dreamverse/scripts/modal/README.md) — see the
[Dreamverse README](apps/dreamverse/README.md).

## Inference

### Generating Your First Video

Here's a minimal example to generate a video using the default settings. Make sure VSA kernels are [installed](https://hao-ai-lab.github.io/FastVideo/attention/vsa/#installation). Create a file called `example.py` with the following code:

```python
import os
from fastvideo import VideoGenerator

def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
```

`num_gpus=1` runs the worker in-process (weights load once, no extra Python process). On Colab/Kaggle-style machines with ~16GB host RAM, keep `num_gpus=1`; free-tier system memory does not grow with extra T4s, so `num_gpus>1` is likely to OOM.

Run the script with:

```bash
python example.py
```

For a more detailed guide, please see our [inference quick start](https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start/).

## More Guides

- [Design Overview](https://hao-ai-lab.github.io/FastVideo/design/overview/)
- [Distillation Guide](https://hao-ai-lab.github.io/FastVideo/distillation/dmd/)
- [Contribution Guide](https://hao-ai-lab.github.io/FastVideo/contributing/overview/)

## Awesome work using FastVideo or our research projects

- [SGLang](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen): SGLang's diffusion inference functionality is based on a fork of FastVideo on Sept. 24, 2025.
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO): A unified framework to adapt Group Relative Policy Optimization (GRPO) to visual generation paradigms. Code based on FastVideo.
- [SRPO](https://github.com/Tencent-Hunyuan/SRPO): A method to directly align the full diffusion trajectory with fine-grained human preference. Code based on FastVideo.
- [DCM](https://github.com/Vchitect/DCM): Dual-expert consistency model for efficient and high-quality video generation. Code based on FastVideo.
- [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay): An action-conditioned world model model trained using FastVideo framework.
- [Hunyuan Video 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5): A leading lightweight video generation model, where they proposed SSTA based on Sliding Tile Attention.
- [Kandinsky-5.0](https://github.com/kandinskylab/kandinsky-5): A family of diffusion models for video & image generation, where their NABLA attention includes a Sliding Tile Attention branch.
- [LongCat Video](https://github.com/meituan-longcat/LongCat-Video): A foundational video generation model with 13.6B parameters with block-sparse attention similar to Video Sparse Attention.

## 🤝 Contributing

We welcome all contributions. Please check out our guide [here](https://hao-ai-lab.github.io/FastVideo/contributing/overview/).
See details in [development roadmap](https://github.com/hao-ai-lab/FastVideo/issues/899).

## Acknowledgement

We learned the design and reused code from the following projects: [Wan-Video](https://github.com/Wan-Video), [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), [DMD2](https://github.com/tianweiy/DMD2), [diffusers](https://github.com/huggingface/diffusers), [xDiT](https://github.com/xdit-project/xDiT), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang). We thank [MBZUAI](https://ifm.mbzuai.ac.ae/), [Anyscale](https://www.anyscale.com/), and [GMI Cloud](https://www.gmicloud.ai/) for their support throughout this project.

## Citation

If you find FastVideo useful, please consider citing our research work:

```bibtex
@article{zhang2025vsa,
  title={Vsa: Faster video diffusion with trainable sparse attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Huang, Haofeng and Lin, Will and Liu, Zhengzhong and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.13389},
  year={2025}
}

@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}
```
