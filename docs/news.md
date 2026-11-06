# News

All FastVideo announcements, newest first. The README carries the latest items; this page keeps the full history.

## Recent news

- `2026/10/21`: Release [FastH3 Omni](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) (Ref2VA): reference-conditioned video-and-audio generation from a prompt plus ordered image, video and audio references, in eight fused forwards. It runs on consumer Blackwell (RTX 5090, RTX PRO 6000, DGX Spark — [NVFP4 QAD](https://huggingface.co/FastVideo/FastH3-OmniRef-NVFP4-QAD)), on RTX 4090 (FP8) and on Apple Silicon (MLX). See the [cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/) and the [inference guide](https://hao-ai-lab.github.io/FastVideo/inference/fasth3_omniref/).
- `2026/10/06`: FastVideo now supports [Kandinsky 6](https://x.com/kandinskylab_ai/status/2107374635218055345) from Kandinsky Lab: text- and image-to-video with synchronized audio (base and 10-step distilled pi-Flow checkpoints) plus video super-resolution up to 4x. See the [Kandinsky 6 recipes](https://haoailab.com/FastVideo/cookbook/kandinsky6/).
- `2026/10/06`: FastH3 V2 now runs on a single consumer machine: NVIDIA RTX 5090, RTX 4090 and RTX PRO 6000 GPUs, DGX Spark and Apple Silicon. We also release [FastH3 Trim](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4), an experimental pruned model that is 4.2× smaller than base H3 and runs in as little as 8 GB of GPU memory. Get the [models](https://huggingface.co/collections/FastVideo/fastvideo-fasth3) and read the [Blog](https://haoailab.com/blogs/fasth3-rtx/).
- `2026/09/15`: Release [FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2), an eight-forward data-free DMD2 checkpoint distilled from MiniMax-H3 with 80% Video Sparse Attention. Run it with `examples/inference/basic/basic_fasth3_8step.py` or the [FastH3 8-Step V2 recipe](https://haoailab.com/FastVideo/cookbook/minimax-h3/).
- `2026/09/01`: FastH3 now runs locally on Apple Silicon through MLX and on NVIDIA DGX Spark through CUDA 13, including two-Spark inference. Follow the [FastH3 recipes](https://haoailab.com/FastVideo/cookbook/minimax-h3/) and read the [Blog](https://haoailab.com/blogs/fasth3-local/).
- `2026/08/27`: [FastH3 Preview v1](https://haoailab.com/blogs/fasth3-preview/) is an open-weight 4-step sparse-distilled MiniMax-H3 model for synchronized video-and-audio generation, developed in collaboration with [Nuva Lab](https://nuvalab.ai/) and the [NVIDIA FastGen team](https://github.com/NVlabs/FastGen). Download the recommended [VSA / Data-Free weights](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), or see the [full FastH3 collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3).
- `2026/08/19`: FastVideo now supports MLX on Apple Silicon with [FastMetal-QAD](https://huggingface.co/collections/FastVideo/fastmetal), a family of 1.3B, 5B, and 14B models optimized for Mac. Follow the [MLX install guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/mlx/) and read the [Blog](https://haoailab.com/blogs/fastmetal/).

## Earlier news

- `2026/06/23`: Release FastWan-QAD: 5s of Video generated in 1.8s E2E. See the [FastWan-QAD models](https://huggingface.co/FastVideo/FastWan-QAD-FP8-1.3B), [Attn-QAT training guide](https://haoailab.com/FastVideo/training/attn_qat/), and [blog](https://haoailab.com/blogs/fastwan-qad/).
- `2026/03/17`: Release demo: Into the Dreamverse: Vibe Directing in FastVideo, check out the [Blog](https://haoailab.com/blogs/dreamverse/).
- `2026/03/13`: Release demo: Create a 5s 1080p Video in 4.5s with FastVideo on a Single GPU, check out the [Blog](https://haoailab.com/blogs/fastvideo_realtime_1080p/).
- `2025/11/19`: Release [CausalWan2.2 I2V A14B Preview](https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers) models, [Blog](https://hao-ai-lab.github.io/blogs/fastvideo_causalwan_preview/) and [Inference Code!](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py).
- `2025/08/04`: Release [FastWan](https://hao-ai-lab.github.io/FastVideo/distillation/dmd) models and [Sparse-Distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/).

## Earlier milestones

- `2025/06/14`: Release finetuning and inference code for [VSA](https://arxiv.org/pdf/2505.13389).
- `2025/04/24`: [FastVideo V1](https://hao-ai-lab.github.io/blogs/fastvideo/) is released!
- `2025/02/18`: Release the inference code for [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).
