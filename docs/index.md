# Welcome to FastVideo

FastVideo is an inference and post-training framework for diffusion models. It features an end-to-end unified pipeline for accelerating diffusion models, starting from data preprocessing to model training, finetuning, distillation, and inference. FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo has you covered.

## Key Features

FastVideo has the following features:

- End-to-end post-training support for bidirectional and autoregressive models
  - Full finetuning and LoRA [finetuning](training/finetune.md) for state-of-the-art open video DiTs
  - [Data preprocessing pipeline](training/data_preprocess.md) for video, image, and text data
  - [Distribution Matching Distillation (DMD2)](distillation/dmd.md) stepwise distillation
  - Sparse attention with [Video Sparse Attention](attention/vsa/index.md)
  - [Sparse distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) to achieve >50x denoising speedup
  - [Attn-QAT training](training/attn_qat.md) for quantization-aware post-training
  - Causal distillation through Self-Forcing
  - Scalable training with FSDP2, sequence parallelism, and selective activation checkpointing
  - See the [training overview](training/overview.md) for the full training workflow
- State-of-the-art performance optimizations for inference
  - Sequence parallelism for distributed inference
  - Multiple state-of-the-art [attention backends](attention/index.md)
  - User-friendly [CLI](inference/cli.md) and Python API
  - See the [support matrix](inference/support_matrix.md) for supported models and [optimizations](inference/optimizations.md) for the full list
- Realtime video generation and editing
  - [Dreamverse](https://github.com/hao-ai-lab/FastVideo/tree/main/apps/dreamverse): stream and "vibe direct" video in realtime ([live demo](https://dreamverse.fastvideo.org/))

## Documentation

Welcome to FastVideo! This documentation will help you get started with our unified inference and post-training framework for accelerated video generation.

Use the navigation menu on the left to explore different sections:

- **Getting Started**: Installation and quick start guides
- **Inference**: Learn how to use FastVideo for video generation
- **Training**: Data preprocessing and fine-tuning workflows
- **Distillation**: Post-training optimization techniques
- **Sliding Tile Attention**: Legacy workflow docs and kernel notes
- **Video Sparse Attention**: Efficient attention for video models
- **Design**: Framework architecture and design principles
- **Developer Guide**: Contributing and development setup
- **API Reference**: Complete API documentation
