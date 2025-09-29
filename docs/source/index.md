# Welcome to FastVideo

:::{figure} ../../assets/logos/logo.svg
:align: center
:alt: FastVideo
:class: no-scaled-link
:width: 60%
:::

:::{raw} html
<p style="text-align:center">
<strong>FastVideo is a unified inference and post-training framework for accelerated video generation.
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

FastVideo is an inference and post-training framework for diffusion models. It features an end-to-end unified pipeline for accelerating diffusion models, starting from data preprocessing to model training, finetuning, distillation, and inference. FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo has you covered.

<div style="text-align: center;">
  <img src=_static/images/fastwan.png width="100%"/>
</div>

## Key Features

FastVideo has the following features:
- State-of-the-art performance optimizations for inference
  - [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507)
  - [TeaCache](https://arxiv.org/pdf/2411.19108)
  - [Sage Attention](https://arxiv.org/abs/2410.02367)
- E2E post-training support
  - Data preprocessing pipeline for video data.
  - [Sparse distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) for Wan2.1 and Wan2.2 using [Video Sparse Attention](https://arxiv.org/pdf/2505.13389) and [Distribution Matching Distillation](https://tianweiy.github.io/dmd2/)
  - Support full finetuning and LoRA finetuning for state-of-the-art open video DiTs.
  - Scalable training with FSDP2, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.

## Documentation

% How to start using FastVideo?

:::{toctree}
:caption: Getting Started
:maxdepth: 1

getting_started/installation
<!-- getting_started/v1_api -->
:::

:::{toctree}
:caption: Inference
:maxdepth: 1

inference/inference_quick_start
inference/examples/examples_inference_index
inference/configuration
inference/optimizations
inference/comfyui
inference/support_matrix
inference/cli
inference/add_pipeline
:::

:::{toctree}
:caption: Training
:maxdepth: 1

training/examples/examples_training_index
training/data_preprocess
<!-- training/finetune -->
:::

:::{toctree}
:caption: Distillation
:maxdepth: 1

distillation/examples/examples_distillation_index
distillation/data_preprocess
distillation/dmd
:::

% What is STA Kernel?

:::{toctree}
:caption: Sliding Tile Attention
:maxdepth: 1

sliding_tile_attention/installation
sliding_tile_attention/demo
:::

% What is VSA Kernel?

:::{toctree}
:caption: Video Sparse Attention
:maxdepth: 1

video_sparse_attention/installation
:::

:::{toctree}
:caption: Design
:maxdepth: 1
design/overview
:::

:::{toctree}
:caption: Developer Guide
:maxdepth: 2

contributing/overview
contributing/developer_env/index
contributing/profiling
:::

:::{toctree}
:caption: API Reference
:maxdepth: 2

<!-- api/summary -->
api/fastvideo/fastvideo
:::

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
