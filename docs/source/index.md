# Welcome to FastVideo

:::{figure} ../../assets/logo.jpg
:align: center
:alt: FastVideo
:class: no-scaled-link
:width: 60%
:::

:::{raw} html
<p style="text-align:center">
<strong>FastVideo is a unified framework for accelerated video generation.
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

It features a clean, consistent API that works across popular video models, making it easier for developers to author new models and incorporate system- or kernel-level optimizations.
With FastVideo's optimizations, you can achieve more than 3x inference improvement compared to other systems.

<div style="text-align: center;">
  <img src=_static/images/perf.png width="100%"/>
</div>

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
inference/configuration
inference/optimizations
inference/support_matrix
inference/examples/examples_inference_index
inference/cli
inference/add_pipeline
inference/v0_inference
:::

:::{toctree}
:caption: Training
:maxdepth: 1

training/examples/examples_training_index
training/data_preprocess
training/distillation
<!-- training/finetune -->
:::

% What is STA Kernel?

:::{toctree}
:caption: Sliding Tile Attention
:maxdepth: 1

sliding_tile_attention/installation
sliding_tile_attention/demo
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
