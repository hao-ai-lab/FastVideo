# Welcome to FastVideo

:::{figure} ../../assets/logo.jpg
:align: center
:alt: FastVideo
:class: no-scaled-link
:width: 60%
:::

:::{raw} html
<p style="text-align:center">
<strong>FastVideo is a lightweight framework for accelerating large video diffusion models.
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/hao-ai-lab/FastVideo/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

FastVideo is a lightweight framework for accelerating large video diffusion models developed by the [Hao AI Lab](https://hao-ai-lab.github.io/).

<div style="text-align: center;">
  <video controls width="800">
    <source src="https://github.com/user-attachments/assets/79af5fb8-707c-4263-b153-9ab2a01d3ac1" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

FastVideo currently offers: (with more to come)

- [NEW!] V1 inference API available. Full announcement coming soon!
- [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).
- FastHunyuan and FastMochi: consistency distilled video diffusion models for 8x inference speedup.
- First open distillation recipes for video DiT, based on [PCM](https://github.com/G-U-N/Phased-Consistency-Model).
- Support distilling/finetuning/inferencing state-of-the-art open video DiTs: 1. Mochi 2. Hunyuan.
- Scalable training with FSDP, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.
- Memory efficient finetuning with LoRA, precomputed latent, and precomputed text embeddings.

Dev in progress and highly experimental.

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
inference/add_pipeline
inference/v0_inference
:::

:::{toctree}
:caption: Training
:maxdepth: 1

training/data_preprocess
training/distillation
training/finetune
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
