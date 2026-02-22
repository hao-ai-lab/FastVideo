
# Optimizations

This page describes the various options for speeding up generation times in FastVideo.

## Table of Contents

- Optimized Attention Backends

  - [Flash Attention](#flash-attention)
  - [Sliding Tile Attention](#sliding-tile-attention)
  - [Sage Attention](#sage-attention)
  - [Sage Attention 3](#sage-attention-3)

## Attention Backends

### Available Backends

- Torch SDPA: `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- Flash Attention 2 and 3: `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN`
- Sliding Tile Attention: `FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN`
- Video Sparse Attention: `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`
- Sage Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN`
- Sage Attention 3: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN_THREE`
- Video MoBA Attention: `FASTVIDEO_ATTENTION_BACKEND=VMOBA_ATTN`
- Sparse Linear Attention: `FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN`
- SageSLA Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_SLA_ATTN`

### Configuring Backends

There are two ways to configure the attention backend in FastVideo.

#### 1. In Python

In python, set the `FASTVIDEO_ATTENTION_BACKEND` environment variable before instantiating `VideoGenerator` like this:

```python
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLIDING_TILE_ATTN"
```

#### 2. In CLI

You can also set the environment variable on the command line:

```bash
FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN python example.py
```

### Flash Attention

**`FLASH_ATTN`**

We recommend always installing [Flash Attention 2](https://github.com/Dao-AILab/flash-attention):

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

And if using a Hopper+ GPU (ie H100), installing [Flash Attention 3](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release) by compiling it from source (takes about 10 minutes for me):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention

cd hopper
pip install ninja
python setup.py install
```

### Sliding Tile Attention

**`SLIDING_TILE_ATTN`**

Sliding Tile Attention is provided by `fastvideo-kernel`.
See [STA docs](../attention/sta/index.md) for installation details.

### Video Sparse Attention

**`VIDEO_SPARSE_ATTN`**

Video Sparse Attention is provided by `fastvideo-kernel`.
See [VSA docs](../attention/vsa/index.md) for installation details.

### Sage Attention

**`SAGE_ATTN`**

To use [SageAttention](https://github.com/thu-ml/SageAttention) 2.1.1, please compile from source:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd sageattention
python setup.py install  # or pip install -e .
```

### Sage Attention 3

**`SAGE_ATTN_THREE`**

[SageAttention 3](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell) is an advanced attention mechanism that leverages FP4 quantization and Blackwell GPU Tensor Cores for significant performance improvements.

#### Hardware Requirements

- RTX5090

#### Installation

Note that Sage Attention 3 requires `python>=3.13`, `torch>=2.8.0`, `CUDA >=12.8`. If you are using `uv` and using `torch==2.8.0` make sure that `sentencepiece==0.2.1` in the pyproject.toml file.

To use Sage Attention 3 in FastVideo, follow the `README.md` in the linked repository to install the package from source.

### V-MoBA / SLA / SageSLA

These backends are model-specific and require the corresponding kernels and
dependencies. Use the support matrix and model examples to confirm compatibility
before enabling them.

## Benchmarking different optimizations

To benchmark backend performance, generate the same prompt with the same seed and compare end-to-end generation times:

```python
import os
import time

for backend in ["TORCH_SDPA", "FLASH_ATTN", "SAGE_ATTN"]:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
    generator = VideoGenerator.from_pretrained("your-model-id")
    start_time = time.perf_counter()
    generator.generate_video(
        prompt="Your prompt",
        seed=1024,
    )
    elapsed = time.perf_counter() - start_time
    print(f"{backend}: {elapsed:.2f}s")
```

Note: reinstantiate `VideoGenerator` after changing `FASTVIDEO_ATTENTION_BACKEND`.
