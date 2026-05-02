
# Optimizations

This page describes the various options for speeding up generation times in FastVideo.

## Table of Contents

- Optimized Attention Backends

  - [Flash Attention](#flash-attention)
  - [Sliding Tile Attention (Archived)](#sliding-tile-attention-archived)
  - [Sage Attention](#sage-attention)
  - [Sage Attention 3](#sage-attention-3)

## Attention Backends

### Available Backends

- Torch SDPA: `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- Flash Attention 2 and 3: `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN`
- Video Sparse Attention: `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`
- Sage Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN`
- Sage Attention 3: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN_THREE`
- Video MoBA Attention: `FASTVIDEO_ATTENTION_BACKEND=VMOBA_ATTN`
- Sparse Linear Attention: `FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN`
- SageSLA Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_SLA_ATTN`
- Sliding Tile Attention (archived branch only):
  `FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN`

### Configuring Backends

There are two ways to configure the attention backend in FastVideo.

#### 1. In Python

In python, set the `FASTVIDEO_ATTENTION_BACKEND` environment variable before instantiating `VideoGenerator` like this:

```python
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
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
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

And if using a Hopper+ GPU (ie H100), installing [Flash Attention 3](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release) by compiling it from source (takes about 10 minutes for me):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention

cd hopper
uv pip install ninja
python setup.py install
```

### FP4 Flash Attention 4 (Blackwell only)

**`FLASH_ATTN`** with **`FASTVIDEO_NVFP4_FA4=1`**

On Blackwell GPUs (B200/B300), you can enable FP4 quantized Q/K attention for up to **1.39x kernel speedup** over BF16 FA4, peaking at **1801 TFLOPS**. This quantizes Q and K to NVFP4 E2M1 with per-block E4M3 scale factors while keeping V in BF16.

See the [Attn-QAT paper](https://arxiv.org/abs/2603.00040) and [flash-attention-fp4 benchmark results](https://github.com/hao-ai-lab/flash-attention-fp4/blob/fp4/flash_attn/cute/README.md) for details.

#### Requirements

- **GPU**: NVIDIA Blackwell (sm100a or sm103a) — B200, B300, GB200, GB300
- **CUDA**: 12.8+
- **Python**: 3.10 or 3.11

#### Installation

Install the FP4 flash attention kernel and its dependencies:

```bash
# 1. Install flash-attention-fp4 (CuTe DSL FP4 kernel)
pip install git+ssh://git@github.com/hao-ai-lab/flash-attention-fp4.git@fp4

# 2. Install NVIDIA CuTe DSL (required by the FP4 kernel)
pip install nvidia-cutlass-dsl

# 3. Install flashinfer (provides nvfp4_quantize for BF16→FP4 conversion)
pip install flashinfer
```

#### Usage

Enable FP4 attention via environment variables:

```bash
FASTVIDEO_NVFP4_FA4=1 CUTE_DSL_ENABLE_TVM_FFI=1 python examples/inference/optimizations/fp4_attn_wan2_1_1_3b.py --nvfp4_fa4
```

Or in Python:

```python
import os
os.environ["FASTVIDEO_NVFP4_FA4"] = "1"
os.environ["CUTE_DSL_ENABLE_TVM_FFI"] = "1"

from fastvideo import VideoGenerator
gen = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    use_fsdp_inference=False,  # FSDP is incompatible with FP4 pointer path
)
gen.generate_video(prompt="A raccoon in sunflowers", save_video=True)
```

#### Known Limitations

- `use_fsdp_inference=True` is incompatible with the FP4 path (FSDP shards invalidate tensor pointers)
- Per-call cosine similarity vs BF16: ~0.99 (slight quantization error accumulates over denoising steps)
- Only supports `headdim >= 128`

### Sliding Tile Attention (Archived)

**`SLIDING_TILE_ATTN`**

The full STA integration in `fastvideo/` is archived from `main` and preserved
at:

- https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete

We keep STA off `main` because we believe VSA is strictly better than STA for
the actively maintained FastVideo path.

Kernel code in `fastvideo-kernel` is still retained. For mask search and STA
inference workflow, see [STA docs](../attention/sta/index.md).

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
python setup.py install  # or uv pip install -e .
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
