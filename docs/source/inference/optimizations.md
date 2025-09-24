(inference-optimizations)=

# Optimizations

This page describes the various options for speeding up generation times in FastVideo.

## Table of Contents

- Optimized Attention Backends

  - [Flash Attention](#optimizations-flash)
  - [Sliding Tile Attention](#optimizations-sta)
  - [Sage Attention](#optimizations-sage)
  - [Sage Attention 3](#optimizations-sage3)

- Caching Techniques
  - [TeaCache](#optimizations-teacache)

(optimizations-backends)=

## Attention Backends

### Available Backends

- Torch SDPA: `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- Flash Attention 2 and 3: `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN`
- Sliding Tile Attention: `FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN`
- Video Sparse Attention: `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`
- Sage Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN`
- Sage Attention 3: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN_THREE`

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

(optimizations-flash)=

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

:::{note}
FastVideo will automatically detect and use `FA3` if it is installed when using `FLASH_ATTN` backend.
:::

(optimizations-sta)=

### Sliding Tile Attention

**`SLIDING_TILE_ATTN`**

```bash
pip install st_attn==0.0.4
```

Please see [this page](#sta-installation) for more installation instructions.

(optimizations-vsa)=

### Video Sparse Attention

**`VIDEO_SPARSE_ATTN`**

```bash
git submodule update --init --recursive
python setup_vsa.py install
```

Please see [this page](#vsa-installation) for more installation instructions.

(optimizations-sage)=

### Sage Attention

**`SAGE_ATTN`**

To use [SageAttention](https://github.com/thu-ml/SageAttention) 2.1.1, please compile from source:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd sageattention
python setup.py install  # or pip install -e .
```

(optimizations-sage3)=

### Sage Attention 3

**`SAGE_ATTN_THREE`**

[SageAttention 3](https://huggingface.co/jt-zhang/SageAttention3) is an advanced attention mechanism that leverages FP4 quantization and Blackwell GPU Tensor Cores for significant performance improvements.

#### Hardware Requirements

- RTX5090

#### Installation

Note that Sage Attention 3 requires `python>=3.13`, `torch>=2.8.0`, `CUDA >=12.8`. If you are using `uv` and using `torch==2.8.0` make sure that `sentencepiece==0.2.1` in the pyproject.toml file.

To use Sage Attention 3 in FastVideo, first get access to the SageAttention3 code, then move `sageattn/` and `setup.py` to the directory `fastvideo/attention/backends`, then install from using:

```bash
python setup.py install
```

(optimizations-teacache)=

## Teacache

TeaCache is an optimization technique supported in FastVideo that can significantly speed up video generation by skipping redundant calculations across diffusion steps. This guide explains how to enable and configure TeaCache for optimal performance in FastVideo.

### What is TeaCache?

See the official [TeaCache](https://github.com/ali-vilab/TeaCache) repo and their paper for more details.

### How to Enable TeaCache

Enabling TeaCache is straightforward - simply add the `enable_teacache=True` parameter to your `generate_video()` call:

```python
# ... previous code
generator.generate_video(
    prompt="Your prompt here",
    sampling_param=params,
    enable_teacache=True
)
# more code ...
```

### Complete Example

At the bottom is a complete example of using TeaCache for faster video generation. You can run it using the following command:

```bash
python examples/inference/optimizations/teacache_example.py
```

### Advanced Configuration

While TeaCache works well with default settings, you can fine-tune its behavior by adjusting the threshold value:

1. Lower threshold values (e.g., 0.1) will result in more skipped calculations and faster generation with slightly more potential for quality degradation
2. Higher threshold values (e.g., 0.15-0.23) will skip fewer calculations but maintain quality closer to the original

Note that the optimal threshold depends on your specific model and content.

## Benchmarking different optimizations

To benchmark the performance improvement, try generating the same video with and without TeaCache enabled and compare the generation times:

```python
# Without TeaCache
start_time = time.perf_counter()
generator.generate_video(prompt="Your prompt", enable_teacache=False)
standard_time = time.perf_counter() - start_time

# With TeaCache
start_time = time.perf_counter()
generator.generate_video(prompt="Your prompt", enable_teacache=True)
teacache_time = time.perf_counter() - start_time

print(f"Standard generation: {standard_time:.2f} seconds")
print(f"TeaCache generation: {teacache_time:.2f} seconds")
print(f"Speedup: {standard_time/teacache_time:.2f}x")
```

Note: If you want to benchmark different attention backends, you'll need to reinstantiate `VideoGenerator`.
