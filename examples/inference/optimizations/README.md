# Speeding Up Generation

This page describes the various options for speeding up generation times.

## Table of Contents
- Optimized Attention Backends
    - [Flash Attention](#optimizations-flash)
    - [Sliding Tile Attention](#optimizations-sta)
    - [Sage Attention](#optimizations-sage)

- Caching Techniques
    - [TeaCache](#optimizations-teacache)


(optimizations-backends)=
## Attention Backends

`attention_example.py` shows how to set `FASTVIDEO_ATTENTION_BACKEND` env var to change attention backends. To run this example:
```bash
python examples/inference/optimizations/attention_example.py
```

In python, set the `FASTVIDEO_ATTENTION_BACKEND` before instantiating `VideoGenerator` like this:

```python
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLIDING_TILE_ATTN"
```

You can also set the env var when running any of the other example like this:
```bash
FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN python example.py
```

(optimizations-flash)=
### Flash Attention

(optimizations-sta)=
### Sliding Tile Attention

(optimizations-sage)=
### Sage Attention


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
start_time = time.time()
generator.generate_video(prompt="Your prompt", enable_teacache=False)
standard_time = time.time() - start_time

# With TeaCache
start_time = time.time()
generator.generate_video(prompt="Your prompt", enable_teacache=True)
teacache_time = time.time() - start_time

print(f"Standard generation: {standard_time:.2f} seconds")
print(f"TeaCache generation: {teacache_time:.2f} seconds")
print(f"Speedup: {standard_time/teacache_time:.2f}x")
```

Note: If you want to benchmark different attention backends, you'll need to reinstantiate `VideoGenerator`.
