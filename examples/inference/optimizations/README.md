# Using TeaCache for Faster Video Generation in FastVideo

TeaCache is an optimization technique supported in FastVideo that can significantly speed up video generation by skipping redundant calculations across diffusion steps. This guide explains how to enable and configure TeaCache for optimal performance in FastVideo.

## What is TeaCache?

See the official [TeaCache](https://github.com/ali-vilab/TeaCache) repo and their paper for more details.


## How to Enable TeaCache

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

## Complete Example

Below a complete example of using TeaCache for faster video generation:

```bash
python examples/inference/optimizations/teacache_example.py
```

## Advanced Configuration

While TeaCache works well with default settings, you can fine-tune its behavior by adjusting the threshold value:

1. Lower threshold values (e.g., 0.1) will result in more skipped calculations and faster generation with slightly more potential for quality degradation
2. Higher threshold values (e.g., 0.15-0.23) will skip fewer calculations but maintain quality closer to the original 

Note that the optimal threshold depends on your specific model and content.

## Comparing With and Without TeaCache

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

For more details on optimization techniques in FastVideo, check out the other examples in the optimization directory.
