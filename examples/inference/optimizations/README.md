# Optimization Examples

```bash
python examples/inference/optimizations/attention_example.py
```

```bash
python examples/inference/optimizations/teacache_example.py
<<<<<<< HEAD
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
=======
```
>>>>>>> 89eaa4ed (update)
