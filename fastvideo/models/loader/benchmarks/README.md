# Weight Loading Benchmarks

Benchmarks for measuring model weight loading speed from safetensors files.

## Scripts

### `benchmark_weight_loading.py`

Measures loading throughput for two modes:
- **`to_cpu=True`**: load weights to CPU (memory-mapped, no broadcast)
- **`to_cpu=False`**: load weights to GPU (rank 0 reads from disk, broadcasts to other ranks)

```bash
# Single GPU (no torchrun needed)
python fastvideo/models/loader/benchmarks/benchmark_weight_loading.py \
    --model-path /path/to/model --subfolder transformer

# Multi-GPU
torchrun --nproc_per_node=4 \
    fastvideo/models/loader/benchmarks/benchmark_weight_loading.py \
    --model-path /path/to/model --subfolder transformer
```

### `benchmark_weight_loading_comparison.py`

A/B comparison of independent read vs rank-0 broadcast, with both sync and async broadcast variants. Use this to measure the speedup from distributed weight loading.

```bash
torchrun --nproc_per_node=4 \
    fastvideo/models/loader/benchmarks/benchmark_weight_loading_comparison.py \
    --model-path /path/to/model --subfolder transformer
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | required | Local directory containing `.safetensors` files |
| `--subfolder` | none | Subfolder within model-path (e.g. `transformer`, `text_encoder`) |
| `--warmup` | 1 | Warmup iterations before timing |
| `--repeats` | 3 | Timed iterations |

## Example results

HunyuanVideo transformer (25.64 GB) on NVIDIA B200:

| GPUs | Independent read | Sync broadcast | Speedup |
|------|-----------------|----------------|---------|
| 1 | 4.12s (6.22 GB/s) | 4.12s (6.22 GB/s) | 1.00x |
| 2 | 4.32s (5.94 GB/s) | 4.32s (5.94 GB/s) | 1.00x |
| 4 | 5.62s (4.56 GB/s) | 4.54s (5.65 GB/s) | 1.24x |

The broadcast approach eliminates disk I/O contention as GPU count increases.
