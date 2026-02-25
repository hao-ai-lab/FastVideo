# NVTX Profiling with Nsight Systems

## Running torchrun with NVTX profiling

Wrap the `torchrun` command with `nsys profile`:

```bash
nsys profile \
    --trace=cuda,nvtx \
    --output=/path/to/traces/my_trace \
    --force-overwrite=true \
    torchrun \
        --nnodes 1 \
        --nproc_per_node 4 \
        --master_port 29500 \
        your_script.py \
            --your-arg1 value1 \
            --your-arg2 value2
```

- `--trace=cuda,nvtx` captures both CUDA kernels and NVTX annotations.
- `--output` sets the path for the resulting `.nsys-rep` file, which can be opened in Nsight Systems for timeline analysis.
- `--force-overwrite=true` overwrites any existing trace at that path.

## Adding custom NVTX ranges

All utilities live in `fastvideo/profiler.py`.

### Context manager — `nvtx_range`

Use for wrapping arbitrary code blocks:

```python
from fastvideo.profiler import nvtx_range

with nvtx_range("range_name"):
    ......
```

As an example, to get per-block visibility in the timeline, wrap each block in the forward loop with an indexed `nvtx_range`:

```python
from fastvideo.profiler import nvtx_range

for i, block in enumerate(self.blocks):
    with nvtx_range(f"transformer.block[{i}].forward()"):
        hidden_states = block(hidden_states, ...)
```

This produces named ranges like `transformer.block[0].forward()`, `transformer.block[1].forward()`, etc. in the Nsight Systems timeline, making it easy to identify per-layer hotspots.
