# Profiling FastVideo

!!! warning
    Profiling is only intended for FastVideo developers and maintainers to understand the proportion of time spent in different parts of the codebase. **FastVideo end-users should never turn on profiling** as it will significantly slow down the inference.

## Profile with PyTorch Profiler

We support tracing FastVideo workers using the `torch.profiler` module. You can enable tracing by setting the `FASTVIDEO_TORCH_PROFILER_DIR` environment variable to the directory where you want to save the traces: `FASTVIDEO_TORCH_PROFILER_DIR=/mnt/traces/`. Additionally, you can control the profiling content by specifying the following environment variables:

- `FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES=1` to enable recording Tensor Shapes, off by default
- `FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY=1` to record memory, off by default
- `FASTVIDEO_TORCH_PROFILER_WITH_STACK=1` to enable recording stack information, on by default
- `FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=1` to enable recording FLOPs, off by default

Traces can be visualized using <https://ui.perfetto.dev/>.

!!! tip
    Only send a few requests through FastVideo when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.

!!! tip
    To stop the profiler - it flushes out all the profile trace files to the directory. This takes time, for example for about 5 step full finetuning job on Wan2.1 T2V 1.3B model, it takes about 30 minutes to flush out on a H200.
