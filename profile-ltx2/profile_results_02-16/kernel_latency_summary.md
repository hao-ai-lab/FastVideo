# Kernel Latency Summary

Profiler kernel `dur` is typically in microseconds; values below are converted to milliseconds.

| Attention Impl | Trace File | Kernel Name | Occurrence | Min (ms) | Max (ms) | Avg (ms) |
|---|---|---|---:|---:|---:|---:|
| vsa | `hpc-rack-1-10_19420.1771310583080545350.pt.trace.json.gz` | `_attn_fwd_sparse` | 96 | 2.753584 | 2.794811 | 2.762085 |
| flash_attn | `hpc-rack-1-10_16077.1771309181463056803.pt.trace.json.gz` | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, true, true, false, false>(flash::Flash_fwd_params)` | 96 | 9.526981 | 9.547398 | 9.534580 |
