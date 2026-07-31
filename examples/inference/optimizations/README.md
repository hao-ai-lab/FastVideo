# Optimization Examples

```bash
python examples/inference/optimizations/attention_example.py
```

## Workload-driven generation launcher

`generation_launcher.py` runs a single generation **mode** from a versioned
MotionKernel/FastVideo workload manifest. Baseline and optimized runs should
use separate processes so environment variables do not leak.

```bash
# Native baseline
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/wan_t2v_1.3b_480p.yaml \
  --mode native \
  --output-dir /tmp/wan_ab

# Optimized / fused candidate (mode_env from the workload is applied)
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/wan_t2v_1.3b_480p.yaml \
  --mode optimized \
  --output-dir /tmp/wan_ab

# Validate request construction without loading weights
python examples/inference/optimizations/generation_launcher.py \
  --workload /path/to/motionkernel/workloads/ltx_480p.yaml \
  --mode native \
  --output-dir /tmp/ltx_ab \
  --dry-run
```

From MotionKernel, the same flow is:

```bash
python workload.py run-ab \
  --fastvideo-checkout /path/to/FastVideo \
  --workload workloads/wan_t2v_1.3b_480p.yaml \
  --output workspace/wan_ab
```

Result files use `schema_version: 1` and record wall time, generation time,
peak memory, environment identity, optional frames path, and failure reasons.
