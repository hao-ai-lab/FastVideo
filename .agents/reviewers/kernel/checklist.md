# Kernel reviewer checklist

## Correctness

- [ ] New/modified kernel has a correctness test in `fastvideo-kernel/tests/`.
- [ ] Test compares against a reference (PyTorch eager or existing kernel).
- [ ] Tolerances are appropriate for the dtype (tight enough to catch bugs,
      loose enough to absorb expected FP drift).
- [ ] Test matrix covers multiple head dims, seq lengths, dtypes, and at
      least one batch > 1.
- [ ] No out-of-bounds access (block-boundary guards in place).
- [ ] `__syncthreads()` / `__syncwarp()` used where shared memory / warp
      primitives require it.

## Numerical

- [ ] Reductions (softmax, layernorm, sum) use FP32 accumulators.
- [ ] No silent dtype downgrade in intermediate tensors.
- [ ] `eps` placement preserved for layernorm / rmsnorm variants.

## Build

- [ ] New `.cu` / `.cpp` files registered in `CMakeLists.txt` or the Python
      build config.
- [ ] `__CUDA_ARCH__` guards present for arch-specific instructions
      (bf16, TMA, cp.async, wgmma).
- [ ] `./build.sh` is expected to succeed (author should confirm in PR body).
- [ ] No reliance on a Triton version not pinned in `fastvideo-kernel/pyproject.toml`.

## Perf claims

- [ ] Title / body claim is quoted verbatim in the review.
- [ ] Benchmark script exists under `fastvideo-kernel/benchmarks/`.
- [ ] Benchmark includes warmup and correct sync.
- [ ] Measured GPU is identified (A100 / L40S / H100 / B100).
- [ ] Baseline is identified (PyTorch eager? previous kernel? specify.)

## Attention backend (if `fastvideo/attention/` is touched)

- [ ] New backend integrates with `FASTVIDEO_ATTENTION_BACKEND` env var.
- [ ] Returns the same tensor layout as existing backends.
- [ ] `SDPAImpl.forward()` signature unchanged, or all call sites updated.
- [ ] Backend has a fallback path when it's unavailable (no `ImportError`
      at import time for users without the dep).

## Training compatibility

- [ ] If the kernel is invoked from `fastvideo/train/` or `fastvideo/training/`,
      it supports backward (or the call site is guarded with `torch.no_grad()`).

## PR template compliance

- [ ] Title starts with `[perf]` / `[feat]` / `[kernel]` / `[bugfix]`.
- [ ] Test Plan in body includes the kernel test command and output.
- [ ] Test Results include numerical parity (for correctness) and throughput
      (for perf).
