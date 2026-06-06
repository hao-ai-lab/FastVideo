# Kernel reviewer — references

## Layout

- `fastvideo-kernel/CMakeLists.txt`, `build.sh`, `pyproject.toml` — build entry.
- `fastvideo-kernel/csrc/` — CUDA / C++ source.
- `fastvideo-kernel/include/` — headers.
- `fastvideo-kernel/python/fastvideo_kernel/` — Python API surface.
- `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/` — Triton kernels.
- `fastvideo-kernel/tests/` — correctness tests.
- `fastvideo-kernel/benchmarks/` — perf benchmarks.
- `fastvideo-kernel/README.md` — build + install notes.

## Known-good kernel references

- **VSA (Video Sparse Attention)**: `fastvideo-kernel/tests/test_vsa.py`,
  `fastvideo-kernel/tests/test_vsa_forward.py` — reference for block-sparse
  attention test pattern.
- **STA (Sliding Tile Attention)**: `fastvideo-kernel/tests/test_sta.py`.
- **VMOBA**: `fastvideo-kernel/tests/test_vmoba_correctness.py`.
- **TurboDiffusion**: `fastvideo-kernel/tests/test_turbodiffusion.py`.
- **Fused RoPE Triton** (recent, PR #1245):
  `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/rope_triton.py` +
  `fastvideo-kernel/tests/test_rope_triton.py`.

## Attention backend layer

- `fastvideo/attention/base.py` — backend ABC + tensor-layout contract.
- `fastvideo/attention/__init__.py` — dispatch, env-var selection.
- `fastvideo/envs.py` — `FASTVIDEO_ATTENTION_BACKEND` values.
- `fastvideo/attention/backends/` — existing implementations (Flash, VSA,
  VMOBA, STA, SDPA).

Recent relevant fixes:
- PR #1183 — `SDPAImpl.forward()` arg mismatch when VSA unavailable.
- PR #1243 — `[perf] Skip bool-mask round-trip in block-sparse VSA attention`.
- PR #1221 — FP4 Flash Attention 4 for Blackwell (draft).

## Kernel↔layer wiring

When a kernel lands in `fastvideo-kernel/`, its call site usually lives in one of:
- `fastvideo/layers/rotary_embedding.py` (RoPE variants)
- `fastvideo/layers/layernorm.py` (layernorm variants, including FP32 cache)
- `fastvideo/attention/backends/<backend>.py` (attention kernels)
- `fastvideo/layers/activation.py` (activation fusions)

Grep for the kernel's Python entry point to find callers. If a kernel is
added without a call site, flag — it's dead code.

## Docs

- `docs/contributing/attention_backend.md` — how to add a new attention backend.
- `docs/contributing/profiling.md` — perf measurement guidance (useful for
  enforcing benchmark expectations).
- `fastvideo-kernel/README.md` — kernel build / install.
