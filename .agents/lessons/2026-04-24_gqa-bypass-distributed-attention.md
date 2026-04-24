---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: attention
severity: critical
---

# GQA models must bypass DistributedAttention — use F.scaled_dot_product_attention

## What Happened
`DaVinciAttention` has `num_heads_q=40, num_heads_kv=8` (GQA, groups=5).
FastVideo's `DistributedAttention` (layer.py) concatenates q, k, v along
`dim=0` assuming equal head counts:
```python
qkv = torch.cat([q, k, v], dim=0)  # [3*batch, seq_len, num_heads, head_dim]
```
First attempt: expanding k/v with `repeat_interleave(5, dim=2)` before passing
to DistributedAttention. This fixed the cat shape error but caused a CUDA
device-side assert inside the flash-attn kernel because the kernel uses the
*registered* `num_kv_heads=8` for internal size checks while receiving tensors
with 40 heads. The error surfaces asynchronously at the next CUDA op in the
following forward pass (`dispatcher.permute(rope)`), making it look like a
completely unrelated OOB index error.

## Root Cause
`DistributedAttention` and its underlying flash-attn backend register head
counts at construction time. Passing expanded tensors that don't match the
registered `num_kv_heads` violates internal kernel assumptions.
`DistributedAttention` is also designed for multi-GPU sequence parallelism;
for SP=1 (single GPU), its all-to-all is a no-op and adds no value.

## Fix
Replace the `DistributedAttention` call with
`torch.nn.functional.scaled_dot_product_attention` using `enable_gqa=True`:

```python
# WRONG — DistributedAttention requires num_heads_q == num_heads_kv
attn_out, _ = self.attn(q_4d, k_4d, v_4d)

# ALSO WRONG — expanding k/v before DistributedAttention passes the cat
# but the flash-attn kernel uses its registered num_kv_heads=8 for internal
# size checks, causing a device-side assert when it receives 40-head tensors.
groups = self.num_heads_q // self.num_heads_kv
k_4d = k_4d.repeat_interleave(groups, dim=2)
v_4d = v_4d.repeat_interleave(groups, dim=2)
attn_out, _ = self.attn(q_4d, k_4d, v_4d)

# ALSO WRONG — repeat_interleave k/v to equal heads then call SDPA.
# The expanded tensors trigger internal device-side asserts inside
# PyTorch's flash attention kernel; the assert surfaces asynchronously
# at the next gather op in the following forward pass.
q_sd = q_4d.permute(0, 2, 1, 3).contiguous()
k_sd = k_4d.permute(0, 2, 1, 3).contiguous()
v_sd = v_4d.permute(0, 2, 1, 3).contiguous()
k_sd = k_sd.repeat_interleave(groups, dim=1).contiguous()
v_sd = v_sd.repeat_interleave(groups, dim=1).contiguous()
attn_out = F.scaled_dot_product_attention(q_sd, k_sd, v_sd)  # ← asserts internally

# CORRECT — bypass DistributedAttention, keep k/v at H_kv heads,
# and use enable_gqa=True so PyTorch's flash attention handles GQA natively.
# This matches how FastVideo's own SDPAImpl handles GQA (attention/backends/sdpa.py).
q_sd = q_4d.permute(0, 2, 1, 3).contiguous()  # [1, H_q, S, D]
k_sd = k_4d.permute(0, 2, 1, 3).contiguous()  # [1, H_kv, S, D]
v_sd = v_4d.permute(0, 2, 1, 3).contiguous()  # [1, H_kv, S, D]
sdpa_kwargs = {}
if num_heads_kv != num_heads_q:
    sdpa_kwargs["enable_gqa"] = True
attn_out = torch.nn.functional.scaled_dot_product_attention(
    q_sd, k_sd, v_sd, **sdpa_kwargs)           # [1, H_q, S, D]
attn_out = attn_out.permute(0, 2, 1, 3).squeeze(0)  # [S, H_q, D]
```

**Critical:** `.contiguous()` is required after `.permute(0,2,1,3)`.
Flash attention 2 requires contiguous inputs; non-contiguous tensors from
permute produce an `vectorized_gather_kernel: ind out of bounds` assert on
the flash-attn CUDA stream, surfacing asynchronously at the next main-stream
GPU op.

## Side effects
`DistributedAttention` handles multi-GPU SP (sequence parallelism) via
all-to-all. Bypassing it means SP>1 inference is not supported on this
attention path. For single-GPU inference ports this is acceptable; flag it
if multi-GPU SP is later needed.

## Prevention
During Phase 0 recon, check `num_heads_q` vs `num_heads_kv` in the arch
config. If `num_heads_q != num_heads_kv` (GQA), plan to use SDPA instead
of `DistributedAttention`. Also: when a CUDA device-side assert appears at
an *indexing* op (`x[perm]`) that has no plausible OOB reason (perm was
built from argsort), suspect a deferred async error from a kernel earlier in
the same or immediately preceding forward pass — add `CUDA_LAUNCH_BLOCKING=1`
to locate the true origin.
