---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: dtype
severity: important
---

# Output accumulation buffers must use model dtype, not hardcoded float32

## What Happened
`DaVinciAdapter.forward` allocated the output buffer as:
```python
out = torch.zeros(S, hidden_size, device=x.device, dtype=torch.float32)
```
The model was loaded in `torch.bfloat16`, so all projection outputs (`v_out`,
`a_out`, `t_out`) were bf16. The assignment `out[video_mask] = v_out` triggered:
```
RuntimeError: Index put requires the source and destination dtypes match,
got Float for the destination and BFloat16 for the source.
```

## Root Cause
Hardcoding `dtype=torch.float32` for an output buffer is fragile whenever the
model might run in half precision. The buffer needs to hold the projection
outputs, so it must match the projection's output dtype.

## Fix
Use `x.dtype` instead of a literal dtype:
```python
# WRONG
out = torch.zeros(S, hidden_size, device=x.device, dtype=torch.float32)

# CORRECT
out = torch.zeros(S, hidden_size, device=x.device, dtype=x.dtype)
```

## Prevention
When porting any module that allocates a scratch/accumulation buffer before
filling it with results from `nn.Linear` or similar layers, always derive
`dtype` from the input tensor or a model parameter — never hardcode `float32`.
Search for `dtype=torch.float32` in newly ported modules and ask: "will this
buffer be assigned bf16/fp16 values at runtime?" If yes, change to `x.dtype`.
