# RMSNorm Comparison: FastVideo vs SGLang

## Finding

The QK-norm in layer 0 diverges because FastVideo and SGLang apply **different ordering of cast vs weight multiply** in RMSNorm.

### FastVideo (`fastvideo/layers/layernorm.py` lines 77-81)

```python
x = x * torch.rsqrt(variance + self.variance_epsilon)
x = x.to(orig_dtype)           # Cast to bf16 FIRST
if self.has_weight:
    x = x * self.weight        # Then multiply weight (in bf16)
```

### SGLang (`sglang/multimodal_gen/runtime/layers/layernorm.py` lines 134-135)

```python
x = x * torch.rsqrt(variance + self.variance_epsilon)
x = (x * self.weight).to(orig_dtype)   # Multiply in float32, THEN cast
```

## Impact

When `orig_dtype` is bf16:

- **SGLang**: Keeps full precision until the end: `(x_norm * weight)` in float32, then a single cast.
- **FastVideo**: Casts to bf16 before the weight multiply, so `x_norm_bf16 * weight` is done in bf16.

The order of operations changes rounding and leads to different outputs (e.g., k_norm max_diff ≈ 2.0).

## Recommended Fix

In FastVideo’s RMSNorm `forward_native`, change:

```python
x = x * torch.rsqrt(variance + self.variance_epsilon)
x = x.to(orig_dtype)
if self.has_weight:
    x = x * self.weight
```

to:

```python
x = x * torch.rsqrt(variance + self.variance_epsilon)
if self.has_weight:
    x = (x * self.weight).to(orig_dtype)
else:
    x = x.to(orig_dtype)
```

This aligns FastVideo with SGLang and should remove the k_norm divergence.
