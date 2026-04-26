---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: shapes
severity: important
---

# Per-head QK norm requires [S*H, head_dim] reshape, not flatten(-2)

## What Happened
`DaVinciAttention.forward` applied `ModalityAwareRMSNorm` (initialized with
`dim=head_dim=128`) to Q and K via:
```python
q = self.q_norm(q.flatten(-2), ...)
```
`q` had shape `[S, H, D]` = `[S, 40, 128]`. `.flatten(-2)` gives `[S, 5120]`.
The norm weight is `[128 * num_modality]`, so the per-modality slice has
shape `[128]`. Broadcasting `[n_tokens, 5120] * [128]` fails:
```
RuntimeError: The size of tensor a (5120) must match the size of tensor b (128)
at non-singleton dimension 1
```

## Root Cause
`flatten(-2)` fuses heads and head_dim into one vector, but the norm was
designed to normalize each head_dim vector independently. The correct reshape
is to fuse the **sequence and head dimensions** (`[S*H, D]`), not the head and
head_dim dimensions.

## Fix
```python
# WRONG — fuses heads+head_dim → norm sees [S, H*D], not [S*H, D]
q = self.q_norm(q.flatten(-2), token_counts)

# CORRECT — fuses seq+heads → norm sees [S*H, D]
S = q.shape[0]
tc_q = [c * self.num_heads_q for c in token_counts] if token_counts else None
q = self.q_norm(
    q.reshape(S * self.num_heads_q, self.head_dim), tc_q,
).reshape(S, self.num_heads_q, self.head_dim)
```

Scaling `token_counts` by `H` preserves modality boundaries after the reshape:
after `[S, H, D] → [S*H, D]`, the first `n_video * H` rows are all heads of
all video tokens, which is exactly what the per-modality norm expects.

## Prevention
When porting a model that applies RMSNorm per head (with a norm whose
`normalized_shape` equals `head_dim`), always check the reshape direction:
- norm weight size = `head_dim` → reshape to `[S*H, head_dim]`
- norm weight size = `H*head_dim` → reshape to `[S, H*head_dim]`
Never use `.flatten(-2)` on `[S, H, D]` when the norm dim is `D` alone.
