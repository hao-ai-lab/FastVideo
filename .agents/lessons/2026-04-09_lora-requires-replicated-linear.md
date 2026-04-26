---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: critical
---

# LoRA injection requires ReplicatedLinear, not nn.Linear

## What Happened
After wiring up the Cosmos 2.5 DiT with plain `nn.Linear` for all attention
projections (`to_q`, `to_k`, `to_v`, `to_out`), LoRA fine-tuning appeared to
run without errors but the adapter had zero effect — trainable parameters were
zero and loss didn't move.

## Root Cause
FastVideo's `get_lora_layer()` in `fastvideo/layers/linear.py` only recognises
`ReplicatedLinear` instances. Plain `nn.Linear` layers are invisible to it, so
no LoRA adapter is injected. The model trains but no parameters are actually
updated.

## Fix / Workaround
Replace every `nn.Linear` used for attention projections with `ReplicatedLinear`
from `fastvideo.layers.linear`:

```python
from fastvideo.layers.linear import ReplicatedLinear
self.to_q = ReplicatedLinear(dim, dim, bias=False)
```

`ReplicatedLinear.forward()` returns a tuple `(output, _)` — unpack it:
```python
query, _ = self.to_q(hidden_states)
```

## Prevention
- For every port: use `ReplicatedLinear` (not `nn.Linear`) for all projection
  layers that should be LoRA targets (`to_q`, `to_k`, `to_v`, `to_out`).
- After implementing the model, run this smoke test before training:
  ```python
  from fastvideo.layers.linear import get_lora_layer
  assert get_lora_layer(model.blocks[0].attn.to_q) is not None, \
      "LoRA injection will silently fail — use ReplicatedLinear"
  ```
- Alignment tests pass either way — this only surfaces at training time.
