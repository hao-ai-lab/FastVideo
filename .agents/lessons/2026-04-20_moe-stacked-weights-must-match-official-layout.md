---
date: 2026-04-20
experiment: daVinci-MagiHuman port (feat/davinci-port)
category: porting
severity: important
---

# MoE / per-modality stacked weights must match official layout for load_state_dict

## What Happened
daVinci uses per-modality linear layers in its 8 sandwich layers (mm_layers).
The official code stores these as a single stacked weight:
  `weight: shape [num_experts * out_features, in_features]`
and dispatches each modality group to its own slice at runtime.

Initial instinct was to split into 3 separate `ReplicatedLinear` instances
(one per modality), which is cleaner for LoRA. But this changes the weight key
structure — `load_state_dict(strict=True)` would fail because the checkpoint
has one key per layer, not three.

## Root Cause
`load_state_dict` matches by parameter name and shape exactly. Splitting a
stacked weight into separate modules changes both the key names and shapes,
making direct checkpoint loading impossible without a custom remapping function.

## Fix / Workaround
Mirror the official stacked layout in the FastVideo implementation:
```python
self.weight = nn.Parameter(torch.empty(num_experts * out_features, in_features))
```
Dispatch to expert slices at runtime via `.chunk(num_experts, dim=0)`.

Use `ReplicatedLinear` only for layers where the official code also has a
single weight (middle/shared layers). For MoE layers, use the stacked layout.

## Prevention
- Before implementing any linear layer, check whether the official weight has
  shape `[N * out, in]` (stacked MoE) or `[out, in]` (single). They look
  similar in a state_dict printout but require different implementations.
- LoRA on MoE layers is deferred — `get_lora_layer()` won't work on stacked
  weights. This is acceptable: the middle shared layers (32 of 40 in daVinci)
  are the LoRA targets anyway.
- Quick check before writing the layer:
  ```python
  for k, v in official_state_dict.items():
      if 'linear_qkv' in k:
          print(k, v.shape)
  # If shape[0] is a multiple of expected out_features → stacked MoE layout
  ```