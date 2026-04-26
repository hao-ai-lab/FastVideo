---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: critical
---

# dtype mismatch at projection layers when --dit_precision fp32 + bf16 validation

## What Happened
Training ran fine but validation crashed with a dtype mismatch error inside
`Cosmos25PatchEmbed` and `crossattn_proj`. The error was:
`RuntimeError: expected scalar type BFloat16 but found Float`

## Root Cause
`--dit_precision fp32` loads the transformer weights in `torch.float32`. During
validation, the inference pipeline feeds `bf16` inputs. Wan-style models use
`nn.Conv3d` / `nn.Linear` with no auto-cast, so the mismatch surfaces at the
first projection layer that touches both training weights and validation inputs.

Two specific sites in Cosmos 2.5:
1. `Cosmos25PatchEmbed.proj` (Conv3d) — input is bf16, weight is fp32.
2. `crossattn_proj` (nn.Sequential with nn.Linear) — encoder_hidden_states
   arrive as bf16 from the text encoder.

## Fix / Workaround
Cast the input to the weight's dtype at each projection site:

```python
# PatchEmbed
hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))

# crossattn_proj (nn.Sequential — index into [0] to get weight)
encoder_hidden_states = self.crossattn_proj(
    encoder_hidden_states.to(self.crossattn_proj[0].weight.dtype))
```

## Prevention
- For every new DiT port: add dtype casts at `PatchEmbed.proj` and any
  cross-attention input projection when `nn.Sequential` or `nn.Conv` is used.
- The pattern `x.to(layer.weight.dtype)` is the safe idiom — it handles fp32,
  bf16, and fp16 without hardcoding.
- This issue is invisible during alignment tests (both models run at the same
  dtype) but appears immediately on the first validation run.
