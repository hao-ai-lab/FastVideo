---
date: 2026-05-07
experiment: PR #1280 (daVinci-MagiHuman port), DiT parity bring-up
category: porting
severity: important
---

# DiT Dtype Boundary Alignment with Flash-Attn-Style Backends

## What Happened

DiT bit-exact parity for daVinci-MagiHuman against the upstream reference
sat at `diff_max=0.5` after the architecture port was complete and weight
loading was correct. The error grew with depth (later layers diverged more
than earlier ones), suggesting an accumulating numerical drift rather than
a structural mismatch. None of the obvious culprits (RoPE, GQA expansion,
attention mask handling) accounted for the pattern.

## Root Cause

Four cumulative dtype-boundary mismatches, each individually small but
together pushing parity from `diff_max=0.5` to bit-exact (`diff_max=0.0`):

1. **SDPA inputs were not cast to bf16.** Upstream's `flash_attn_with_cp`
   internally casts Q/K/V to bf16 at `dit_module.py:508` before the kernel.
   FastVideo was passing fp32 tensors through, getting numerically different
   intermediates even though the kernel accepts both.

2. **Post-attention output was kept in bf16 across the per-head gating
   multiply.** Upstream upcasts to fp32 before the gating, FastVideo did the
   gate in bf16 then upcast.

3. **A residual-stream cast at the block boundary.** FastVideo had a
   `.to(bf16)` then `.to(fp32)` at the start of each block. Upstream keeps
   the residual stream **continuously in fp32** across all 40 layers; only
   the inputs to specific kernels are temporarily downcast.

4. **Parity test scheduler used a double-shift.** A separate per-block fix
   (Wave 11 production migration) — single-shift schedule is what upstream
   uses; the parity test was double-shifting.

## Fix / Workaround

Four cumulative changes in `fastvideo/models/dits/magi_human.py` (commit
`3a4816cb`), each with a comment at the call site explaining the upstream
parity rationale:

- Cast SDPA inputs to bf16 right before the attention call.
- Upcast attention output to fp32 before the per-head gate multiply.
- Drop the residual-stream `.to(bf16)`/`.to(fp32)` wrapper at the block
  boundary; let the residual stay fp32 throughout.
- Single-shift schedule in the parity test fixture (matches upstream Wave 11).

## Prevention

1. **For any DiT port with a flash-attn-style backend**, treat the dtype of
   the residual stream as a load-bearing invariant, not a performance knob.
   Document it in the model's per-pipeline AGENTS.md. MagiHuman's invariant:
   *residual stream stays fp32 across all blocks; only kernel inputs are
   temporarily bf16*.

2. **Use layer-by-layer activation hooks** when DiT parity is close-but-not-
   bit-exact and the gap grows with depth. The
   `fastvideo/hooks/activation_trace.py` infra exists exactly for this case
   (`add-model-trace` skill). In MagiHuman's case it would have localized the
   first divergence point in one pass.

3. **The `add-model-port-dit` skill** should explicitly call out:
   - SDPA input dtype must match the upstream kernel's internal cast.
   - Post-attention upcast happens **before** any per-head gate, not after.
   - Residual stream dtype across block boundaries is a parity invariant.
   These rules apply to any DiT port whose upstream uses a flash-attn-style
   backend (`flash_attn_with_cp`, `flex_flash_attn_func`, etc.).

4. **Add an "intermediate-layer parity" test** for new DiT ports — comparing
   activations at layer 5, 10, 20, 30 — not just the final output. A growing-
   with-depth pattern is otherwise indistinguishable from "almost right".
