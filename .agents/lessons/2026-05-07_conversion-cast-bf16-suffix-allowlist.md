---
date: 2026-05-07
experiment: PR #1280 (daVinci-MagiHuman port), distill DiT parity bring-up
category: porting
severity: important
---

# Conversion `--cast-bf16` Needs an FP32-Keep Suffix Allowlist

## What Happened

`scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py --cast-bf16`
produced a converted distill DiT checkpoint that loaded cleanly, ran end-to-
end, and emitted reasonable output — but `test_magi_human_distill_parity`
showed `diff_mean=0.114` against the upstream reference. The base DiT was
bit-exact with the same conversion script. Only the distill variant
regressed.

The error was small enough that visual quality looked normal, but large
enough to fail bit-exact parity. The MagiHuman base + distill DiTs share
most of their architecture, so a difference that affected only distill was
counterintuitive.

## Root Cause

`--cast-bf16` was downcasting **all** fp32 tensors to bf16 indiscriminately.
The base checkpoint and the FastVideo `final_linear` / adapter modules
require eight specific tensors to remain in fp32:

- LayerNorm `gamma` / `beta` weights for the final residual exit
- Adapter projection biases
- A handful of scale parameters in the output projection chain

These tensors participate in chains where bf16 precision causes accumulation
error large enough to drift the parity check. The base DiT happened to not
hit those specific chains in the path the test exercised (different
attention mask shape, different audio interleave); the distill variant did.

## Fix / Workaround

Added `_FP32_KEEP_SUFFIXES` allowlist to
`convert_magi_human_to_diffusers.py` (commit `829f70d3`) and gated `--cast-
bf16` on it. Tensors whose state-dict key ends with any allowlisted suffix
keep their original fp32 dtype regardless of the flag.

Distill DiT parity went from `diff_mean=0.114` (silently wrong) to bit-exact
in one commit.

## Prevention

1. **Treat `--cast-bf16` as opinionated, not blanket.** Any conversion
   script that supports a global dtype downcast flag MUST own an explicit
   allowlist of fp32-keep tensors, documented at the top of the file.

2. **The `add-model-conversion` skill** should enforce two checks for any
   converter that ships a `--cast-bf16`-style flag:
   - Run the parity test for **every** variant of the model (base, distill,
     SR, etc.), not just the headline variant. Different variants exercise
     different code paths.
   - Diff the converted checkpoint's dtype map against the upstream
     reference and assert the allowlist covers every fp32 tensor in the
     reference.

3. **For MagiHuman specifically**: if you add or rename DiT modules that
   touch `final_linear`, the adapter, or any LayerNorm in the residual exit
   path, **check that any fp32-required tensors are covered by
   `_FP32_KEEP_SUFFIXES`** in the conversion script and re-run
   `test_magi_human_distill_parity` (it's the canary).

4. The lesson generalizes beyond MagiHuman: any DiT that uses bf16 mixed
   precision but keeps specific tensors in fp32 (a common pattern with
   flash-attn-style backends) needs this allowlist for any conversion that
   downcasts.
