---
date: 2026-05-07
experiment: PR #1280 (daVinci-MagiHuman port), Wave 14
category: porting
severity: critical
---

# Silent Channel-Major Token-Packing Bugs

## What Happened

While porting daVinci-MagiHuman (`fastvideo/pipelines/basic/magi_human/`),
the pipeline-parity test passed bit-exactly but the E2E user-visible output
was **pure static noise**. Latent tensors compared identically against the
upstream reference at every checkpointed boundary, yet decoded videos showed
no recognizable content. The discrepancy reproduced on every variant
(base / distill / SR-540p / SR-1080p) with the same noise profile.

## Root Cause

Video tokens were being packed **spatial-major** instead of **channel-major**:

```python
# What we had (spatial-major, WRONG)
einops.rearrange(x, "b c (T pT) (H pH) (W pW) -> b (T H W) (pT pH pW C)", ...)

# What upstream's UnfoldNd produces (channel-major, CORRECT)
einops.rearrange(x, "b c (T pT) (H pH) (W pW) -> b (T H W) (C pT pH pW)", ...)
```

A single-character einops reorder. The pipeline-parity test used FastVideo's
own packer on **both** sides of the comparison, so the bug was invisible there
— both sides agreed on the wrong layout. The DiT consumed those tokens
without complaint because the channel dimension only matters at decode time,
when the VAE's first conv expects channel-major input. By that point the test
boundary was already passed.

The bug was load-bearing for any token-packed format that downstream feeds
into a `UnfoldNd`-shaped consumer. Wave 14 of the port took multiple bug-hunt
iterations and an Oracle consultation to localize.

## Fix / Workaround

Single-character einops change in `stages/latent_preparation.py:_img2tokens`
(commit `6d190693` of the original PR). After the fix, all four variants
produced expected E2E output and the pipeline-parity tests still passed
because both sides of the parity check are now correct.

## Prevention

1. **Never use the FastVideo-side packer on both sides of a parity test.**
   At least one parity boundary must compare against an upstream tensor
   produced by the upstream packer. For MagiHuman this means a separate
   `_img2tokens` parity test that feeds upstream `UnfoldNd` output as the
   reference, not FastVideo's reformatted equivalent.

2. **Add an E2E hash check** alongside latent-parity. The mp4 SHA was the
   first signal that something was wrong; if it had been part of the standard
   parity battery, the bug would have surfaced in Wave 1, not Wave 14. See
   `fastvideo/tests/ssim/test_magi_human_similarity.py` for the CI version.

3. **For any new model port that involves explicit tensor reshaping into
   tokens**, document the expected packing order (`(C pT pH pW)` vs
   `(pT pH pW C)`) at the call site and assert the layout matches the
   downstream consumer's expectation.

4. The `add-model-port-dit` skill's parity gate should require an E2E hash
   check for any DiT that does video token packing, not just latent
   bit-exactness.
