---
date: 2026-10-22
category: porting
severity: important
---

# A Model's First Parameter Does Not Identify Its Compute Dtype

## What Happened

Bernini-R component parity passed, but its denoising stage selected FP32
inputs for a primarily BF16 transformer. Reading
`next(transformer.parameters()).dtype` selected the FP32 scale/shift table,
not the dtype of patch, text, attention and feed-forward projections.

## Root Cause

Parameter registration order is unrelated to compute precision. Mixed-precision
checkpoints intentionally retain selected time, normalization and modulation
parameters in FP32. A stage that infers its working dtype from the first
parameter can therefore disagree with the model's projection weights even
when loading and isolated model forwards are correct.

## Fix

Read the stage's working dtype from the pipeline's configured component
precision. Cast target/source latents and positive/negative text conditioning
consistently at that boundary. Preserve the model's parameter-specific loading
policy and move it between devices without blanket-casting all parameters.

## Prevention

1. Check the released checkpoint's mixed-precision exceptions before loading.
   Casting an FP32 exception to BF16 and later back to FP32 cannot recover it.
2. Do not treat the first parameter's dtype as a component-wide precision
   contract. Use explicit configuration or a documented component interface.
3. Test the stage with an FP32 first parameter and BF16 projection weights,
   in addition to direct transformer parity.
4. Exercise the public generator with real weights through encoding,
   denoising and decoding. Component-only forwards do not validate dtype
   handoffs between stages.
