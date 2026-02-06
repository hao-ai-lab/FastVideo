# Zero Initialization Fixes Summary

## Problem
New parameters (`action_embedder`, `to_out_prope`) were not learning - weights stayed at zero after training.

## Root Causes & Fixes

### 1. FSDP Loader Overwriting Model Initialization

**File:** `fastvideo/models/loader/fsdp_load.py`

**Problem:** FSDP loader initialized ALL new parameters (not in checkpoint) with zeros, overwriting the model's `__init__` initialization.

**Fix:** Added `KAIMING_INIT_PATTERNS` to selectively apply proper initialization:

```python
ALLOWED_NEW_PARAM_PATTERNS = ["gate_compress", "proj_l", "to_out_prope", "action_embedder"]
KAIMING_INIT_PATTERNS = ["fc_in.weight", "lora_A"]  # Input projections need non-zero init

for new_param_name in unused_keys:
    use_kaiming = any(pattern in new_param_name for pattern in KAIMING_INIT_PATTERNS)
    if use_kaiming:
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))  # Non-zero for gradient flow
    else:
        torch.zeros_like(...)  # Zero for output projections (residual behavior)
```

**Why:** 
- Input projections (`fc_in.weight`) need non-zero weights for gradients to flow
- Output projections (`fc_out.weight`) should be zero-initialized for stable residual learning (ControlNet/adapter pattern)

### 2. Attention Mask Shape Mismatch

**File:** `fastvideo/models/dits/wangame/hyworld_action_module.py`

**Problem:** Attention mask had shape `[B, L]` but query tensor had shape `[2*B, L, ...]` (rope + prope concatenated). The prope batch (second half) had no mask coverage → output was zeros.

**Fix:**
```python
# Before (wrong):
attention_mask = torch.ones(batch_size, seq_len, ...)  # [B, L]

# After (correct):
attention_mask = torch.ones(batch_size * 2, seq_len, ...)  # [2*B, L]
```

## Files Modified

1. `fastvideo/models/loader/fsdp_load.py` - KAIMING_INIT_PATTERNS
2. `fastvideo/models/dits/wangame/hyworld_action_module.py` - attention mask shape
