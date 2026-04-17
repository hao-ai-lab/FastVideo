---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: important
---

# bf16 text embeddings crash .numpy() — must cast to float32 first

## What Happened
Preprocessing failed during the text embedding save step with:
`RuntimeError: Cannot convert a non-writable tensor to numpy. Use tensor.numpy() ... TypeError: Got unsupported ScalarType BFloat16`

## Root Cause
`preprocess_pipeline_base.py` calls `.cpu().numpy()` on text embeddings to
save them to parquet. NumPy does not support bfloat16, so this fails when the
text encoder runs in bf16 (which Reason1/Qwen2.5 does by default).

```python
# Fails for bf16 embeddings:
text_embedding = prompt_embeds[idx].cpu().numpy()
```

## Fix / Workaround
Insert `.float()` before `.numpy()`:
```python
text_embedding = prompt_embeds[idx].cpu().float().numpy()
```

## Prevention
- This affects any model whose text encoder runs in bf16 (most modern VLMs).
- The fix is in `fastvideo/pipelines/preprocess/preprocess_pipeline_base.py`
  and is universal — it should be safe for all models since float32 parquet
  storage is lossless for bf16 values.
- When adding a new model with a bf16 text encoder, verify this line exists.
