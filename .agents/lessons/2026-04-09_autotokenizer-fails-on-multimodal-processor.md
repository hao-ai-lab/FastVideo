---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: important
---

# AutoTokenizer.from_pretrained raises ValueError/OSError for multimodal processors

## What Happened
`preprocessing_datasets.py` calls `AutoTokenizer.from_pretrained(tokenizer_path)`
to load the tokenizer. For Cosmos 2.5, the tokenizer directory contains a
`Qwen2_5_VLProcessor`, which `AutoTokenizer` cannot load — it raises a
`ValueError` ("Unrecognized model") or `OSError`.

This caused preprocessing to abort before any data was processed.

## Root Cause
`AutoTokenizer` only handles text tokenizers. Multimodal processors
(like `Qwen2_5_VLProcessor`) must be loaded with `AutoProcessor` instead.
`preprocessing_datasets.py` used `AutoTokenizer` unconditionally.

## Fix / Workaround
Wrap the `AutoTokenizer.from_pretrained` call in a try/except and set
`tokenizer = None` on failure. The preprocessing pipeline handles `None`
tokenizer gracefully (falls back to loading tokenizer from the pipeline):

```python
tokenizer = None
if os.path.exists(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, cache_dir=args.cache_dir)
    except (ValueError, OSError):
        # Multimodal processors (e.g. Qwen2_5_VLProcessor) cannot be loaded
        # via AutoTokenizer — the pipeline will load the processor directly.
        logger.warning("Could not load tokenizer at %s as AutoTokenizer "
                       "(may be a multimodal processor): %s", tokenizer_path, e)
```

## Prevention
- When porting any model with a VLM text encoder, check whether the tokenizer
  directory contains a processor config (`preprocessor_config.json` or
  `processor_config.json`). If so, `AutoTokenizer` will fail — use the
  try/except pattern above so preprocessing degrades gracefully.
- Quick check: `ls <model_path>/tokenizer/` — if you see `preprocessor_config.json`,
  it's a multimodal processor, not a plain tokenizer.
