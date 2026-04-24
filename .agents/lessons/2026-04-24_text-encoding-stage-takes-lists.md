---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: porting
severity: minor
---

# TextEncodingStage takes lists, not singular values

## What Happened
`TextEncodingStage.__init__` signature is `(text_encoders, tokenizers)` — plural
lists. Passing `text_encoder=...` (singular kwarg) silently fails or raises a
confusing error because `self.tokenizers` is never set.

## Root Cause
FastVideo supports multi-encoder pipelines (e.g. CLIP + T5). The stage is
designed to handle N encoders. Even single-encoder pipelines must wrap in a list.

## Fix
```python
# WRONG
TextEncodingStage(text_encoder=self.get_module("text_encoder"))

# CORRECT
TextEncodingStage(
    text_encoders=[self.get_module("text_encoder")],
    tokenizers=[self.get_module("tokenizer")],
)
```

## Prevention
When writing `create_pipeline_stages`, always use the plural list form for
`TextEncodingStage`. Check that `len(text_encoders) == len(tokenizers) ==
len(pipeline_config.text_encoder_configs)` — the stage asserts this at runtime.
