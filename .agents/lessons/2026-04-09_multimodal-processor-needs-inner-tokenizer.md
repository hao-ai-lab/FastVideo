---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: important
---

# Multimodal processor (Qwen2_5_VLProcessor) must be unwrapped for text-only encoding

## What Happened
Preprocessing crashed when calling the tokenizer on plain text strings.
`Qwen2_5_VLProcessor` expects multimodal inputs (text + images/video) in its
standard `__call__`. Passing text-only strings raised a type error or produced
wrong output shapes.

## Root Cause
Cosmos 2.5's Reason1 text encoder uses `Qwen2_5_VLProcessor` as its tokenizer.
The processor wraps an inner `Qwen2Tokenizer` at `processor.tokenizer`. For
text-only preprocessing, the inner tokenizer must be used directly.

`Qwen2_5_VLConfig` also sets `is_chat_model=True`, so the `apply_chat_template`
path runs, which also lives on the inner tokenizer.

## Fix / Workaround
Unwrap before calling:
```python
tok = getattr(tokenizer, "tokenizer", tokenizer)
text_inputs = tok(processed_texts, **tok_kwargs).to(target_device)
```

`getattr(tokenizer, "tokenizer", tokenizer)` is safe — it falls back to the
original object for standard tokenizers that don't have an inner `.tokenizer`.

Also guard against empty strings (Qwen tokenizer raises on empty input):
```python
if not processed_text.strip():
    processed_text = "."
```

## Prevention
- When a model uses a VLM or multimodal processor as its text encoder, always
  check `text_encoding.py` to verify the tokenizer call uses the inner
  tokenizer for text-only paths.
- Look for `is_chat_model=True` in the encoder config as a signal that the
  encoder is a chat/VLM model requiring this treatment.
