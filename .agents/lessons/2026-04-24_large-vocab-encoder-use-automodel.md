---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: porting
severity: important
---

# Text encoders with large vocabularies need AutoModel, not T5EncoderModel

## What Happened
T5Gemma-9B uses the Gemma tokenizer (~256K vocab). Loading it with
`T5EncoderModel.from_pretrained(path)` creates an embedding table with the
standard T5 vocab size (32128). Any token with ID ≥ 32128 triggers a CUDA
device-side assert: `ind >= 0 && ind < ind_dim_size` in the embedding kernel.
This assertion fires on first forward pass, core dumps the worker, and the
error traceback points to `torch.arange` in T5 (asynchronous CUDA reporting
makes the line number misleading).

## Root Cause
`T5EncoderModel` hard-codes vocab_size=32128 as a default. The actual
vocab_size in the model's `config.json` is only used if the model subclass
reads it correctly. For T5Gemma (and any model using a Gemma/Llama/SentencePiece
tokenizer with a large vocab), the mismatch causes OOB token IDs.

## Fix
Load the full model with `AutoModel` then extract the encoder submodule:
```python
# WRONG — T5 default vocab (32K) vs Gemma tokenizer (256K) → OOB CUDA assert
text_encoder = T5EncoderModel.from_pretrained(path, torch_dtype=dtype)

# ALSO WRONG — AutoModel loads full seq2seq model; calling it without decoder
# input_ids raises "You must specify exactly one of input_ids or inputs_embeds"
text_encoder = AutoModel.from_pretrained(path, ...)  # runs encoder+decoder

# CORRECT — load full model, extract encoder, free the decoder
_full = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=dtype)
text_encoder = _full.encoder.cuda().eval()
del _full  # frees decoder weights
```

`AutoModel` reads `model_type` and `vocab_size` from `config.json`, so the
embedding table is sized correctly. Using `.encoder` directly bypasses the
decoder which has no generation inputs in a diffusion pipeline.

## Prevention
During Phase 0 recon, check the tokenizer type. If the HF repo uses a non-T5
tokenizer (Gemma, Llama, SentencePiece with large vocab), note it and use
`AutoModel` (or `AutoModelForSeq2SeqLM` + `.encoder`) rather than a hardcoded
model class. The CUDA assert error is misleading — if you see
`vectorized_gather_kernel: ind out of bounds` in the text encoding stage,
suspect vocab mismatch before anything else.
