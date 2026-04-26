---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: pipeline
severity: critical
---

# CFG unconditional pass: build coords from null_text_tokens length, not prompt length

## What Happened
`DaVinciDenoisingStage` does classifier-free guidance (CFG) by running two
forward passes per step: one conditional (positive prompt) and one
unconditional (negative prompt or zeros). `text_coords` was built once for
the positive prompt token count (`n_t=26`) and then reused unchanged for the
unconditional pass.

The negative prompt encoded to 81 tokens. `_pack_tokens` assembled:
- `mod` tensor: `n_v + n_a + 81 = 465` entries
- `coords` tensor: `n_v + n_a + 26 = 410` entries (stale coords from cond pass)

`ModalityDispatcher` built `permute_mapping = argsort(mod)` with 465 entries,
then indexed into `rope = self.rope(coords)` which had only 410 rows.
`rope[perm]` raised:

```
IndexError: vectorized_gather_kernel: ind >= 0 && ind < ind_dim_size
```

at `dispatcher.permute(rope)`, making it appear to be an unrelated RoPE bug.

## Root Cause
Coords are assembled by concatenating per-modality coord tensors. If any
modality's coord tensor has the wrong length, the total coords tensor is
shorter than the mod tensor, causing OOB at the RoPE gather.

Negative prompts are almost always a different length from positive prompts —
longer negative prompts are common practice for video generation models.

## Fix
Build `null_text_coords` separately from `null_text_tokens`, not from the
positive prompt's `text_coords`:

```python
# WRONG — reuses cond coords; null_text_tokens may have different length
uncond_out = self._forward_packed(
    vid_tokens, vid_coords, audio_latents, audio_coords,
    null_text_tokens, text_coords,   # ← text_coords built for cond length
    target_dtype)

# CORRECT — build coords from null_text_tokens actual length
null_text_coords = _build_text_coords(null_text_tokens.shape[0], device)
uncond_out = self._forward_packed(
    vid_tokens, vid_coords, audio_latents, audio_coords,
    null_text_tokens, null_text_coords,   # ← always matches token count
    target_dtype)
```

## Prevention
Any time a packed-token model runs multiple forward passes (CFG, multi-scale,
etc.), every modality's coord tensor must be rebuilt from the actual token
count for that specific pass — never copy coord tensors across passes unless
the token count is guaranteed identical. When seeing a RoPE/gather OOB that
has no plausible reason (perm built from argsort, indices look valid), add a
print of `hidden.shape[0]`, `coords.shape[0]`, and `mod.shape[0]` before the
gather — a mismatch between coords and mod sizes is the immediate cause.
