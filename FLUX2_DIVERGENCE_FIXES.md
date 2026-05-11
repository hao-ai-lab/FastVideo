# Flux2 Divergence Fixes

All fixes address discrepancies between FastVideo's Flux2 pipeline and the
official Diffusers `Flux2KleinPipeline` / `Flux2Pipeline`.

## 1. Timestep scaling (1000x too large)

**File:** `fastvideo/pipelines/stages/denoising.py`

**Problem:** The scheduler returns timesteps in `[0, 1000]` range. FastVideo
passed these directly to the transformer, which internally multiplies by 1000
(`timestep * 1000`), producing timestep embeddings in `[0, 1_000_000]`. Diffusers
divides by 1000 first (`timestep / 1000`), so the transformer sees `[0, 1]` and
scales back to `[0, 1000]`.

**Fix:** Added `t_expand = t_expand / 1000.0` for Flux2 (detected via
`dit_config.prefix == "Flux"`) before passing to the transformer.

## 2. Double guidance scaling (1000x again)

**File:** `fastvideo/pipelines/stages/denoising.py`

**Problem:** `guidance_expand` was created with `* 1000.0`, but the transformer
already multiplies guidance by 1000 internally. Net effect: guidance embedded at
`cfg_scale * 1_000_000` instead of `cfg_scale * 1000`.

**Fix:** Removed the external `* 1000.0` from `guidance_expand` creation.

## 3. Klein `embedded_cfg_scale` was `0.0`, should be `None`

**File:** `fastvideo/configs/pipelines/flux_2.py`

**Problem:** `Flux2KleinPipelineConfig` set `embedded_cfg_scale = 0.0`. This
produced a guidance tensor of `[0.0]` which, after the transformer's `* 1000`,
became `[0.0]` — a valid but wrong embedding. Diffusers Klein passes
`guidance=None` (no guidance embedding at all for distilled models).

**Fix:** Changed to `embedded_cfg_scale: float | None = None`.

## 4. Decode path: spurious `scaling_factor` division

**File:** `fastvideo/pipelines/stages/decoding.py`

**Problem:** After BN denormalization and unpatchify, the decode path called
`_denormalize_latents` which divides by `scaling_factor` (0.13025), effectively
multiplying latents by ~7.68x. Diffusers does **not** apply
`scaling_factor`/`shift_factor` for Flux2 — the BN denorm is the complete
inverse normalization.

**Fix:** Removed the `_denormalize_latents` call after
`_flux2_bn_denorm_and_unpatchify`.

## 5. Decode path: hardcoded channel counts

**File:** `fastvideo/pipelines/stages/decoding.py`

**Problem:** `is_flux2_packed` detection used hardcoded `128` and `32` channel
checks, which failed for Klein (64 packed channels, 16 VAE channels).

**Fix:** Added `_is_flux2_packed()` helper that dynamically checks channel count
against `vae.post_quant_conv.weight.shape[1] * 4`.

## 6. Text encoder: missing chat template

**Files:** `fastvideo/configs/models/encoders/qwen3.py`,
`fastvideo/pipelines/stages/text_encoding.py`

**Problem:** `Qwen3TextConfig` had `is_chat_model = False`. FastVideo tokenized
raw prompt strings directly, while Diffusers wraps prompts in Qwen3's chat
template (`{"role": "user", "content": prompt}`) with `enable_thinking=False`
and `add_generation_prompt=True`.

**Fix:** Set `is_chat_model = True` in `Qwen3TextConfig`. Updated the chat model
branch in `TextEncodingStage.encode_text` to use the two-step approach matching
Diffusers: format with `apply_chat_template(tokenize=False)` first, then
tokenize the resulting strings.

## 7. Decode path: docstrings and generic channel handling

**File:** `fastvideo/pipelines/stages/decoding.py`

**Problem:** Docstrings for `_unpatchify_latents` and
`_flux2_bn_denorm_and_unpatchify` hardcoded `128 -> 32` channel references.

**Fix:** Made docstrings generic (`C*4 -> C`, handles any channel count).

## 8. Block comparison tooling: `(0,0)` weight crash

**File:** `compare_flux2_dit_blocks.py`

**Problem:** `ff_context.linear_in.weight` showed shape `(0,0)` under
FSDP/DDP wrapping, causing `mat1 and mat2 shapes cannot be multiplied
(512x3072 and 0x0)` during module replay.

**Fix:** Rewrote `_get_ff_context_linear_tensors` to resolve weights via
`model.named_parameters()` and `model.state_dict()` with
`_materialize_weight_tensor()` for DTensor/shard handling. Removed unsafe
direct submodule weight access fallback.
