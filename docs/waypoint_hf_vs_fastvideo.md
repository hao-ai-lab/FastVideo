# Waypoint-1-Small: Hugging Face (Diffusers) vs FastVideo

Comparison of the official **Overworld/Waypoint-1-Small** implementation on Hugging Face (diffusers modular pipeline) with **FastVideo’s** waypoint pipeline and transformer.

Reference: [Overworld/Waypoint-1-Small](https://huggingface.co/Overworld/Waypoint-1-Small) (before_denoise.py, denoise.py, encoders.py, decoders.py, transformer, VAE).

---

## 1. Pipeline structure

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Layout** | Modular pipeline blocks: `WorldEngineSetTimestepsStep`, `WorldEngineSetupKVCacheStep`, `WorldEnginePrepareLatentsStep`, `WorldEngineDenoiseLoop`, `WorldEngineDecodeStep` | Single `WaypointPipeline` with `streaming_reset()`, `streaming_step()`, `streaming_clear()` |
| **State** | `PipelineState` + block state (e.g. `scheduler_sigmas`, `kv_cache`, `latents`, `frame_timestamp`) | `StreamingContext`: `batch`, `frame_index`, `kv_cache`, `prompt_emb`, `prompt_pad_mask` |
| **Entry** | Modular pipeline invocation with components (transformer, VAE, text_encoder, etc.) | Same components via `get_module()`; control inputs as tensors in `streaming_step()` |

---

## 2. KV cache (main behavioral difference)

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Cache type** | `StaticKVCache` → per-layer `LayerKVCache`: ring buffer with fixed capacity `L` (tokens) + tail frame. Uses `BlockMask` + `flex_attention`. | **No cache used**: `StreamingContext.kv_cache` is always `None`; pipeline never creates or passes a real cache. |
| **Creation** | `WorldEngineSetupKVCacheStep`: builds `StaticKVCache(transformer.config, batch_size=1, dtype)` and stores in state. | Not implemented; `ctx.kv_cache` stays `None`. |
| **Usage in attention** | `Attn`: `k, v, bm = kv_cache.upsert(k, v, pos_ids, layer_idx)` then `flex_attention(q, k, v, block_mask=bm)`. History frames are in the ring; current frame in tail. | `GatedSelfAttention`: accepts `kv_cache` but **does not use it**. Uses `F.scaled_dot_product_attention(..., is_causal=True)` on **current frame only** (no cross-frame history). |
| **Frozen vs unfrozen** | Denoise: `kv_cache.set_frozen(True)` (read-only). Cache pass: `kv_cache.set_frozen(False)` then forward with sigma=0 to **write** current frame into ring. | Cache pass runs a second forward with sigma=0 and `ctx.kv_cache` (None), so no state is persisted. |
| **Effect** | Autoregressive generation uses **causal attention over all previous frames** (within ring capacity). | Each frame is generated with **no access to previous frames**; only current-frame self-attention. |

So: **HF** implements full autoregressive history via a ring-buffer KV cache and flex attention; **FastVideo** currently does not—it only passes a placeholder `kv_cache` and never fills or reads it.

---

## 3. Denoising (rectified flow)

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Update** | `x = x + step_dsig * v` with `sigmas.diff()` (same as rectified flow). | Same: `x = x + (sigma_next - sigma_curr) * v_pred`. |
| **Sigmas** | From config / block state, e.g. `[1.0, 0.94921875, 0.83984375, 0.0]`. | From `pipeline_config.scheduler_sigmas`. |
| **Loop** | `WorldEngineDenoiseLoop`: for each sigma step call transformer, then cache pass (sigma=0) and `frame_timestamp.add_(1)`. | `streaming_step()`: for each time step, loop over sigmas, then optional “cache pass” (no-op when kv_cache is None). |

Denoising math and sigma schedule are aligned; only the cache update after denoising differs (HF writes cache, FastVideo does nothing useful).

---

## 4. First-frame image (start from image)

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Support** | Yes. `WorldEnginePrepareLatentsStep`: if `image` is provided, encode with VAE, run **cache pass** to put that frame in KV cache, then `frame_timestamp.add_(1)`. Next step uses random noise for the *next* frame. | **Not supported.** No path to pass an initial image; every frame starts from `torch.randn(...)` latents. |

To match HF behavior for “start from image”, FastVideo would need: (1) VAE encode of the image, (2) a real KV cache, (3) a cache pass that writes that latent into the cache before generating the next frame.

---

## 5. Encoders

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Text** | `WorldEngineTextEncoderStep`: UMT5, `prompt_clean` (ftfy + whitespace), max_length 512, `prompt_embeds` + `prompt_pad_mask`. | `streaming_reset()`: tokenizer + text_encoder (UMT5), same max_length idea; stores `prompt_emb` and `prompt_pad_mask` in context. |
| **Controller** | `WorldEngineControllerEncoderStep`: `button` (set of IDs) → one-hot `[1,1,n_buttons]`, `mouse` (x,y) → `[1,1,2]`, `scroll` → sign in `[1,1,1]`. | `streaming_step(keyboard_action, mouse_action)`: user passes tensors; pipeline builds `button`, `mouse`, `scroll` with same shapes (scroll zeros if not provided). `CtrlInput.to_tensors()` in pipeline is analogous to HF’s controller step. |

Semantics and tensor shapes are aligned; only the API differs (HF: high-level button/mouse/scroll, FastVideo: pre-built or helper-built tensors).

---

## 6. VAE

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Model** | `WorldEngineVAE` (DCAE): encode (e.g. for first-frame image), decode. | Same weights (e.g. from same HF repo); decode path only. |
| **Encode** | Used in prepare_latents when an input image is given. | Not used in waypoint pipeline. |
| **Decode** | `WorldEngineDecodeStep`: `vae.decode(latents.squeeze(1))`, then postprocess (pil/np/pt). | After denoising: `vae.decode(latent_in)` with optional `scaling_factor` / `shift_factor`, then permute to [B,C,H,W]. |

Decode path and scaling/shift handling are consistent; encode is only used on HF for first-frame image.

---

## 7. Transformer / attention

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **Block layout** | WorldDiTBlock: AdaLN → self-attn (RoPE, GQA, value residual, gated) → cross-attn (prompt) → MLPFusion (ctrl) → MLP. | WaypointBlock: same ordering (cond_head, attn, cross_attn, ctrl MLPFusion, MLP). |
| **Self-attn** | `Attn` / `MergedQKVAttn`: QK norm, OrthoRoPE, then `kv_cache.upsert()` + `flex_attention(q,k,v, block_mask=bm)`. | `GatedSelfAttention`: QK norm, OrthoRoPE, then `F.scaled_dot_product_attention(..., is_causal=True)`; **kv_cache ignored**. |
| **RoPE** | OrthoRoPE: time (lang freqs) + height/width (pixel freqs), cos/sin buffers. | `OrthoRoPE` in waypoint_transformer: same recipe (time/height/width, same freqs). |
| **Optimizations** | MergedQKVAttn, CachedDenoiseStepEmb, SplitMLPFusion. | Separate q/k/v projections; no merged QKV or cached sigma embedding. |

Architecture and conditioning are aligned; the only major difference is that HF uses flex_attention + KV cache for history, while FastVideo uses SDPA with no cache.

---

## 8. Decoder (latents → image)

| Aspect | Hugging Face (Diffusers) | FastVideo |
|--------|--------------------------|-----------|
| **When** | `WorldEngineDecodeStep` after denoise: decode, then clear latents for next frame. | Inside `streaming_step()` after each frame’s denoise (+ no-op cache pass): decode, append to list, then increment `frame_index`. |
| **Output** | State holds `images` (pil/pt/np). | Frames stacked into `ctx.batch.output` as [B, C, T, H, W]. |

Same idea; different state shape (HF single image per step, FastVideo batch of frames).

---

## 9. Summary table

| Feature | Hugging Face | FastVideo |
|---------|--------------|-----------|
| Rectified flow denoising | Yes | Yes |
| Sigma schedule | Config | Config |
| Text encoding (UMT5) | Yes | Yes |
| Controller (button/mouse/scroll) | Yes | Yes |
| OrthoRoPE (3-axis) | Yes | Yes |
| **KV cache (cross-frame history)** | **Yes (ring buffer + flex_attention)** | **No (kv_cache always None, unused)** |
| First-frame image conditioning | Yes | No |
| VAE encode | Used for first frame | Not used |
| VAE decode | Yes | Yes |
| Modular pipeline | Yes (blocks) | No (single pipeline class) |

---

## 10. What FastVideo would need for parity

1. **KV cache**: Implement or plug in a `StaticKVCache`-style structure (or equivalent) and create it in `streaming_reset()` (or first step). In `GatedSelfAttention`, use the cache: either (a) integrate a ring buffer + block mask and use `flex_attention`, or (b) maintain a simple per-layer K/V list and concatenate with current frame for SDPA (with a causal mask over time). Then the “cache pass” (forward with sigma=0) must actually write the current frame into that cache.

2. **First-frame image (optional)**: In the pipeline, accept an optional initial image; encode with VAE, run one cache pass to fill the KV cache with that frame, then continue with random noise for the next frame.

3. **Inference optimizations (optional)**: Merged QKV, cached denoise-step embeddings, and split MLP fusion can be added for speed and to match HF’s optimizations.

The critical functional gap for autoregressive “world model” behavior is **(1)**: without a real KV cache and using it in attention, each frame is generated without seeing previous frames.

---

## 11. Blur fixes (FastVideo pipeline)

If the Waypoint pipeline produces blurry images, the following were aligned with HF/Overworld:

1. **VAE scaling**: WorldEngine/DCAE uses `scale_factor=1.0`. The pipeline only applies `latent / scaling_factor` when `scaling_factor` is not `None` and differs from `1.0`, so an SD-style factor (e.g. 0.18215) is not applied and the decode is not over-scaled/blurred.

2. **Prompt padding**: Prompt embeddings are zeroed where `attention_mask == 0` (match HF encoders), and `prompt_pad_mask` is `attention_mask.eq(0)` so cross-attention does not attend to padding.

3. **Sigma schedule**: Default `scheduler_sigmas` is the HF schedule `[1.0, 0.94921875, 0.83984375, 0.0]` (4 values = 3 denoising steps). More steps than the reference can over-denoise and blur output.
