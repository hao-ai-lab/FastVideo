# Official vs FastVideo Waypoint: Differences

Comparison of the official Overworld Waypoint pipeline (produces good video) vs FastVideo's Waypoint pipeline (can produce blurrier output).

## 1. Sigma schedule (HIGH IMPACT)

| | Official | FastVideo |
|---|----------|-----------|
| **Source** | Overworld `transformer/config.json` | `waypoint.py` pipeline config |
| **Values** | `[1.0, 0.861, 0.729, 0.321, 0.0]` | `[1.0, 0.949, 0.840, 0.0]` |
| **Denoise steps** | 4 | 3 |
| **File** | HF Overworld/Waypoint-1-Small | `fastvideo/configs/pipelines/waypoint.py:67` |

Official uses 4 steps with a denser schedule; FastVideo uses 3 steps and a different sigma trajectory. The Waypoint transformer config (`waypoint_transformer.py:71`) defines sigmas closer to official, but the pipeline config overrides with different values.

---

## 2. Noise initialization (MEDIUM IMPACT)

| | Official | FastVideo |
|---|----------|-----------|
| **Call** | `torch.randn(..., generator=g)` | `torch.randn(...)` |
| **Seed** | `seed + frame_idx` per frame | Global RNG (no per-frame generator) |
| **File** | `scripts/official_waypoint.py:282` | `waypoint_pipeline.py:442` |

Official uses an explicit generator per frame for reproducibility; FastVideo relies on global state. If the streaming entrypoint sets `torch.manual_seed` once, all frames share the same base seed instead of `seed + frame_idx`.

---

## 3. Prompt encoding

| | Official | FastVideo |
|---|----------|-----------|
| **Encoder** | `T5EncoderModel` (encoder only) | UMT5 (from model repo) |
| **Mask** | `last_hidden_state * attention_mask.unsqueeze(-1).float()` | `last_hidden_state * attention_mask.unsqueeze(-1).to(prompt_emb.dtype)` |
| **Dtype** | `.to(torch.bfloat16)` after mask | Keeps encoder dtype; cast to bfloat16 when passing to transformer |
| **Stage** | Inline in script | `WaypointTextEncodingStage` (does *not* use `umt5_postprocess_text`) |
| **File** | `scripts/official_waypoint.py:236-239` | `waypoint_stages.py:63-66` |

Both mask padded positions and use `padding="max_length", max_length=512`. The Waypoint pipeline uses `WaypointTextEncodingStage` directly, so `umt5_postprocess_text` from the pipeline config is unused. Worth comparing `prompt_emb` stats (mean, std, min, max) between official and FastVideo debug output.

---

## 4. VAE decode

| | Official | FastVideo |
|---|----------|-----------|
| **Latent prep** | `x.squeeze(1).float()` | `x[:, 0].to(vae_dtype)` |
| **Scaling** | None | `latent / scaling_factor` if != 1.0 |
| **Shift** | None | `latent + shift` if configured |
| **File** | `scripts/official_waypoint.py:317-318` | `waypoint_pipeline.py:673-692` |

WorldEngine VAE likely has `scaling_factor=1.0`; if so, FastVideo matches official. Check logs for `scaling_factor` and `shift`.

---

## 5. Denoising formula

Both use the same update rule:

- Official: `x = x + step_dsig * v` where `step_dsig = sigmas.diff()`
- FastVideo: `x = x + (sigma_next - sigma_curr) * v_pred`

---

## 6. Dtype

| | Official | FastVideo |
|---|----------|-----------|
| **Transformer** | bfloat16 | bfloat16 |
| **Latent (x)** | bfloat16 | bfloat16 (from pipeline) |
| **VAE input** | float32 (`latent.float()`) | VAE dtype (fp32) |
| **sigma** | `(1, 1)` bfloat16 | `(B, 1)` pipeline dtype |

Official casts latent to float before decode; FastVideo uses VAE dtype. Both should be fp32 at decode.

---

## 7. KV cache

| | Official | FastVideo |
|---|----------|-----------|
| **Type** | Overworld `StaticKVCache` | Custom per-layer cache |
| **Frozen** | `set_frozen(True)` | Different lifecycle |

Different implementations can change cross-frame behavior.

---

## 8. Control inputs (mouse / button / scroll)

Both use `mouse [1,1,2]`, `button [1,1,256]`, `scroll [1,1,1]` with `button[0,0,17]=1`. Same layout.

---

## Summary of likely causes of blur

1. **Sigma schedule** – FastVideo uses 3 steps vs 4 and different sigma values.
2. **Noise initialization** – Per-frame generator/seed may differ.
3. **Prompt encoding** – Small differences in masking or dtype could matter.

---

## Recommended next steps

1. Align FastVideo’s `scheduler_sigmas` with official:  
   `[1.0, 0.8609585762023926, 0.729332447052002, 0.3205108940601349, 0.0]`
2. Use an explicit generator for `torch.randn` in the pipeline, with `seed + frame_index`.
3. Compare `prompt_emb` stats (mean, std, min, max) between official and FastVideo debug output.
4. Verify WorldEngine VAE `scaling_factor` and `shift` in FastVideo logs.
