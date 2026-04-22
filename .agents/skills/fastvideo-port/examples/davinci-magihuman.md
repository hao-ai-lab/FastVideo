# Worked Example: daVinci-MagiHuman

**Repo:** https://github.com/GAIR-NLP/daVinci-MagiHuman  
**Architecture:** 15B unified single-stream Transformer. Text, video, and audio
processed jointly via self-attention only. No cross-attention. **Timestep-free**
— no AdaLN, no time embedding in the DiT.

## Key architecture details

- 40 layers; sandwich design: outer 8 (layers 0–3, 36–39) use per-modality
  MoE projections; inner 32 share one set of weights.
- Attention: GQA 40q/8kv, head_dim=128, hidden=5120. Per-head sigmoid output
  gate for stability.
- Activation: outer layers use GELU7 (intermediate=20480); inner layers use
  SwiGLU7 (intermediate=13652, doubled to 27304 with gate projection).
- **Timestep-free** — the transformer has no conditioning mechanism. Flow
  matching scheduler (shift=5.0) runs externally in a custom denoising stage.
- 9-dim RoPE: `(t,h,w, T,H,W, ref_T,ref_H,ref_W)` — each modality gets
  different coordinate semantics. Use v2-style audio ref: `(N_audio-1)//4+1`.
  Text tokens use negative offsets to avoid colliding with video coords.
- Dynamic CFG: `guidance_scale if t > 500 else 2.0`.
- **param_names_mapping = {} (empty)** — official checkpoint keys already
  match FastVideo naming convention exactly. Verify before writing rules.
- VAE: Wan 2.2, z_dim=48, temporal stride 4, spatial stride 8. No published
  per-channel normalization stats — stub zeros/ones and flag for calibration.
- Text encoder: T5-Gemma-9B, hidden_dim=3584, 640-token target.
- Audio model: stable-audio-open-1.0 (Apache-licensed).

## Non-standard aspects requiring attention

- **Timestep-free** — create `DaVinciDenoisingStage` owning the full
  scheduler loop. Do not use `DenoisingStage`. See lesson
  `2026-04-22_timestep-free-dit-needs-custom-denoising-stage.md`.
- **MoELinear stacked weights** — outer-layer expert projections use
  `weight: [num_experts * out_features, in_features]`. Mirror this exactly
  for `load_state_dict(strict=True)`. LoRA on outer layers is deferred (inner
  32 shared layers are the LoRA targets). See lesson
  `2026-04-20_moe-stacked-weights-must-match-official-layout.md`.
- **ModalityDispatcher** — tokens must be sorted by modality before dispatch
  to per-expert projections, then unsorted after. The permutation and its
  inverse must thread through the entire `TransformerBlock.forward()`.
- **9-dim RoPE coordinate assembly** — the official code has v1/v2/v3 path
  variants. Use v2. Read `inference/common/sequence_schema.py` carefully.
- **Audio fallback** — when no audio path is provided, fall back to
  `torch.randn` for audio latents (mirrors official T2V inference).
- **VAE normalization stats** — z_dim=48 has no published stats. Stub and
  flag. See lesson
  `2026-04-22_vae-normalization-stats-may-not-exist-for-new-z-dim.md`.
- **Two-stage SR pipeline** — base → SR-540p → SR-1080p not yet implemented.
  Implement T2V parity first, SR second.

## Recommended port order

1. VAE (reuse `WanVAE` with Wan2.2 weights — same arch, update z_dim=48).
2. Text encoder (T5Gemma-9B — check if FastVideo has T5 support already).
3. DiT forward pass T2V only — skip audio conditioning first.
4. Custom `DaVinciDenoisingStage` with scheduler loop + dynamic CFG.
5. Audio encoder (second pass once T2V parity passes).
6. Super-resolution stage.

## SwiGLU7 intermediate size

```python
# Inner layers (4–35): SwiGLU7 — value proj + gate proj
intermediate = int(hidden_size * 4 * 2/3) // 4 * 4  # = 13652
# Weight shape: [intermediate * 2, hidden_size] (value + gate concatenated)

# Outer layers (0–3, 36–39): GELU7 — no gate
intermediate = hidden_size * 4  # = 20480
```

The `// 4 * 4` rounding is required — off-by-one silently produces wrong shapes.

## Credentials required

- `HF_TOKEN` with access to `google/t5gemma-9b` (gated).
- `STABILITY_API_KEY` for `stable-audio-open-1.0`.