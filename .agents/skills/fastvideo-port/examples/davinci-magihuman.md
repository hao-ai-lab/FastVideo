# Worked Example: daVinci-MagiHuman

**Repo:** https://github.com/GAIR-NLP/daVinci-MagiHuman  
**Architecture:** 15B unified single-stream Transformer (not a standard DiT).
Text, video, and audio processed jointly via self-attention only. No
cross-attention.

## Key architecture details

- 40 layers total; first 4 and last 4 use modality-specific projections;
  middle 32 layers share parameters across modalities (sandwich design).
- Per-head scalar gating (sigmoid) for stability.
- Two-stage inference: low-res generation → latent-space super-resolution.
- Text encoder: `t5gemma-9b` (Google).
- Audio model: `stable-audio-open-1.0` (Stability AI).
- VAE: `Wan2.2-TI2V-5B` latent space + custom lightweight turbo VAE decoder.

## Non-standard aspects requiring attention

- Unified sequence model — `DistributedAttention` applies to the full
  joint token sequence, not just video patches.
- Sandwich weight sharing: middle 32 layers share the same `nn.Module`
  instance. Use `ReplicatedLinear` for shared layers, `MoELinear` (stacked
  `[num_experts * out_features, in_features]`) for per-modality sandwich layers.
- Audio conditioning is a new modality type; may require a new pipeline
  stage (`AudioEncodingStage`).
- Two-stage pipeline requires two `DenoiseStage` calls or a custom stage.
- `stable-audio-open-1.0` weights are Apache-licensed; confirm redistribution
  rights before uploading a converted HF repo under `fastvideo/`.

## Recommended port order

1. VAE (reuse `WanVAE` with Wan2.2-TI2V-5B weights — likely same arch).
2. Text encoder (T5Gemma-9B — check if FastVideo has T5 support already).
3. DiT forward pass, T2V only (skip audio conditioning first).
4. Audio encoder (second pass once T2V parity passes).
5. Super-resolution stage.

## param_names_mapping

Official checkpoint key prefixes → FastVideo key prefixes:

| Official | FastVideo |
|----------|-----------|
| `block.layers.N.*` | `layers.N.*` |
| `adapter.video_embedder.*` | `adapter.video_proj.*` |
| `adapter.text_embedder.*` | `adapter.text_proj.*` |
| `adapter.audio_embedder.*` | `adapter.audio_proj.*` |
| `final_linear_video.*` | `final_proj_video.*` |
| `final_linear_audio.*` | `final_proj_audio.*` |
| `final_norm_video.*` | `final_norm_video.*` (pass-through) |
| `final_norm_audio.*` | `final_norm_audio.*` (pass-through) |

## Credentials required

- `HF_TOKEN` with access to `google/t5gemma-9b` (gated).
- `STABILITY_API_KEY` for `stable-audio-open-1.0`.