# 02 — TrackWan Model Architecture & Training Wrapper

TrackWan = Wan2.2 + MotionStream point-track motion control. Stage-1 is the
**bidirectional I2V finetune** with the original text prompt kept.

## Channel layout (concatenated at the patch embed)

```
16 (noisy latent)  +  20 (I2V cond: 4 mask + 16 first-frame latent)  +  16 (track map)  =  52 in
out = 16
```

- **I2V** via first-frame-latent **concat** (MatrixGame2-style: 4 mask + 16 first-frame
  latent = 20ch), **no CLIP** image cross-attention.
- **Text** prompt **kept** (Wan T5/UMT5 cross-attention stays active) — this is the
  key difference from MatrixGame2 (mouse/keyboard) and the main MotionStream-style choice.
- **Track map** (16ch) is produced *inside* the DiT by a `TrackEncoder`, so train and
  inference share one code path.

## How point tracks become a conditioning map

`fastvideo/models/dits/trackwan/track_encoder.py` — `TrackEncoder`:
- Input: `track_points [B,T,N,2]` (normalized [0,1]) + `track_visibility [B,T,N]`.
- Scatter visible points onto a latent-aligned grid (occluded → 0), front-pad time by
  `(t_comp-1)`, temporal-compress conv (kernel/stride `(4,1,1)`) to align to latent
  frames, optional spatial `F.interpolate`, then a **zero-init** 1×1×1 conv head.
- Zero-init head ⇒ at step 0 the track contribution is 0 ⇒ the model behaves exactly
  like the teacher (tracks are added gradually as training learns to use them).

## DiT subclass

`fastvideo/models/dits/trackwan/model.py` — `TrackWanTransformer3DModel(WanTransformer3DModel)`:
- `supports_track_input = True`.
- `forward(hidden_states, encoder_hidden_states, timestep, ..., track_points,
  track_visibility, track_ids)`: builds `cm = track_encoder(...)` (or zeros when tracks
  absent), `hidden_states = cat([hidden_states, cm], dim=1)`, then `super().forward`.
- When tracks are absent (no-track inference, motion-CFG uncond branch, or a fully
  masked frame) the track channels are zeros — i.e. "no tracks" == "p_mask masking".
- Returns a **bare tensor** `[B,C,T,H,W]` (not a tuple), regardless of `return_dict`.
- `track_config` is read from the raw `hf_config` (config.json) because the loader's
  plain `WanVideoArchConfig` drops unknown fields.

Config: `fastvideo/configs/models/dits/trackwan.py` — `in_channels=52`, `out_channels=16`,
`track_config={track_channels:16, id_dim:128, vae_spatial_compression:8,
vae_temporal_compression:4, max_track_id:100000, zero_init_head:True}`.
Registered in `fastvideo/models/registry.py` as `"TrackWanTransformer3DModel"`.

## Init model (52-ch from a 16-ch teacher)

`data_pipeline/convert_trackwan_init.py` converts a base Wan diffusers model →
52-ch track init:
- Zero-pads `patch_embedding.weight` `[1536,16,1,2,2] → [1536,52,1,2,2]` (pretrained
  weights occupy the first 16 input channels).
- Adds zero-init `track_encoder.*` keys.
- Sets `config.json` `in_channels=52` + `track_config`; symlinks the other components.
- Output: `/mnt/weka/home/hao.zhang/shao/data/models/trackwan_1.3b_init` (829 keys).

So step-0 behavior ≈ the teacher (padded channels + zero-init head contribute nothing).

## Training wrapper (new modular trainer)

`fastvideo/train/models/wantrack/wantrack.py` — `WanTrackModel(WanModel)`,
`_transformer_cls_name = "TrackWanTransformer3DModel"`. Mirrors the MatrixGame2 I2V
path but **keeps text** and uses point tracks.

- `init_preprocessors`: loads VAE, builds the parquet dataloader with
  `pyarrow_schema_i2v_track`, sets `_requires_negative_conditioning=False`, inits augments.
- `prepare_batch`: reads text/vae_latent/first_frame_latent/tracks; truncates tracks to
  `expected_frames=(num_latent_t-1)*ratio+1`; applies MotionStream augments; normalizes
  the main latents (`normalize_dit_input`); first_frame_latent is already normalized so
  used as-is.
- `_prepare_dit_inputs`: builds the 20-ch I2V cond (`_build_i2v_cond_concat`), concats to
  the 16-ch noisy latent → 36-ch `noisy_model_input`, stashes tracks in `conditional_dict`.
- `_build_distill_input_kwargs`: passes `hidden_states` (36ch), text, `timestep` (bf16),
  `track_points`, `track_visibility`. The DiT appends the 16-ch track map → 52ch.

## Flow-matching convention (for samplers / validation)

- Forward: `x_σ = (1-σ)·clean + σ·noise`, σ from `FlowMatchEulerDiscreteScheduler(shift=flow_shift)`.
- Target / model output: **velocity** `v = noise − clean` (`precondition_outputs=False`).
- Loss: `MSE(v_pred, noise − clean)` (`FineTuneMethod`).
- Sampling: start from noise, `scheduler.step(v_pred, t, latents)` (Euler), decode with
  `decode_latents` (denormalizes). GT reference must be `normalize_dit_input`'d before
  `decode_latents` (vae_latent is stored raw; first_frame_latent is stored normalized).

## MotionStream train-time augmentations (env-gated, no trainer changes)

In `WanTrackModel` (`_init_track_aug` / `_augment_tracks` in `prepare_batch`):

| env var | default | effect |
|---|---|---|
| `WANTRACK_AUG` | `1` | master toggle (`0` = clean overfit) |
| `WANTRACK_MIN_POINTS` / `WANTRACK_MAX_POINTS` | `1` / `200` | per-step point subsample (via visibility) |
| `WANTRACK_PMASK` | `0.2` | per-frame stochastic temporal masking |
| `WANTRACK_MOTION_DROP` | `0.1` | drop all motion (tracks→None ⇒ cm=zeros) |
| `WANTRACK_TEXT_DROP` | `0.0` | zero the text embedding (text CFG dropout) |

> ⚠️ The overfit runs so far used `WANTRACK_AUG=0` (clean signal). That model has only
> seen the **dense full grid**, so sparse drag control is out-of-distribution for it.
