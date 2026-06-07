# Cosmos3 Audio (PR2) — Port Plan

Branch: `feat/cosmos3-audio` (stacked on `feat/cosmos3-i2v`, which has T2V/I2V/T2I).
Goal: text-to-video+sound (**t2vs**) — generate synchronized audio alongside video.

## How the framework does audio (studied 2026-06-07)

- **Sound tokenizer = AVAE** (`cosmos_framework/model/vfm/tokenizers/audio/avae.py`
  + `avae_utils/`, ~2268 lines): a 48 kHz **stereo** neural audio codec.
  - checkpoint: `official_weights/cosmos3/sound_tokenizer/` (`model_type:
    autoencoder_v2`, ~1.9 GB). enc=`spec_convnext` (enc_dim 192, latent_dim 128,
    n_fft 64), dec=`oobleck` (dec_dim 320, strides [2,4,5,6,8]), VAE bottleneck,
    `snakebeta` activations, hop_size 1920.
  - interface: `encode(audio[1,C,N]) -> latent`, `decode(latent) -> audio`,
    `get_latent_num_samples(N)`, `sample_rate=48000`, `audio_channels=2`,
    `sound_latent_fps=25`.
- **DiT sound pathway** (`cosmos3_vfm_network.py`, 136 sound/audio refs): the MoT
  has `sound2llm` / `llm2sound` / `sound_modality_embed` + `pack_sound_latents`
  and joint vision+sound denoising (`preds_sound`, sound `condition_mask`, sound
  noise init `cond_mask*x0 + (1-cond_mask)*noise`, velocity `pred*(1-cond_mask)`).
  - FastVideo's native DiT ALREADY constructs the dormant heads
    (`audio_proj_in`/`audio_proj_out`/`audio_modality_embed`, gated on
    `arch.sound_gen`) for strict-load — the forward just doesn't use them yet.
- **Inference flow** (`cosmos_framework/inference/sound.py`): t2vs builds a
  zero **placeholder audio** sized to the video duration (sets sound latent
  length), `inject_sound_into_batch` upgrades the SequencePlan to has_sound,
  the omni model denoises vision+sound jointly, then AVAE-decodes the sound
  latent and `mux_audio_into_video` (PyAV, AAC) muxes it into the mp4
  (`save_sound` writes a WAV).

## Components (each: native port + framework parity test, per methodology)

1. **AVAE codec** — `fastvideo/models/.../cosmos3_avae.py` + config. Port
   encoder/decoder/bottleneck/snake. Parity: tiny AVAE, framework weights copied
   in, bit-exact `decode` (and `encode`) on CPU/fp32. **(largest piece)**
2. **DiT sound pathway** — activate the dormant heads in `forward`; port
   `pack_sound_latents` + sound token scatter/proj/modality-embed/velocity.
   Parity: extend the DiT harness with sound tokens.
3. **Sound sequence packing** — extend `sequence_packing.py` with the sound
   modality (positions, attn mode, condition mask). Parity vs framework
   `pack_input_sequence` with sound.
4. **Pipeline (t2vs)** — placeholder audio -> joint denoise -> split ->
   AVAE-decode sound -> mux into mp4 / save wav. Extend `Cosmos3DenoisingStage`
   + a sound-decode/mux stage.
5. **FastVideo AV infra** — audio in `OutputConfig` / a mux stage (check what
   exists; `cosmos_framework.inference.sound.mux_audio_into_video` is the ref).

## Open decisions
- **D1 (AVAE approach)** — full native port (methodology-consistent; ~2.3k lines)
  vs a documented lazy-wrapper around the framework AVAE (faster; but pulls heavy
  deps and bends the "native + no-framework-at-runtime" rule). Default per
  methodology: native port.
- **D2 (scope)** — t2vs (T+video+sound) first; defer audio-conditioned / v2vs.
- **D3** — confirm FastVideo can mux/emit audio (output format).

## Status
- [x] Branch forked, framework audio path studied, plan written.
- [ ] D1 decision -> start AVAE port + parity harness.
- [ ] DiT sound pathway, packing, pipeline, AV infra.
