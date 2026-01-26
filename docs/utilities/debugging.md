# Debugging Tools

FastVideo provides opt-in debugging utilities designed to help track numerical parity
without impacting performance when disabled. These tools are available via CLI flags
and are no-ops unless explicitly enabled.

## Stage-level sums

Log sums of key tensors after each pipeline stage (latents, outputs, prompt embeds).

Example:

```bash
python -m fastvideo.entrypoints.cli.generate \
  --model-path converted/ltx2_diffusers \
  --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers." \
  --height 64 --width 64 --num-frames 9 \
  --debug-stage-sums \
  --debug-stage-sums-path outputs/debug/ltx2_stage_sums.log
```

Log format:

```
fastvideo:stage=LTX2DenoisingStage latents=... prompt_embeds=... extra.ltx2_audio_latents=...
```

## Model-level sums (LTX-2)

Enable LTX-2 DiT internal sum logging (blocks, modality prep, etc.).

Example:

```bash
python -m fastvideo.entrypoints.cli.generate \
  --model-path converted/ltx2_diffusers \
  --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers." \
  --height 64 --width 64 --num-frames 9 \
  --debug-model-sums \
  --debug-model-sums-path outputs/debug/ltx2_model_sums.log
```

For even more detail (forward hooks inside the DiT blocks):

```bash
python -m fastvideo.entrypoints.cli.generate \
  --model-path converted/ltx2_diffusers \
  --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers." \
  --height 64 --width 64 --num-frames 9 \
  --debug-model-detail \
  --debug-model-detail-path outputs/debug/ltx2_model_detail.log
```

## Recursive module sums

Attach forward hooks to all parameterized submodules (recursively) and log output
sums. This is useful for debugging where a mismatch begins inside encoders, VAEs,
or DiT blocks.

Example:

```bash
python -m fastvideo.entrypoints.cli.generate \
  --model-path converted/ltx2_diffusers \
  --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers." \
  --height 64 --width 64 --num-frames 9 \
  --debug-module-sums \
  --debug-module-sums-path outputs/debug/ltx2_module_sums.log \
  --debug-module-sums-include transformer text_encoder vae
```

Use `--debug-module-sums-exclude` to skip noisy modules by name substring.

## Deterministic noise replay (LTX-2)

These flags let you save/load the exact noise used for reproducible runs and parity
testing.

Stage 1:
- `--ltx2-initial-latent-path` (video)
- `--ltx2-audio-latent-path` (audio)

Stage 2 (refinement):
- `--ltx2-refine-noise-path` (video noise)
- `--ltx2-refine-audio-noise-path` (audio noise)

Example:

```bash
python -m fastvideo.entrypoints.cli.generate \
  --model-path converted/ltx2_diffusers \
  --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers." \
  --height 64 --width 64 --num-frames 9 \
  --ltx2-refine-enabled \
  --ltx2-refine-upsampler-path converted/ltx2_spatial_upscaler \
  --ltx2-initial-latent-path outputs/debug/ltx2_stage1_latent.pt \
  --ltx2-refine-noise-path outputs/debug/ltx2_stage2_noise.pt
```

If the file exists, it is loaded. If it does not exist, FastVideo will save the
generated tensor to that path for future runs.

## Parity testing (tests)

Some local tests allow debug logging via environment variables, for example:

```bash
FASTVIDEO_DEBUG_STAGE_SUMS=1 \
FASTVIDEO_DEBUG_STAGE_SUMS_PATH=outputs/debug/two_stage_stage_sums.log \
FASTVIDEO_DEBUG_REF_SUMS=1 \
FASTVIDEO_DEBUG_REF_SUMS_PATH=outputs/debug/two_stage_reference_sums.log \
pytest tests/local_tests/pipelines/test_ltx2_two_stage_parity.py -q -rs
```

This writes both FastVideo and reference sums so you can compare where divergence
begins.
