# `scripts/checkpoint_conversion/` — Official → FastVideo Converters

**Generated:** 2026-05-02

> **Pre-commit excludes `scripts/`.** Format / lint by hand against neighboring
> converters before opening a PR.

## What Lives Here

```
checkpoint_conversion/
├── convert_gamecraft_full.py            # Combined DiT + VAE
├── convert_gamecraft_vae.py             # VAE only
├── convert_gamecraft_weights.py         # DiT only
├── convert_gen3c_to_fastvideo.py
├── convert_ltx2_weights.py
├── convert_turbodiffusion_to_diffusers.py
├── convert_turbodiffusion_i2v_to_diffusers.py
├── extract_llava_text_encoder.py        # Encoder extraction from a multimodal repo
├── longcat_to_fastvideo.py
├── stable_audio_to_diffusers.py
├── wan_to_diffusers.py
├── validate_longcat_weights.py          # Post-conversion validation
├── pt_to_safetensors.py                 # Generic format flip
└── create_hf_repo.py                    # Push to HF after conversion
```

## Naming Convention

| Pattern | Use |
|---------|-----|
| `convert_<model>_*.py` / `<model>_to_<format>.py` | One-shot converter for a model family |
| `extract_<role>_*.py` | Pull a sub-component out of a multimodal repo |
| `validate_<model>_*.py` | Post-conversion shape / norm sanity checks |
| `<format>_to_<format>.py` | Generic format conversion (no model knowledge) |

## Authoring a New Converter

1. **Prototype the FastVideo-native component first** under
   `fastvideo/models/<role>/<model>.py`. Its `state_dict()` is the target
   surface — never the other way around.
2. Mirror the official-checkpoint key pattern → FastVideo-native key pattern in
   a `param_names_mapping` (declared on the arch config in
   `fastvideo/configs/models/<role>/<model>.py`).
3. The converter's only job: load the official checkpoint, apply the mapping
   (with split/fuse for QKV / packed MLP), and write a safetensors directory
   that the `ComponentLoader` (`fastvideo/models/loader/`) can read.
4. Record intentionally skipped keys (training-only EMA, optimizer state,
   logvar, dynamic buffers) in a constant near the top of the converter — with
   a one-line reason each.
5. Add a smoke test under `tests/local_tests/<model>/` or
   `fastvideo/tests/<role>/` that loads the converted weights and runs a
   parity assertion against the official reference (1 forward pass).

## Cross-Reference

- `fastvideo/layers/AGENTS.md` — which native layer to target (fused QKV,
  parallel MLP, etc.). Determines the split/fuse logic in the converter.
- `fastvideo/models/AGENTS.md` — model-side discipline (lint excluded, native
  state-dict is the source of truth).

## Anti-Patterns

- Hard-coding tensor renames inside the model class instead of the converter.
- Converting via a private fork of `transformers` / `diffusers` weight loaders.
  Read tensors directly with `safetensors` or `torch.load`.
- Pushing to HF before the validate / smoke test passes locally.
- Skipping the explicit "skipped keys" list — it documents intent and prevents
  silent precision loss across re-conversions.
