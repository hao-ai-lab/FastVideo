---
name: add-model-04-port-vae
description: Use during /add-model Phase 4 or Phase 6 to prototype or parity-debug one FastVideo-native VAE component.
---

# Add Model Port VAE

## Goal

Prototype or parity-debug one VAE or autoencoder in FastVideo-native code. This
skill covers video, image, and audio VAEs.

## Inputs

Follow `../add-model/shared/component_skill_common.md` and require the complete
packet from `../add-model/contracts/component_context.md`.

VAE-specific packet fields:

- `component`: VAE or autoencoder name.
- `parity_test`: `tests/local_tests/vaes/test_<family>_<component>_parity.py`.
- `weights`: converted VAE dir, HF subfolder, or local official path.
- `target_files`: `fastvideo/models/vaes/<arch_or_family>.py` and
  `fastvideo/configs/models/vaes/<arch_or_family>.py`.

## Modes

Use the common prototype and parity-debug modes from
`../add-model/shared/component_skill_common.md`.

VAE-specific prototype concerns include latent normalization, stochastic
posterior behavior, tiling incompatibility, temporal/spatial/audio layout, and
decode output containers.

## Reuse Proof

Apply the shared reuse proof. VAE-specific comparison must include latent layout,
temporal/spatial/audio compression, scaling factor, mean/std normalization,
posterior behavior, encode/decode output objects, tiling flags, and cropping.

## Existing FastVideo Patterns

- Shared tiling wrapper: `fastvideo/models/vaes/common.py::ParallelTiledVAE`.
- Config bases: `VAEConfig` and `VAEArchConfig` in
  `fastvideo/configs/models/vaes/base.py`.
- Use the matching VAE config bucket. Wrong bucket inheritance can typecheck but
  fail during pipeline wiring.
- Config export: add the config to
  `fastvideo/configs/models/vaes/__init__.py`.
- Registry discovery: set `EntryClass = <ClassName>` in the model file.
- Loader path: VAE loaders resolve `_class_name` through `ModelRegistry` and
  load converted component weights from the VAE subdir.
- Reference examples: `oobleck.py`, `autoencoder_kl.py`, `wanvae.py`,
  `ltx2vae.py`, and `gamecraftvae.py`.
- Layer guidance: `fastvideo/layers/AGENTS.md`.

## Implementation Rules

- Name reusable VAE architectures by architecture (`oobleck.py`,
  `autoencoder_kl.py`); name family-specific VAEs by family.
- Match official encode/decode contracts exactly: input layout, latent layout,
  temporal/spatial/audio compression, scaling factor, mean/std normalization,
  posterior sampling behavior, decode output object, and frame/sample cropping.
- Compare deterministic outputs in parity: decode outputs, encode mean/mode, or
  round-trip tensors. Do not compare stochastic samples unless the RNG path is
  explicitly controlled.
- Use FastVideo tiling only when it preserves official numerics for the tested
  shape; disable it in config for audio or unsupported dimensions.
- Put architecture constants on `VAEArchConfig`; put `load_encoder`,
  `load_decoder`, tiling, dtype, and pretrained path fields on `VAEConfig`.
- Follow the production import boundary in
  `../add-model/shared/common_rules.md`.

## Prototype Checks

Follow the shared prototype success criteria.

## Parity-Debug Loop

Run the shared parity-debug loop. The component test command is:

```bash
pytest <parity_test> -v -s
```

For numerical drift, check normalization, latent scaling, posterior mode vs
sample, channel order, and temporal/spatial/audio cropping before changing
layers.

## Escape Hatches

Follow `../add-model/shared/common_rules.md` and the component-specific guidance
in `../add-model/shared/component_skill_common.md`. VAE-specific ask cases include
dropping an encode/decode path, accepting an unsupported private op, or choosing
between incompatible official VAE definitions.

## Handoff

Return `../add-model/contracts/component_skill_handoff.md` following the common
handoff rules in `../add-model/shared/component_skill_common.md`.
