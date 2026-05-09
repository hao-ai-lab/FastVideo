---
name: add-model-06-port-generic
description: Use during /add-model Phase 4 or Phase 6 to prototype or parity-debug one non-DiT, non-VAE, non-encoder FastVideo component.
---

# Add Model Port Generic

## Goal

Prototype or parity-debug one scheduler, conditioner, upsampler, vocoder,
adapter, preprocessor, or unknown component in FastVideo-native code.

## Inputs

Follow `../add-model/shared/component_skill_common.md` and require the complete
packet from `../add-model/contracts/component_context.md`.

Generic-component packet fields:

- `component`: component name.
- `component_type`: scheduler, conditioner, upsampler, vocoder, adapter,
  preprocessor, or unknown.
- `parity_test`: `tests/local_tests/<bucket>/test_<family>_<component>_parity.py`.
- `weights`: converted component dir, HF subfolder, or none.
- `target_files`: matching `fastvideo/models/` and `fastvideo/configs/models/`
  bucket files when applicable.

## Modes

Use the common prototype and parity-debug modes from
`../add-model/shared/component_skill_common.md`.

Generic-component prototype concerns include stateless/stateful ambiguity,
missing loader buckets, source prefixes, mutable scheduler state, and output
container shape.

## Reuse Proof

Apply the shared reuse proof. Generic-component comparison must include mutable
state, scaling constants, scheduler/conditioner semantics, output containers, and
whether the component owns state or is stateless.

## Existing FastVideo Patterns

- Schedulers live under `fastvideo/models/schedulers/` and expose `EntryClass`.
- Upsamplers use `fastvideo/models/upsamplers/` plus configs under
  `fastvideo/configs/models/upsamplers/`; see `hunyuan15.py`.
- Vocoders and audio-specific modules can live under `fastvideo/models/audio/`
  with configs under `fastvideo/configs/models/audio/`; see `ltx2_audio_vae.py`.
- Compound conditioners may fit the encoder bucket when the pipeline loader uses
  `ConditionerLoader`; see `stable_audio_conditioner.py`.
- Registry discovery uses `EntryClass`; config bucket exports are required when
  pipeline configs import them by bucket.
- Use the narrowest matching config bucket. Wrong bucket inheritance can typecheck
  but fail during pipeline wiring.
- Layer guidance: `fastvideo/layers/AGENTS.md`.

## Bucket Decision

- If the component is a transformer/DiT, stop and use `add-model-03-port-dit`.
- If the component is a VAE/autoencoder, stop and use `add-model-04-port-vae`.
- If the component is a text/image/audio encoder or encoder-like conditioner,
  stop and use `add-model-05-port-encoder` unless the loader requires a different
  bucket.
- Otherwise choose the narrowest existing bucket. Add a new bucket only when no
  existing loader/config shape can represent the component without misleading
  names or unsafe runtime behavior.

## Implementation Rules

- Match official behavior, not just shapes: constructor args, default values,
  runtime flags, RNG use, dtype/autocast, scaling constants, masks, and output
  containers all matter.
- Keep the implementation minimal and native. Do not keep a runtime import of
  the official implementation as the production component.
- For schedulers, compare timesteps, sigmas/noise levels, step outputs, shift
  handling, prediction type, and any mutable internal state.
- For upsamplers, compare resize mode, align_corners, residual branches,
  causal padding, normalization, and exact target-shape behavior.
- For vocoders/audio components, compare waveform shape, sample-rate contract,
  channel order, hop length, normalization, and dtype.
- If private upstream deps are required only for tests, keep stubs under
  `tests/local_tests/helpers/` and do not import them from production code.

## Prototype Checks

Follow the shared prototype success criteria.

## Parity-Debug Loop

Run the shared parity-debug loop. The component test command is:

```bash
pytest <parity_test> -v -s
```

For numerical drift, add targeted intermediate comparisons in the test to
identify the first divergent operation.

## Escape Hatches

Follow `../add-model/shared/common_rules.md` and the component-specific guidance
in `../add-model/shared/component_skill_common.md`. Generic-component ask cases
include creating a new loader bucket, accepting an unsupported private op,
choosing between incompatible official definitions, or dropping a required
component.

## Handoff

Return `../add-model/contracts/component_skill_handoff.md` following the common
handoff rules in `../add-model/shared/component_skill_common.md`.
