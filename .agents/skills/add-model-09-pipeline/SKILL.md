---
name: add-model-09-pipeline
description: Use during /add-model Phase 7 after all required component parity tests pass to define FastVideo pipeline wiring, configs, presets, registry entries, examples, smoke tests, and pipeline parity tests.
---

# Add Model Pipeline

## Goal

Implement and verify the end-to-end FastVideo pipeline after the native
components and converted weights have passed non-skip component parity. This
skill owns pipeline class/stage wiring, pipeline configs, presets, registry
entries, examples, smoke tests, and pipeline parity-debug.

FastVideo has one pipeline architecture: stage-based composition through
`ComposedPipelineBase`. Add or specialize stages only when existing stages cannot
represent the official behavior safely.

## Hard Gate

Do not start pipeline work until every required component, including reused
components, has a non-skip local parity PASS.

If any component row is missing, skipped, red, or blocked, return to `/add-model`
Phase 6. Pipeline parity cannot distinguish stage wiring mistakes from broken
component numerics when component parity is still unresolved.

## Inputs

Follow `../add-model/shared/common_rules.md` for token/auth safety, state files,
escape hatches, production boundaries, and skip/pass semantics.

Require a complete packet matching
`../add-model/contracts/pipeline_context.md`.

The packet must include:

- official pipeline files and official call/default sources;
- workload types, input/output modalities, and output contract;
- converted or source `model_index.json` path;
- component parity rows, all `non_skip_pass`;
- target FastVideo pipeline/config/preset/registry/example/test paths;
- `local_tests_readme` and `port_state_file` paths.

## Outputs

- Pipeline package under `fastvideo/pipelines/basic/<family>/`.
- Pipeline config under `fastvideo/configs/pipelines/<family>.py` or a documented
  family-local config file when that matches existing project style.
- Presets under `fastvideo/pipelines/basic/<family>/presets.py`.
- Registry updates in `fastvideo/registry.py`.
- Basic example under `examples/inference/basic/basic_<family>*.py`.
- Local smoke and parity tests under `tests/local_tests/pipelines/`.
- Updated `tests/local_tests/<model_family>/README.md`.
- Updated `tests/local_tests/<model_family>/PORT_STATUS.md`.
- Handoff matching `../add-model/contracts/pipeline_handoff.md`.

## Mode: Pipeline Definition

Use this mode first.

1. Read the official pipeline call path before editing FastVideo code.
2. Compare official defaults against the planned FastVideo config and presets:
   steps, CFG scales, secondary CFG, flow shift, schedulers, sigmas, seed/RNG,
   resolution, frames, FPS, duration, VAE scaling, decode slicing, negative
   prompt defaults, and output heads.
3. Create or update the pipeline class with `_required_config_modules` matching
   the emitted `model_index.json` and `ComposedPipelineBase.load_modules`.
   Runtime pipeline resolution is exact: `model_index.json["_class_name"]` must
   match a registered `EntryClass.__name__`, or a wrapper/alias class in
   `EntryClass`. Registry detectors do not select the executable pipeline class.
4. Add new public generation kwargs to `fastvideo/api/sampling_param.py` before
   examples or presets use them. `SamplingParam.update()` ignores unknown keys
   except for logging, and preset defaults apply only to declared fields. Add CLI
   args when the option should be available from command-line entrypoints.
5. Put loader-time changes in `load_modules()` or earlier, not
   `initialize_pipeline()`. `ComposedPipelineBase.__init__` loads modules before
   `post_init()` calls `initialize_pipeline()`, so process-global flags, loader
   path rewrites, dtype overrides, and tokenizer path changes needed for loading
   cannot be introduced there.
6. Use `self.get_module("transformer_2", None)` and similar optional accessors
   for truly optional modules. Do not hard-require optional modules by accident.
7. Avoid mutating class-level `_required_config_modules` in custom code. If a
   pipeline needs dynamic modules, copy the list to an instance-owned value or
   pass `required_config_modules` explicitly so one pipeline instance cannot leak
   module requirements into another.
8. Create the stage chain in official execution order. Prefer existing shared
   stages for standard text encoding, timestep preparation, latent preparation,
   denoising, and decoding.
9. Add model-specific stages only for family-specific behavior that does not fit
   the shared stage contracts.
10. Add pipeline config classes for wiring and runtime defaults. Do not duplicate
   component architecture fields unless a loader requires them in the subconfig.
    Family-local config files such as
    `fastvideo/pipelines/basic/<family>/pipeline_configs.py` are valid only when
    `fastvideo/registry.py` imports and registers the classes explicitly.
11. Add `InferencePreset` objects with `model_family`, `name`, `version`,
    `defaults`, optional validation-only `stage_schemas`, and an `ALL_PRESETS`
    tuple. `stage_schemas` validates user-facing `stage_overrides` names; it does
    not drive `create_pipeline_stages()` execution.
12. Register config classes and presets in `fastvideo/registry.py`: add
    `register_configs(...)`, import the family's `ALL_PRESETS`, and append it to
    `_register_presets()`. Detectors should cover HF paths and `_class_name`
    strings for config/preset lookup, but not as a replacement for exact pipeline
    class-name resolution.
13. Add a basic example with a user-story docstring and normal file-path inputs
    for image, audio, or video references. Keep orchestration glue in the
    pipeline or a helper, not in the example.
14. Add a separate smoke test
    `tests/local_tests/pipelines/test_<family>_pipeline_smoke.py` that proves
    imports, `EntryClass`, registry, presets, config defaults, and at least one
    real load/generate path when weights are local. Older local tests sometimes
    colocate smoke checks in parity files; new ports should use the separate file
    convention.
15. Add or update pipeline parity test scaffolding with
    `templates/pipeline_parity_test.py`.
16. Update `local_tests_readme` and `port_state_file` with commands, statuses,
    default sources, decisions, and blockers.

Production import boundaries are defined in
`../add-model/shared/common_rules.md`.

## Mode: Pipeline Parity Debug

Run after pipeline definition and after smoke can execute far enough to load the
pipeline. Loop until pipeline parity is a non-skip PASS or a precise blocker is
returned.

Mandatory order:

```bash
pytest tests/local_tests/pipelines/test_<family>_pipeline_smoke.py -v -s
DISABLE_SP=1 pytest tests/local_tests/pipelines/test_<family>_pipeline_parity.py -v -s
python examples/inference/basic/basic_<family>.py
```

Pipeline parity must compare real outputs, not only successful generation:

- denoised latents when decode parity is expensive or nondeterministic;
- decoded videos/images when visual output should be deterministic enough;
- decoded waveform or audio features for audio pipelines;
- separate video and audio targets for joint AV pipelines unless a validated
  joint metric exists.

Debug pipeline drift in this order:

1. Confirm both sides use the same component weights and component parity PASS
   results are still valid.
2. Align official and FastVideo call arguments, presets, and default values.
3. Align scheduler timesteps, sigmas/noise levels, prediction type, flow shift,
   guidance math, and secondary-guidance branches.
4. Align RNG: initial latents/noise, generator device, seed, per-step noise, VAE
   sampling, and any official `+1 frame` or crop/slice behavior.
5. Align conditioning: prompt templates, negative prompts, masks, image/audio
   preprocessing, modality packing, text truncation, and dtype/autocast.
6. Align decode: latent scaling, per-channel mean/std, tiling flags, output
   channel order, sample rate, FPS, and final slicing.
7. Add targeted stage-level diagnostics to identify the first divergent stage.

If the first divergence belongs to component implementation, strict loading, or
conversion mapping, stop pipeline edits and return `next_step=return_to_phase_6`
with the exact failing evidence. Do not patch conversion from this skill.

## Stage And Variant Rules

- Canonical video T2V order: `InputValidationStage`, `TextEncodingStage`,
  `ConditioningStage`, `TimestepPreparationStage`, `LatentPreparationStage`,
  `DenoisingStage`, `DecodingStage`.
- Canonical video I2V delta adds image loading/encoding and image VAE encoding in
  the official order, commonly: `TextEncodingStage`, `ImageEncodingStage`,
  `ConditioningStage`, `TimestepPreparationStage`, `LatentPreparationStage`,
  `ImageVAEEncodingStage`, `DenoisingStage`, `DecodingStage`.
- Treat `ConditioningStage` as default-present for Wan-style pipelines, but still
  follow the reference if another family truly skips or replaces it.
- T2V video pipelines usually use validation, text encoding, conditioning,
  timestep preparation, latent preparation, denoising, and decoding.
- I2V adds image loading/encoding and image-latent preparation according to the
  official pipeline, not by assuming CLIP or Wan-specific branches.
- Pick image, audio, and video encoders from the reference. Do not assume CLIP or
  any other common encoder unless the reference uses it.
- Cross-attention class names are not prescribed; match the family style and
  preserve the official tensor contract.
- `WorkloadType` currently has no `T2A`, `A2A`, or `AV` values. Until that enum
  is extended, audio-only pipelines may register with `WorkloadType.T2V` and
  preset `workload_type="t2v"` as a compatibility shim, but must document the
  rationale in code and `PORT_STATUS.md`.
- Audio-only pipelines should not force real video semantics into presets. Use
  minimal video-shaped placeholders such as small `height`/`width` and
  `num_frames=1` only when shared `VideoGenerator`/validation paths require them,
  and document that the real output is audio.
- Record modality-specific shape knobs and output contract in the pipeline
  handoff: video uses `height`, `width`, `num_frames`, and `fps`; audio uses
  `audio_seconds` and `sampling_rate`; joint AV records both plus whether output
  is muxed or paired files.
- Use sibling pipeline classes/configs when required modules, HF repo layout,
  stage chains, or inputs differ materially.
- Use one kwargs-driven pipeline class only when variants share weights,
  modules, stage chain, and safe call semantics.
- Split later if components diverge, workload tags require separate discovery,
  signatures become unsafe, or stage branches become substantial.
- If the DiT branches on `added_kv_proj_dim`, document the T2V/I2V split.
- If the reference uses `transformer_2`, `boundary_ratio`, `guidance_scale_2`, or
  DMD step lists, keep those on config, presets, or stages deliberately.
- Support every official output head in scope. If a head is out of scope, record
  explicit user approval in `PORT_STATUS.md`.

Pipeline verification order:

```bash
pytest tests/local_tests/pipelines/test_<family>_pipeline_smoke.py -v -s
DISABLE_SP=1 pytest -v -s tests/local_tests/pipelines/test_<family>_pipeline_parity.py
python examples/inference/basic/basic_<family>.py
```

Smoke tests prove loadability only. They are not a substitute for numerical
component or pipeline parity.

## Escape Hatches

Follow `../add-model/shared/common_rules.md`. Pipeline-specific ask cases include
dropping a public mode, modality, or output head; adding a new workload enum;
changing official defaults for user-facing behavior; accepting a known pipeline
parity blocker; running GPU-heavy quality work outside the agreed scope; or
publishing/uploading generated references or converted weights.

## Handoff

Return `../add-model/contracts/pipeline_handoff.md` and update the shared state
files before handoff.

Do not hand back a green pipeline if smoke or parity skipped locally. A skip is a
setup gap, not a pass.

## References

- `fastvideo/pipelines/composed_pipeline_base.py` for module loading and stage
  execution.
- `fastvideo/pipelines/basic/wan/` for standard video T2V/I2V/DMD variants.
- `fastvideo/pipelines/basic/stable_audio/` for audio-specific stage composition.
- `fastvideo/configs/pipelines/stable_audio.py` and
  `fastvideo/pipelines/basic/stable_audio/presets.py` for config/preset shape.
- `fastvideo/registry.py` for `register_configs(...)` and preset registration.
- `tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py` for latent
  parity structure.
- `tests/local_tests/pipelines/test_stable_audio_pipeline_parity.py` for audio
  parity structure.
- `tests/local_tests/pipelines/test_stable_audio_pipeline_smoke.py` for no-GPU
  import/registry/preset preflight shape.
