---
name: add-model
description: Manual /add-model workflow for implementing a FastVideo model or first-class component port after add-model-01-prep has staged reference code and weights. Organizes the port into numbered phases with conversion rules, component policies, parity gates, and handoff checks.
---

# Add Model

## Manual Invocation

This skill is for explicit `/add-model` use only. Do not auto-start it from a
casual model-port mention. The setup-only workflow is
`../add-model-01-prep/SKILL.md`.

## Goal

Port a new FastVideo model family, model variant, or first-class reusable
component so it can be loaded through FastVideo's native model, config, stage,
registry, preset, and test infrastructure.

FastVideo has one pipeline architecture: stage-based composition via
`ComposedPipelineBase`. Vary the stages and modules, not the architecture.

## Scope Shapes

Use this skill for either shape:

| Shape | Required output |
|---|---|
| Full model family or variant | Native components, conversion if needed, pipeline config/class, presets, registry, smoke test, local parity tests, example, quality regression. |
| First-class component contribution | Native component class/config, bucket export, component parity test, and a documented downstream pipeline that will consume it. Skip pipeline/preset/registry rows only when the contribution is intentionally component-only. |

If upstream ships many variants, lock scope before coding. "Base model" means
checkpoint variant, not a modality subset. If the base checkpoint produces
audio, pose, depth, masks, or other output heads, either support those outputs
or get explicit user agreement to drop them.

## Required Input

Start from an `add-model-01-prep` handoff, or equivalent fields matching
`contracts/prep_handoff.md`.

Before Phase 0, read the shared rules and all relevant schemas:

- `shared/common_rules.md`
- `contracts/prep_handoff.md`
- `contracts/port_state.md`
- `contracts/escape_hatch.md`
- `contracts/component_context.md`
- `contracts/parity_status.md`
- `contracts/conversion_request.md`
- `contracts/conversion_handoff.md`
- `contracts/component_skill_handoff.md`
- `contracts/pipeline_context.md`
- `contracts/pipeline_handoff.md`
- `contracts/final_handoff.md`

## Hard Rules

- Follow `shared/common_rules.md` for token/auth safety, state files, escape
  hatches, production import boundaries, and skip/pass semantics.
- If the prep handoff is missing or ambiguous, stop and run
  `../add-model-01-prep/SKILL.md`.
- If a needed component is not ported, do not ship the pipeline that needs it.
- Wan is grandfathered for missing local parity; do not copy its missing-test
  precedent for new work.

## Escape Hatches

Follow `shared/common_rules.md` and `contracts/escape_hatch.md`. The main
orchestrator should ask only when no phase skill can safely continue under the
shared rules.

## Files Map

| Area | Paths |
|---|---|
| DiT | `fastvideo/models/dits/<family>.py`, `fastvideo/configs/models/dits/<family>.py`, bucket `__init__.py`. |
| VAE | `fastvideo/models/vaes/<arch_or_family>.py`, `fastvideo/configs/models/vaes/<arch_or_family>.py`, bucket `__init__.py`. Name by shared arch when reusable (`oobleck.py`, `autoencoder_kl.py`), otherwise by family (`wanvae.py`). |
| Encoder / conditioner / scheduler / upsampler | Native class/config in the matching `fastvideo/models/<bucket>/` and `fastvideo/configs/models/<bucket>/` bucket. |
| Lazy loader wrapper | Optional `fastvideo/models/<bucket>/<family>_loader.py` or similar thin `nn.Module` wrapper when a component is fetched from an external HF repo and should be hidden from host-pipeline state-dict matching. |
| Conversion | `scripts/checkpoint_conversion/<family>_to_diffusers.py` only when `needs_conversion=yes`. |
| Pipeline | `fastvideo/pipelines/basic/<family>/<family>_pipeline.py` plus sibling files for variants whose components or required modules differ. |
| Pipeline config | `fastvideo/configs/pipelines/<family>.py` or `fastvideo/pipelines/basic/<family>/pipeline_configs.py`. |
| Stages | `fastvideo/pipelines/basic/<family>/stages/` only for model-specific stage subclasses. |
| Presets / registry | `fastvideo/pipelines/basic/<family>/presets.py`, `fastvideo/registry.py`. |
| Tests | Component parity under `tests/local_tests/<bucket>/`; pipeline smoke/parity under `tests/local_tests/pipelines/`; CI-backed quality tests under `fastvideo/tests/`. |
| Example | `examples/inference/basic/basic_<family>*.py`, one per public mode/variant. |

## Phase 0: Scope And Handoff Gate

1. Validate every required handoff field.
2. Resolve `needs_conversion=unknown` before component work:

```bash
python ".agents/skills/add-model-01-prep/scripts/inspect_hf_layout.py" \
    "<hf-or-local-path>" \
    --json
```

3. List first-PR scope across both axes:
   - Variant axis: base, distill, SR/refine, causal, DMD, I2V, V2V, etc.
   - Modality axis: video, image, audio, pose, depth, masks, text, etc.
4. For component-only work, explicitly name the downstream full-pipeline PR or
   planned consumer.
5. Confirm `official_env_status` is `imports_ok` or
   `private_deps_need_stubs`. If it is `blocked`, return to
   `../add-model-01-prep/SKILL.md` before parity scaffolding.
6. Confirm `local_tests_readme` exists and records official setup, HF weights,
   dependency changes, and planned parity commands for reviewers.
7. Confirm `port_state_file` exists, follows `contracts/port_state.md`, and has
   rows for open questions/issues found during prep.
8. If there are multiple official implementations, choose the one whose
   architecture matches the published weights. A blessed library port can be a
   better parity reference than a highly configurable research repo; document
   the choice in tests.

## Phase 1: Reference And Architecture Study

Read the official pipeline call path before writing code.

Record:

- Required modules from `model_index.json` or equivalent: transformer, VAE,
  text encoders, tokenizers, scheduler, image encoders, audio VAE, vocoder,
  conditioners, upsamplers.
- Input/output modalities and every dedicated DiT output head.
- Text/image/audio encoding flow, latent shape, dtype, scaling, packing,
  scheduler/timestep math, guidance math, VAE normalization, and decode flow.
- Whether the official code relies on private deps, custom ops, or special
  kernels that parity tests must stub.

Arch config rule:

- `ArchConfig` fields must match the emitted per-component config, especially
  `transformer/config.json`, one-to-one.
- Pipeline knobs do not belong on the DiT arch config: inference steps, CFG
  scales, flow shift, FPS, VAE stride, text target length, data-proxy knobs,
  eval defaults, and sampling defaults go on `PipelineConfig`, presets, or
  stages.
- If the HF repo is raw or has empty configs, synthesize
  `transformer/config.json` from the official Python model-config class, not
  from data/eval config classes.

## Phase 2: Early Parity Scaffolding

Create component parity tests before or alongside implementation. Use
`../add-model-02-parity/SKILL.md` and its `templates/component_parity_test.py`.
The official reference must import in the current FastVideo environment, or the
prep handoff must identify private deps that will be stubbed locally for tests.
Use `local_tests_readme` as the reviewer-facing source for setup commands and
update its planned test table as parity scaffolds are added.

This phase is early by design:

- Official loading can be implemented from the reference study.
- FastVideo loading can target planned standardized class/config/loader paths.
- Tests may initially skip because the FastVideo class or converted weights do
  not exist yet.
- The scaffold must still contain real official loading, deterministic inputs,
  output extraction, and concrete tensor comparisons. No unconditional skips,
  no shape-only tests.

Use subagents here: dispatch one parity-test subagent per required component,
including components that may be reused. Their output becomes the red/skip
target that porting or reuse-verification subagents make pass later.

## Phase 3: Reuse Gate And Component Dispatch

Build a component inventory before implementation:

| Field | Meaning |
|---|---|
| Component | transformer, VAE, text encoder, image encoder, scheduler, conditioner, upsampler, vocoder, etc. |
| Official definition | Repo-relative source file, class/function name, and relevant line/range if known. |
| Official instantiation | Repo-relative pipeline/config/factory call site plus constructor args and runtime flags. |
| FastVideo target | Existing class to reuse or new bucket/file/config to add. |
| Parity test | Required local test path, including reused components. |
| Status | `reuse_pending`, `reuse_proven`, `port_pending`, `non_skip_pass`, or `blocked`. |

Reuse is allowed only from the checked-out FastVideo tree. Do not wait for or
depend on an open PR adding a native class; add the native port directly in this
PR if the current tree cannot be reused.

Reuse decision:

1. Record exact official definition and instantiation evidence for every
   component.
2. If an existing FastVideo class and config match both definition and
   instantiation, pass that reused target to the bucket-specific skill in
   `mode=prototype` and require reuse evidence plus key/shape dumps.
3. If either definition or instantiation differs, port the component directly as
   FastVideo-native code through the bucket-specific skill.
4. Reused components still require non-skip component parity against the exact
   official instantiation used by the target pipeline.

Porting subagent dispatch:

- Dispatch one subagent per component after Phase 2 parity scaffolds exist.
- Use `../add-model-03-port-dit/SKILL.md` for DiTs/transformers.
- Use `../add-model-04-port-vae/SKILL.md` for VAEs.
- Use `../add-model-05-port-encoder/SKILL.md` for text, image, audio, or compound
  encoders/conditioners that fit the encoder config bucket.
- Use `../add-model-06-port-generic/SKILL.md` for schedulers, upsamplers,
  vocoders, adapters, preprocessors, or unknown components.
- Each subagent owns one component only and must loop on that component's local
  parity test until it produces a non-skip PASS or returns a precise blocker.

Every component subagent must receive a complete packet matching
`contracts/component_context.md`. If any required path is unknown, pass `unknown`
plus the exact search already performed. Do not silently omit ambiguous official
files or prototype concerns.

Bucket, layer, and attention rules live in the bucket-specific skills and
`fastvideo/layers/AGENTS.md`.

## Phase 4: Native Component Prototype

Conversion needs a FastVideo state-dict surface. Use the Phase 3
bucket-specific skill in `mode=prototype` for every required component, including
reused components.

Prototype success criteria:

- the FastVideo-native or reused class/config can import and instantiate with the
  exact official architecture args;
- official and FastVideo key/shape dumps exist for every stateful component;
- `local_tests_readme` and `port_state_file` record prototype status and concerns;
- the returned handoff matches `contracts/component_skill_handoff.md`.

Do not chase numerical parity in Phase 4. Prototype mode ends when conversion has
the key/shape surface it needs, or when the component skill returns a precise
blocker or escape hatch.

## Phase 5: Param Mapping And Weight Conversion

Use `../add-model-07-conversion/SKILL.md` after Phase 4 prototypes exist.
Send a request matching `contracts/conversion_request.md`; consume the returned
`contracts/conversion_handoff.md` update before Phase 6.

Use the prep handoff's `needs_conversion` value:

- `no`: verify the source already has the component layout FastVideo loaders can
  consume, then record any passthrough components.
- `yes`: write `scripts/checkpoint_conversion/<family>_to_diffusers.py` and
  output `converted_weights/<family>/`.
- `unknown`: return to Phase 0.

The conversion skill owns source-layout handling, mapping derivation, config and
`model_index.json` emission, passthrough assets, strict-load verification, and
Phase 6 retry requests. Component skills must not patch conversion scripts or
converted weights ad hoc.

## Phase 6: Component Parity Debug

This is the expected expensive loop. Dispatch one subagent per required
component, including reused components, using the bucket-specific skill in
`mode=parity-debug`.

Each subagent gets:

- the complete component context packet from Phase 3/4;
- updated conversion mapping notes and strict-load result from Phase 5;
- any prototype concerns or unknowns that were not resolved before conversion.

The bucket-specific skills own parity-debug tactics. If a failure belongs to
conversion, route it through `../add-model-07-conversion/SKILL.md` with a retry
request matching `contracts/conversion_request.md`, then resume the component
skill with the updated conversion handoff.

Phase 6 ends only when every required component handoff reports
`parity_status=non_skip_pass`, or when a precise blocker or escape hatch is
recorded in `port_state_file`.

## Phase 7: Pipeline, Stages, And Variants

Do not start Phase 7 until every required component, reused or ported, has a
non-skip local parity PASS from Phase 6. If any component parity test is still
`scaffold_skip`, `debug_red`, `blocked`, or missing, resume Phase 6 first.

Use `../add-model-09-pipeline/SKILL.md` for pipeline definition and parity-debug.
Send a complete packet matching `contracts/pipeline_context.md`; consume the
returned `contracts/pipeline_handoff.md` before moving to quality regression or
final handoff.

The pipeline skill owns:

- pipeline class, stage chain, and optional model-specific stages;
- pipeline config, presets, registry updates, and examples;
- official args/defaults/presets comparison before setting FastVideo defaults;
- pipeline smoke and parity tests;
- continuous pipeline parity-debug until non-skip PASS or precise blocker;
- updates to `local_tests_readme` and `port_state_file`.

The pipeline handoff must explicitly cover stage order, variants, modality and
output-head handling, config/preset/registry/example status, smoke/parity tests,
and any return-to-Phase-6 evidence.

## Phase 8: PipelineConfig, Presets, Registry, Examples

This phase is implemented through `../add-model-09-pipeline/SKILL.md` after the
Phase 7 component-parity gate passes. Accept the pipeline handoff only if it
covers configs, presets, registry detection/exact class resolution, examples,
new `SamplingParam` fields for public kwargs/defaults, and local smoke/parity
status. Detailed rules live in `../add-model-09-pipeline/SKILL.md`.

## Phase 9: Parity Activation And Local Verification

Local parity is author-run, not CI-enforced. CI may only run package-level
quality tests later. Before handoff, Phase 2 scaffolds must be activated into
non-skip PASS results.

Order is mandatory:

1. Run conversion if needed.
2. Run component parity for every required component, including reused ones.
3. Run pipeline smoke.
4. Run pipeline parity.
5. Run the basic example.

If pipeline smoke or parity points back to component implementation,
strict-load, or conversion mapping, return to Phase 6 or Phase 5 rather than
patching around the issue in the pipeline.

Skip policy:

- Follow `shared/common_rules.md`: a committed local test may skip for absent
  clones/weights, but a local skip is not a verified pass.

Use the commands and tolerance guidance from `../add-model-02-parity/SKILL.md` for
component checks and from `../add-model-09-pipeline/SKILL.md` for pipeline smoke,
pipeline parity, and examples. Record exact commands, status, and blockers in
`local_tests_readme` and `port_state_file`.

## Phase 10: Quality Regression

Video outputs:

- Add `fastvideo/tests/ssim/test_<family>_similarity.py` when output video
  quality must be preserved.
- Seed references through `seed-ssim-references` after the test exists.

Audio outputs:

- SSIM does not apply. Use an audio-specific regression metric such as
  mel-spectrogram L1, multi-resolution STFT, CLAP cosine, or a project-approved
  learned metric.
- Document the metric and hardware/runtime assumptions in the test.

Joint AV outputs:

- Keep video and audio regression checks separate unless there is a validated
  joint metric.

## Phase 11: Post-Parity Review And Handoff

After parity is green, run a hot-path review before handoff:

- Hoist constant tensor allocations out of sampler/denoising loops.
- Replace per-step `randn_like` churn with preallocated buffers plus
  `.normal_()` when safe.
- Move `torch.backends.*` flag changes to one-shot setup/load paths.
- Delete `batch.extra` writes that nothing reads.
- Derive magic constants from configs when possible.

Pre-handoff checklist:

```text
[ ] Prep handoff is complete and committed nowhere with token values.
[ ] Conversion was run if needed and output loads with real weights.
[ ] Every required component, reused or newly ported, has a non-skip local parity PASS.
[ ] `local_tests_readme` lists every component parity test, command, status, and blocker if any.
[ ] `port_state_file` has every open question/issue either resolved or listed as an explicit blocker.
[ ] Any `next_step=ask_user` has a matching `escape_hatch` block and `E###` row.
[ ] Pipeline smoke has a non-skip local PASS.
[ ] Pipeline parity has a non-skip local PASS against the official reference.
[ ] Basic example runs and writes a non-corrupt output.
[ ] Video SSIM or audio-specific quality regression is added or explicitly deferred.
[ ] Runtime production code has no diffusers/transformers model-class imports.
[ ] Production comments are WHY-focused; examples have user-story docstrings.
[ ] Post-parity hot-path pass is complete.
```

Ask before deleting any reference clone or staged weights created by
`add-model-01-prep`. Leave `.gitignore` entries so future parity assets stay
untracked. Never commit the clone, weights, `.env`, credentials, or anything
matching `*secret*`.

## References

- `../add-model-01-prep/SKILL.md` for user-input collection, HF inspection,
  weight staging, reference cloning, and setup handoff.
- `contracts/` for canonical handoff schemas used by prep, parity, conversion,
  component porting, escape hatches, and final handoff.
- `../add-model-02-parity/SKILL.md` for early component parity scaffolds and
  activation templates.
- `../add-model-07-conversion/SKILL.md` for Phase 5 mapping, conversion scripts,
  monolithic checkpoint splitting, and strict-load checks.
- `../add-model-03-port-dit/SKILL.md`, `../add-model-04-port-vae/SKILL.md`,
  `../add-model-05-port-encoder/SKILL.md`, and
  `../add-model-06-port-generic/SKILL.md` for component subagent implementation
  and parity-debug loops.
- `../add-model-09-pipeline/SKILL.md` for pipeline definition, config/preset/
  registry/example wiring, smoke tests, and pipeline parity-debug.
- `fastvideo/layers/AGENTS.md` for native layer selection and state-dict surface
  guidance.
- `docs/contributing/coding_agents.md` for narrative context.
- `docs/design/overview.md` for pipeline/config/registry architecture.
- `fastvideo/pipelines/basic/wan/` for standard T2V/I2V/DMD/Causal variants.
- `fastvideo/pipelines/basic/ltx2/` for non-standard stages and audio/video
  patterns.
- `tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py` for pipeline
  parity shape.
- `tests/local_tests/transformers/test_ltx2.py`,
  `tests/local_tests/vaes/test_ltx2_vae.py`, and
  `tests/local_tests/encoders/test_ltx2_gemma_parity.py` for component parity.
- `scripts/checkpoint_conversion/convert_ltx2_weights.py` for modern conversion
  script shape.
- `scripts/checkpoint_conversion/wan_to_diffusers.py` for legacy regex mapping
  reference only.
- `REVIEW.md` is historical; its decisions are incorporated here as of
  2026-04-30.

## Changelog

| Date | Change |
|---|---|
| 2026-04-24 | Initial FastVideo add-model workflow. |
| 2026-04-30 | Split external setup into `add-model-01-prep`. |
| 2026-04-30 | Rewrote as manual `/add-model` phase workflow and incorporated `REVIEW.md` decisions. |
| 2026-04-30 | Extracted early parity scaffolding into `add-model-02-parity` and moved it before conversion/component implementation. |
| 2026-04-30 | Added component reuse proof gate, bucket-specific porting skills, and parity PASS requirement for reused components. |
| 2026-04-30 | Split prototype, conversion, and parity-debug phases; added conversion skill for monolithic and separate checkpoint layouts. |
| 2026-04-30 | Extracted handoff schemas into `contracts/` for shared use across skills. |
| 2026-04-30 | Added pipeline skill contract and Phase 7 component-parity gate. |
| 2026-04-30 | Added escape-hatch contract for user decisions and `ask_user` handoffs. |
