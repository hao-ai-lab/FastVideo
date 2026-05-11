---
name: add-model-10-pr-review
description: Review rubric for FastVideo PRs that add or modify model families, variants, first-class components, checkpoint conversion, pipelines, parity coverage, or generated-media quality baselines. Use when reviewing a PR whose diff touches fastvideo/models/, fastvideo/pipelines/basic/, fastvideo/registry.py, scripts/checkpoint_conversion/, fastvideo/tests/ssim/, or related model-port surfaces. Pairs with review-pr-link as a project-scoped review pass; produces findings, not fixes.
---

# Add-Model PR Review

Use this skill when a reviewed PR appears to add, port, or substantially modify
a FastVideo model family, model variant, first-class model component,
checkpoint conversion, model pipeline, or local parity coverage.

This is a review skill, not an implementation workflow. Do not run `/add-model`
or start writing missing port code during review. Use the add-model skill stack
as a rubric for findings.

## Trigger Paths

Trigger this skill if `git diff --name-only <base>...HEAD` includes any of:

- `fastvideo/models/dits/`, `fastvideo/configs/models/dits/`
- `fastvideo/models/vaes/`, `fastvideo/configs/models/vaes/`
- `fastvideo/models/encoders/`, `fastvideo/configs/models/encoders/`
- `fastvideo/models/schedulers/`, `fastvideo/configs/models/schedulers/`
- `fastvideo/models/upsamplers/`, `fastvideo/configs/models/upsamplers/`
- `fastvideo/models/audio/`, `fastvideo/configs/models/audio/`
- `fastvideo/pipelines/basic/`, `fastvideo/configs/pipelines/`
- `fastvideo/registry.py`, `fastvideo/api/sampling_param.py`
- `scripts/checkpoint_conversion/`
- `examples/inference/basic/`
- `tests/local_tests/`, especially component or pipeline parity tests
- `fastvideo/tests/ssim/` or other quality-regression tests for generated media

Also trigger when the PR title/body claims a new model, model variant, VAE,
encoder, scheduler, conditioner, pipeline, conversion script, or generated-media
quality baseline even if the path list is incomplete.

## Review Inputs

Read these add-model references as review checklists:

- `../add-model/SKILL.md`: phase gates and final handoff requirements.
- `../add-model/shared/common_rules.md`: token/auth safety, production import
  boundaries, state files, and skip/pass semantics.
- `../add-model/contracts/final_handoff.md`: final evidence expected from a
  complete port.
- `../add-model/contracts/component_context.md` and
  `../add-model/contracts/component_skill_handoff.md`: component evidence and
  parity-debug expectations.
- `../add-model/contracts/conversion_request.md` and
  `../add-model/contracts/conversion_handoff.md`: conversion evidence,
  strict-load status, config validation, and retry context.
- `../add-model/contracts/pipeline_context.md` and
  `../add-model/contracts/pipeline_handoff.md`: pipeline class/stage/config/
  preset/registry/example evidence.

Then read only the satellite skill(s) that match touched areas:

- DiT/transformer changes: `../add-model-03-port-dit/SKILL.md`.
- VAE changes: `../add-model-04-port-vae/SKILL.md`.
- Encoder/conditioner changes: `../add-model-05-port-encoder/SKILL.md`.
- Scheduler/upsampler/vocoder/other components:
  `../add-model-06-port-generic/SKILL.md`.
- Component parity tests: `../add-model-02-parity/SKILL.md`.
- Checkpoint conversion: `../add-model-07-conversion/SKILL.md`.
- Pipeline/config/presets/registry/examples:
  `../add-model-09-pipeline/SKILL.md`.
- Prep/state docs: `../add-model-01-prep/SKILL.md`.

## Required Review Lanes

For a full model-family or model-variant PR, cover all lanes. For a
component-only PR, cover the component, conversion/parity as applicable, and the
documented downstream consumer.

1. Scope and source-of-truth lane:
   Verify the PR clearly identifies the official reference, weights/revision,
   supported variants, modalities, output heads, and any approved scope cuts.

2. Component lane:
   Verify each required component is FastVideo-native or has a documented and
   accepted lazy-wrapper exception. Check bucket/config inheritance, `EntryClass`,
   state-dict surface, reused-component evidence, and output heads.

3. Conversion lane:
   Verify mappings are derived from prototype key/shape dumps, source layout is
   supported, skipped keys are intentional, emitted configs validate through
   production paths, component strict-load status is recorded, `model_index.json`
   library tokens match loaders, and revisions are pinned when converting from
   HF.

4. Component parity lane:
   Verify local parity tests exist for every required component, including reused
   components. Scaffolds may skip in CI, but the PR must provide local non-skip
   PASS evidence or an explicit accepted blocker.

5. Pipeline lane:
   Verify stage order, required modules, `_class_name` / `EntryClass.__name__`
   resolution, config defaults, presets, `SamplingParam` fields, registry
   registration, examples, smoke tests, and pipeline parity.

6. Quality and evidence lane:
   Verify media quality regression is added or explicitly deferred, examples run,
   generated outputs are non-corrupt, `tests/local_tests/<family>/README.md` and
   `PORT_STATUS.md` are current, and final blockers are surfaced in the review.

## Findings To Prioritize

Prioritize review findings in this order:

- Missing or skipped required component parity without accepted blocker.
- Pipeline parity/smoke/example missing or skipped for a pipeline PR.
- Conversion emits unloadable or unvalidated configs/weights.
- Wrong `model_index.json` `_class_name`, component library token, or registry
  class resolution.
- Runtime diffusers/transformers model-class imports for components that own
  weights or numerical behavior.
- Dropped modalities, output heads, variants, or conditioning streams without
  explicit approval.
- Reused FastVideo component lacks exact definition/instantiation proof or
  non-skip parity.
- Public generation kwargs/preset defaults missing from `SamplingParam`.
- Tests only check shapes, importability, or successful generation without
  numerical/media comparison.
- Tokens, credentials, reference clones, staged weights, or generated bulk assets
  committed to the PR.

## Output Format

Write normal code-review findings first, ordered by severity. Include file and
line references from the PR diff when possible.

Use this phrasing for missing add-model evidence:

```text
This PR does not satisfy the add-model <component|conversion|pipeline|final>
gate because <specific required evidence> is missing. The risk is <runtime load,
numerical parity, dropped output, registry resolution, etc.>.
```

Keep the summary short. Mention which lanes were reviewed and which could not be
verified because assets, GPU time, or external credentials were unavailable.
