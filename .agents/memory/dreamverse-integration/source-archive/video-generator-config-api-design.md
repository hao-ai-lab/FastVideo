# Exploration Log: Video Generator Config API Design

## Status: draft

## Context
FastVideo's Python inference API currently mixes generator-instance settings,
pipeline initialization settings, and per-request sampling/runtime settings
through broad `**kwargs` surfaces on `VideoGenerator.from_pretrained(...)` and
`VideoGenerator.generate_video(...)`.

This exploration compares the current FastVideo design with
`sglang/multimodal_gen` and examines how to upstream multi-stage LTX2 /
Dreamverse behavior without growing more ad hoc top-level flags.

## Progress
- [x] Read FastVideo onboarding, codebase map, and relevant design docs.
- [x] Inspect current FastVideo generator, args, sampling, registry, and
  workflow abstractions.
- [x] Inspect internal LTX2 streaming server usage and current two-stage /
  continuation requirements.
- [x] Inspect SGL diffusion generator, server args, sampling params, and
  request preparation boundary.
- [x] Inspect vLLM-Omni stage config, stage metadata, request, and orchestration
  surfaces for multi-stage pipeline ideas.
- [x] Inspect current FastVideo CLI/config-file loading and compare with the
  training YAML-only entrypoint.
- [ ] Convert findings into a concrete implementation plan for FastVideo.

## Findings
- FastVideo already has the right internal separation points:
  `FastVideoArgs`, `PipelineConfig`, `SamplingParam`, and `ForwardBatch`.
- The public boundary is the unstable part:
  init-time and request-time knobs are mixed through `**kwargs`.
- Unknown init keys can be silently filtered, while unknown request keys can be
  only logged rather than rejected. This makes API drift hard to detect.
- SGL's split is cleaner:
  `ServerArgs` for engine/runtime, `PipelineConfig` for model-family wiring,
  and `SamplingParams` for per-request settings.
- SGL also has better merge semantics for user request overrides:
  it preserves model defaults, tracks explicitly provided fields, and validates
  request params against pipeline task type.
- SGL still has a design smell worth avoiding in FastVideo:
  `SamplingParams._adjust(...)` depends on `ServerArgs`, which leaks
  engine/pipeline concerns back into the request object.
- vLLM-Omni contributes a useful extra abstraction beyond SGL:
  model-owned multi-stage topology via `ModelPipeline` and `StageConfig`,
  with per-stage defaults (`default_sampling_params`) and runtime override
  layering.
- vLLM-Omni's best reusable idea for FastVideo is not the serving stack, but
  the separation between:
  1. model-defined stage topology and per-stage defaults,
  2. runtime engine overrides,
  3. request-time sampling/state handoff.
- vLLM-Omni also shows the downside of exposing stage-indexed request lists too
  directly: `sampling_params_list` works for a serving engine, but is too
  positional and low-level for FastVideo's higher-level Python API.
- FastVideo already supports YAML/JSON config files for inference CLI, but the
  current mechanism flattens nested documents back into argparse flags. This
  preserves backward compatibility but keeps the CLI surface as the canonical
  schema instead of a typed document model.
- The training stack has a cleaner precedent: a YAML-first config loaded into a
  typed schema, with dotted CLI overrides applied onto the nested document
  before parsing. Inference can likely adopt a lighter variant of that pattern.
- Multi-stage generation should be unified at the orchestration layer, not by
  forcing LongCat refine, Hunyuan SR, and LTX2 continuation into one leaf config.

## Mistakes / Dead Ends
- A fully free-form string-dict API would lose too much type safety and would
  likely recreate the current drift problem under a different shape.
- A single universal `RefineConfig` for all models would become a sparse bag of
  nullable fields and would not map cleanly to existing model families.

## Proposed Standardization
- Introduce a typed public split:
  `GeneratorConfig` for instance-lifetime engine/init settings and
  `GenerationRequest` for per-call inputs/sampling/output.
- Allow dict input only as an interchange layer that is parsed immediately into
  typed configs with strict unknown-key validation.
- Add a typed `GenerationPlan` / multi-stage orchestration layer with
  discriminated stage configs:
  `SampleStageConfig`, `LongCatRefineStageConfig`,
  `HunyuanSRStageConfig`, `LTX2ContinuationStageConfig`.
- Let model families own stage defaults and stage topology through named
  profiles or model-defined stage plans, similar in spirit to vLLM-Omni's
  pipeline YAMLs, but expose them through typed Python config objects rather
  than raw stage-indexed lists in the primary API.
- Make YAML/JSON a first-class serialization of the same typed inference
  schema, not just a file format that expands into CLI flags.
- Prefer a YAML-first CLI pattern for nested configs:
  `fastvideo generate --config run.yaml --request.sampling.seed 42`,
  while keeping a compatibility layer for existing flat flags during migration.
- Upstream LTX2 two-stage / continuation behavior as a first-class stage or
  pipeline profile rather than more `ltx2_*` top-level kwargs.
