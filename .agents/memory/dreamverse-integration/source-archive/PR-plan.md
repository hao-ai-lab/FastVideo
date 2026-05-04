# FastVideo API Refactor PR Plan

## Related Documents
- [apirefactor.md](apirefactor.md) — design spec this plan implements
- [.agents/exploration/streaming-server-upstream-plan.md](.agents/exploration/streaming-server-upstream-plan.md) — plan for upstreaming the internal LTX2 streaming server and Dynamo backend contract (shapes PRs 5.5-7.10)
- `../FastVideo-internal/.agents/exploration/rebase-upstream-fastvideo.md` — rebasing FastVideo-internal onto upstream (in progress; enables PRs 6-8)
- `../FastVideo-internal/ui/ltx2-streaming/` — upstream source for streaming server (server/, gpu_pool.py, prompt_enhancer.py, router/)
- `../dynamo/` — local clone of ai-dynamo/dynamo; backend shape at `components/src/dynamo/sglang/` is the template for the FastVideo native backend consumed in PR 7.10
- https://github.com/ai-dynamo/dynamo/pull/7544 — draft PR (CLOSED) that promotes FastVideo to a native Dynamo backend; establishes the contract this plan must satisfy

## Status

Working plan for implementing the design in [apirefactor.md](apirefactor.md).

### PR Landing History

Landed:
- PR 0 + PR 1 as `[1/n]` (#1218) on `main` — parity inventory + typed inference schema + strict parser/validation/overrides + API test suite
- PR 2 as `[2/n]` (#1220) on `main` — typed `VideoGenerator` constructors and request path + compatibility translation
- PR 3 as `[3/n]` (#1226) on `main` — CLI/YAML-first typed config loading for `generate` and `serve`
- PR 4 as `[4/n]` (#1234) on `main` — preset registry + presets for all 13 model families; `SamplingParam` moved to `fastvideo/api/`; `fastvideo/configs/sample/` deleted entirely
- PR 5 as `[5/n]` (#1237) on `main` — `ServeConfig.default_request` wired into the stateless OpenAI server
- PR 5.5 as `[5.5/n]` (`5d1d71fc`) on `will/api_5.5` — streaming server package skeleton, typed `StreamingConfig`/`GpuPoolConfig`/`PromptEnhancerConfig`/`PromptSafetyConfig`/`WarmupConfig`, `streaming-serve` CLI stub
- PR 6 as `[6/n]` (#1239) on `main` — LTX2 public preset + asset wiring + gpu_pool translation
- PR 7 as `[7/n]` (#1250) on `main` — typed LTX2 continuation state + streaming session store + blob store

In progress:
- PR 7.5 (#1251) on `will/api_7.5` — streaming server skeleton (WebSocket + fMP4 + single generator). Open for review.
- PR 7.6 on `will/api_7.6` — GPU pool upstream. Branch ready, not yet PR'd; rebased on `will/api_7.5`.

Remaining (PR 7.7 onward):
- PR 7.7: prompt enhancer upstream with `LLMProvider` abstraction
- PR 7.8: streaming auxiliaries (prompt safety, session logger, mock server, rewrite)
- PR 7.9: router upstream
- PR 7.10: Dynamo backend contract — `VideoGenerator.generate_async` + health-check helper; the Dynamo backend package itself lives in the Dynamo repo
- PR 8: internal-UI ↔ public-server contract docs + Dynamo integration reference
- PR 9: LongCat preset migration + colocation
- PR 10: Hunyuan15 SR preset migration + colocation + SR field migration POC
- PR 11: SSIM/performance test migration
- PR 12: docs/examples migration (includes streaming server + Dynamo)
- PR 13: deprecation cleanup (includes flat LTX2 kwargs the internal gpu_pool used to consume)

PRs 5.5-7.10 are driven by the decision to upstream `FastVideo-internal/ui/ltx2-streaming/server/` into the public repo; see [streaming-server-upstream-plan.md](.agents/exploration/streaming-server-upstream-plan.md).

### Landed Artifacts

Reference points for later PRs:

- parity inventory: [inference_schema_parity_inventory.yaml](/home/william5lin/FastVideo/docs/design/inference_schema_parity_inventory.yaml) + guard [test_schema_parity_inventory.py](/home/william5lin/FastVideo/fastvideo/tests/api/test_schema_parity_inventory.py)
- typed public schema: [schema.py](/home/william5lin/FastVideo/fastvideo/api/schema.py)
- parser / validation / overrides: [parser.py](/home/william5lin/FastVideo/fastvideo/api/parser.py), [errors.py](/home/william5lin/FastVideo/fastvideo/api/errors.py), [overrides.py](/home/william5lin/FastVideo/fastvideo/api/overrides.py)
- compatibility translation + typed result: [compat.py](/home/william5lin/FastVideo/fastvideo/api/compat.py), [results.py](/home/william5lin/FastVideo/fastvideo/api/results.py)
- typed `VideoGenerator`: [video_generator.py](/home/william5lin/FastVideo/fastvideo/entrypoints/video_generator.py)
- CLI typed config loading: [inference_config.py](/home/william5lin/FastVideo/fastvideo/entrypoints/cli/inference_config.py), [generate.py](/home/william5lin/FastVideo/fastvideo/entrypoints/cli/generate.py), [serve.py](/home/william5lin/FastVideo/fastvideo/entrypoints/cli/serve.py), [main.py](/home/william5lin/FastVideo/fastvideo/entrypoints/cli/main.py)
- preset system: [presets.py](/home/william5lin/FastVideo/fastvideo/api/presets.py) plus `fastvideo/pipelines/basic/<family>/presets.py` for all 13 families (Wan, LTX2, Hunyuan, Hunyuan15, HYWorld, GameCraft, LingBotWorld, MatrixGame, GEN3C, Cosmos, Cosmos25, SD35, TurboDiffusion, LongCat); `model_family` + `default_preset` on `ConfigInfo` in [registry.py](/home/william5lin/FastVideo/fastvideo/registry.py)
- `SamplingParam` canonical home: [sampling_param.py](/home/william5lin/FastVideo/fastvideo/api/sampling_param.py) — all 12 subclass files removed; defaults flow through `SamplingParam.from_pretrained()` → `_from_preset()`
- API test suite: [fastvideo/tests/api](/home/william5lin/FastVideo/fastvideo/tests/api) (98+ tests)
- streaming package + config types (PR 5.5): `fastvideo/entrypoints/streaming/`
- LTX2 preset + asset wiring + typed pipeline / refine config (PR 6): `fastvideo/pipelines/basic/ltx2/`, `fastvideo/api/schema.py` typed `pipeline.preset_overrides`/`pipeline.components`, `compile.extras`, `gpu_pool.py` translation
- LTX2 typed continuation state (PR 7): [`fastvideo/pipelines/basic/ltx2/continuation.py`](/home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/continuation.py) with `LTX2ContinuationState`, `_pack_frame_blobs`/`_unpack_frame_blobs` (bounds-checked), public `ContinuationState` envelope on `GenerationRequest`, `register_continuation_kind` registry validation in [`compat.py`](/home/william5lin/FastVideo/fastvideo/api/compat.py)
- streaming session + blob stores (PR 7): [`fastvideo/entrypoints/streaming/session_store.py`](/home/william5lin/FastVideo/fastvideo/entrypoints/streaming/session_store.py)

### Notable Decisions Carried Forward From Landed PRs

- public inference boundary remains plain dataclasses + plain dict/YAML/JSON (not OmegaConf or runtime config wrappers)
- every public entrypoint normalizes into typed config objects before touching legacy `FastVideoArgs` or `SamplingParam`
- legacy `generate_video(..., **kwargs)` stays on its direct legacy execution path until PR 11's SSIM/performance migration — prevents golden baselines from drifting during the compat period
- typed requests use schema defaults while legacy `generate_video(...)` continues to inherit model-specific `SamplingParam` defaults during the compat period
- preset registry uses an explicit `_register_presets()` pattern matching `_register_configs()`; lookup keyed by `model_family`
- stateless OpenAI server clones `ServeConfig.default_request` and merges user overrides; preset validation runs before legacy generation
- streaming server added as sibling `fastvideo/entrypoints/streaming/` rather than extending `fastvideo/entrypoints/openai/` (PR 5.5 added only the package stub + typed config; live server lands in PRs 7.5-7.9)

## Goals

- Break the API refactor into reviewable PRs.
- Keep each PR testable and bisectable.
- Add explicit CI coverage for the new inference schema.
- Include the LTX2/Dreamverse upstream work as part of the rollout.
- Avoid one giant mixed refactor that is hard to land or debug.

## Ground Rules

- Every PR must leave the repo in a shippable state.
- Every PR must add or update tests with the code change.
- The new schema cannot freeze before a public-field parity inventory is reviewed.
- Compatibility shims stay in place until all downstream entrypoints are migrated.
- Every public inference entrypoint must normalize into typed config objects before touching legacy internals.
- LTX2 upstreaming should be staged so private Dreamverse behavior is not partially depended on before the public FastVideo API can represent it.
- Private Dreamverse-only fields should be translated in a private adapter layer, not silently turned into public FastVideo compatibility promises.
- GPU-heavy validation should remain in Buildkite or nightly jobs, but core API and config semantics must be covered by regular CI.
- Public inference terminology should use `config`, not `document`.
- The canonical public inference boundary should remain plain dataclasses plus plain dict/YAML/JSON, not OmegaConf objects or runtime-specific config wrappers.

## Known Technical Debt

Identified during PR 0-5.5 review; deferred to later PRs.

### Model-specific fields in shared schema

Most model-specific SamplingParam fields were promoted to base `SamplingParam` in PR 4 (LTX2 multi-modal CFG/STG, HYWorld attention masks, GameCraft camera/action control). These are classified as `preset_owned` in the parity inventory.

`SamplingConfig` still contains fields used by only one or two model families:
- `height_sr`, `width_sr`, `num_inference_steps_sr` (Hunyuan15 SR) — target PR 10
- `guidance_scale_2`, `boundary_ratio` (Wan2.2, LingBotWorld)

`InputConfig` still contains model-specific conditioning fields:
- `mouse_cond`, `keyboard_cond`, `grid_sizes` (MatrixGame)
- `c2ws_plucker_emb` (LingBotWorld)
- `refine_from`, `stage1_video` (LongCat) — target PR 9

Target: migrate to preset-owned typed override classes or request extensions.

### LTX2-specific defaults leaked onto shared `SamplingParam` (CFG-force fixed in PR 6; remaining fields still leak)

**Fixed in PR 6**: `SamplingParam.ltx2_cfg_scale_video` /
`ltx2_cfg_scale_audio` class defaults were `3.0` / `7.0`, which tripped
`ForwardBatch.__post_init__`'s
`ltx2_text_cfg_enabled = (ltx2_cfg_scale_video != 1.0 or ltx2_cfg_scale_audio != 1.0)`
check → force-enabled `do_classifier_free_guidance` for every model
family whose preset didn't explicitly reset them. Surfaced as the
TurboDiffusion I2V SSIM crash
(`text_encoding.py:81` asserting `isinstance(batch.negative_prompt, str)`).
Defaults are now `1.0` (CFG off); `ltx2_base` preset already sets
`3.0`/`7.0` explicitly so its behavior is unchanged. Regression guard
at `test_presets.py::TestPresetDefaultTypes::test_ltx2_cfg_defaults_are_off`.

**Still leaking (not fixed)**: `SamplingParam` still carries the other
LTX-2 multi-modal knobs — `ltx2_modality_scale_video`/`_audio`,
`ltx2_rescale_scale`, `ltx2_stg_scale_video`/`_audio`,
`ltx2_stg_blocks_video`/`_audio` — with LTX-2-on class defaults
(`3.0`/`3.0`/`0.7`/`1.0`/`1.0`/`[29]`/`[29]`). These only affect LTX-2
denoising stages today (no cross-family code path reads them), so
there's no active bug, but they are still "model-specific fields in
shared schema" tech debt.

**Consequence for existing LTX-2 reference videos**: `ltx2_distilled`
and `ltx2_two_stage` presets don't explicitly set
`ltx2_cfg_scale_video`/`_audio`, so they previously inherited
`3.0`/`7.0` (CFG accidentally on). After the default-change they
inherit `1.0`/`1.0` (CFG off). This matches the design intent for
few-step distilled variants but may shift SSIM output. Re-seed
`ltx2_distilled` + `ltx2_two_stage` references if the existing refs
fail threshold; alternatively, explicitly set the CFG fields on those
presets to preserve old behavior.

**Remaining cleanup**: subsumed by the "Model-specific fields in
shared schema" entry above — migrate the still-leaking LTX-2 knobs
(`ltx2_modality_scale_*`, `ltx2_rescale_scale`, `ltx2_stg_*`) to a
typed `LTX2SamplingOverride` during the per-model migration sweep.

### Escape hatch usage

The compat layer (`compat.py:134`) routes unrecognized legacy kwargs into `generator.pipeline.experimental`, which `generator_config_to_fastvideo_args` dumps back into flat kwargs. This untyped pass-through should shrink as presets absorb model-specific fields; each model migration PR should review and reduce it.

### Quantization code paths need a careful audit

Before any public-API freeze, quantization routing needs end-to-end verification — this was not finished in PR 6 and several overlapping surfaces coexist:

- `QuantizationConfig.text_encoder_quant` / `.transformer_quant` are string fields on `EngineConfig.quantization` (`api/schema.py`).
- `FastVideoArgs.override_text_encoder_quant` is a legacy flat kwarg that compat.py routes to the typed field above.
- Internal's `ui/ltx2-streaming/server/gpu_pool.py:220,231` skips those typed paths entirely and does `pipeline_config.dit_config.quant_config = FP4Config()` — a direct object mutation on the PipelineConfig instance before calling `VideoGenerator.from_pretrained`.
- `fastvideo/layers/quantization/fp4_config.py` is where the actual `FP4Config()` lives; the wiring from the string (`"fp4"`) to that class is not currently covered by any API-layer test.

Open questions to resolve before freeze:
- Does `transformer_quant="fp4"` construct an `FP4Config()` end-to-end, or is the string path an unused vestige?
- Does `text_encoder_quant` have the same string→class wiring, or different semantics?
- When the realtime runtime (PR 7.6) drops the in-place `dit_config.quant_config = FP4Config()` mutation, which typed path replaces it?
- Should DiT, text-encoder, VAE, and image-encoder quant each be a separate typed field (current `QuantizationConfig` only has DiT + text-encoder)?

Action: audit `fastvideo/layers/quantization/` ↔ `api/schema.QuantizationConfig` ↔ `FastVideoArgs` ↔ `PipelineConfig.*.quant_config` wiring in one sweep. Add an equivalence test (`"fp4"` string → same runtime behavior as `FP4Config()` instance) before PR 7.6 lands, or document a deliberate decision to keep the object-mutation path.

### Scattered model-family config layout

Each family's config is still spread across:
- `fastvideo/configs/pipelines/<family>.py` — PipelineConfig subclasses (DiT, VAE, precision, flow_shift)
- `fastvideo/api/sampling_param.py` — base SamplingParam with promoted model-specific fields
- `fastvideo/pipelines/basic/<family>/` — pipeline + presets
- `fastvideo/pipelines/stages/` — model-specific stages mixed with shared stages

PR 4 deleted `fastvideo/configs/sample/` entirely and moved defaults to `pipelines/basic/<family>/presets.py`. Remaining colocation targets: `configs/pipelines/<family>.py` and model-specific stages in `pipelines/stages/`. See "Pipeline Package Structure" below.

### Minor duplication

`_serialize_config` in `request_metadata.py` duplicates `config_to_dict` in `parser.py`. Consolidate when convenient.

## Config Layer Plan

The repo should end up with fewer inference-facing config abstractions. Layers with distinct jobs:

1. **Public Inference Config** — `fastvideo.api`: `GeneratorConfig`, `GenerationRequest`, `RunConfig`, `ServeConfig`, `ContinuationState`, preset selection, stage overrides. Stable API for Python, CLI, YAML/JSON, and OpenAI/server request normalization. Strict, typed, validated.

2. **Legacy Inference Compatibility Config** — transitional internal layer: `FastVideoArgs`, `SamplingParam` (now at `fastvideo/api/sampling_param.py` with preset-based defaults), legacy inference-facing `PipelineConfig` usage. Shrinks as the runtime migrates to use the typed API directly.

3. **Pipeline Preset / Default Config** — model-owned presets at `fastvideo/pipelines/basic/<family>/presets.py`. Engine/architecture config from `configs/pipelines/<family>.py` should be **colocated** with pipeline implementations (`pipelines/basic/<family>/pipeline_configs.py`) but not absorbed by presets — those classes carry internal component wiring (DiT, VAE, precision) that is not user-facing.

4. **Model Architecture / Component Config** — `fastvideo/configs/models/` stays separate (architecture and component loading, not user request normalization).

5. **Runtime Execution State** — `ForwardBatch` in `pipeline_batch_info.py` is internal execution state, not a public config object.

6. **Training Config** — stays separate from inference config. At most shares dotted override parsing/merging, YAML/JSON loading helpers, and error-path formatting patterns.

### Naming Guidance

- keep public inference schema names namespaced under `fastvideo.api`
- avoid exporting them from top-level `fastvideo/__init__.py` until migration is farther along
- envelope types `RunConfig` / `ServeConfig` get sufficient disambiguation from training config via the namespace
- consider renaming the public engine quantization type to `EngineQuantizationConfig` if a collision arises (deferred)

### Inference Package Structure

Current single-file [schema.py](/home/william5lin/FastVideo/fastvideo/api/schema.py) is acceptable but not the final shape. Later cleanup should split by concern (config/model classes, envelope types, parser/loading, compatibility translation).

## Pipeline Package Structure

The `fastvideo/pipelines/` package should converge on a layout where each model family directory is self-contained: pipeline implementation, presets, engine/arch config, and model-specific stages all in one place.

### Current layout

```
fastvideo/
├── configs/
│   ├── pipelines/          # flat: wan.py, ltx2.py, hunyuan15.py, ...
│   │                       # (PipelineConfig subclasses — engine/arch config)
│   └── models/             # hierarchical: dits/, vaes/, encoders/
│                           # (architecture definitions — stays here)
│                           # note: configs/sample/ was removed entirely in PR 4
├── pipelines/
│   ├── basic/
│   │   ├── wan/            # pipeline classes + presets.py
│   │   ├── ltx2/           # pipeline classes + presets.py
│   │   ├── hunyuan15/      # pipeline classes + presets.py
│   │   └── ...             # all families have presets.py (PR 4)
│   └── stages/             # flat: 14 shared + 19 model-prefixed files
└── api/                    # public inference schema + preset registry +
                            # sampling_param.py (canonical SamplingParam home)
```

### Target layout

```
fastvideo/
├── configs/
│   ├── models/             # stays: architecture definitions (dits/, vaes/, ...)
│   └── pipelines/
│       └── base.py         # stays: PipelineConfig base class
├── pipelines/
│   ├── basic/
│   │   ├── wan/
│   │   │   ├── wan_pipeline.py
│   │   │   ├── ...
│   │   │   ├── presets.py          # user-facing presets (done in PR 4)
│   │   │   └── pipeline_configs.py  # from configs/pipelines/wan.py
│   │   ├── ltx2/
│   │   │   ├── ltx2_pipeline.py
│   │   │   ├── presets.py
│   │   │   ├── pipeline_configs.py  # from configs/pipelines/ltx2.py
│   │   │   └── stages/             # from pipelines/stages/ltx2_*.py
│   │   ├── longcat/
│   │   │   ├── longcat_pipeline.py
│   │   │   ├── presets.py
│   │   │   ├── pipeline_configs.py
│   │   │   └── stages/             # from pipelines/stages/longcat_*.py (9 files)
│   │   └── ...
│   └── stages/             # shared-only stages remain here
│       ├── base.py, denoising.py, encoding.py, decoding.py, text_encoding.py, ...
└── api/                    # unchanged
```

### What moves and when

| Source | Destination | Status |
|--------|------------|--------|
| `configs/sample/<family>.py` | Absorbed by `presets.py` defaults | **Done** (PR 4) — directory deleted entirely |
| `configs/pipelines/<family>.py` | `pipelines/basic/<family>/pipeline_configs.py` | Pending — per-model colocation PR |
| `pipelines/stages/<family>_*.py` | `pipelines/basic/<family>/stages/` | Pending — optional, per-model PR |

### What stays

- **`configs/pipelines/base.py`** — `PipelineConfig` base class, shared CLI arg generation, JSON serialization.
- **`configs/models/`** — Model architecture definitions (DiT, VAE, encoder).
- **`pipelines/stages/` shared stages** — `denoising.py`, `encoding.py`, `decoding.py`, `text_encoding.py`, `timestep_preparation.py`, etc.
- **`pipelines/stages/` model-specific stages** — moving into family directories is optional and should only happen when the family has enough model-specific stages to justify it (LongCat: yes; Wan: no).

### Cross-family inheritance

Some families extend another family's PipelineConfig (e.g., `LingBotWorld` extends `Wan2_2_I2V_A14B_Config`; `TurboDiffusion` reuses `t5_postprocess_text` from Wan). After relocation these become cross-family imports; shared utilities like `t5_postprocess_text` should be extracted to a common location to avoid coupling unrelated families.

### Migration approach

Config colocation is not a separate PR. Each per-model colocation PR (PRs 6, 9, 10) includes the colocation step for that family:
1. `presets.py` already exists (done in PR 4)
2. Move `configs/pipelines/<family>.py` → `pipeline_configs.py` in family dir
3. `configs/sample/` already removed entirely (done in PR 4)
4. Update `registry.py` imports
5. Update any cross-family imports

## CI / Testing Baseline

- GitHub Actions runs pre-commit only in [.github/workflows/ci-precommit.yml](/home/william5lin/FastVideo/.github/workflows/ci-precommit.yml)
- Aggregate/full-suite orchestration via Buildkite status flows ([ci-aggregate-status.yml](/home/william5lin/FastVideo/.github/workflows/ci-aggregate-status.yml), [ci-trigger-full-suite.yml](/home/william5lin/FastVideo/.github/workflows/ci-trigger-full-suite.yml))
- Useful test anchors:
  - [test_video_generator.py](/home/william5lin/FastVideo/fastvideo/tests/entrypoints/test_video_generator.py)
  - [test_openai_api.py](/home/william5lin/FastVideo/fastvideo/tests/entrypoints/test_openai_api.py) / [test_openai_api_integration.py](/home/william5lin/FastVideo/fastvideo/tests/entrypoints/test_openai_api_integration.py) (real-server GPU integration belongs in Buildkite/nightly)
  - [test_sp_ltx2.py](/home/william5lin/FastVideo/fastvideo/tests/distributed/test_sp_ltx2.py)
  - [test_ltx2_pipeline_smoke.py](/home/william5lin/FastVideo/tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py)
  - [test_ltx2_registry.py](/home/william5lin/FastVideo/tests/local_tests/test_ltx2_registry.py)

## New CI Coverage To Add

### GitHub Actions

A workflow such as `.github/workflows/ci-inference-api-schema.yml` should run on every PR:

```bash
pytest -q \
  fastvideo/tests/api \
  fastvideo/tests/entrypoints/test_video_generator.py \
  fastvideo/tests/entrypoints/test_openai_api.py
```

`fastvideo/tests/api/` already covers schema parsing, YAML/JSON loading, dotted overrides, strict unknown-key validation, legacy flat-flag translation, and preset selection / stage override validation.

### Buildkite Fastcheck

GPU fastcheck jobs for LTX2 preset smoke, preset-to-pipeline dispatch, and one old-vs-new compatibility equivalence check:

```bash
pytest -q fastvideo/tests/distributed/test_sp_ltx2.py
pytest -q tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py
pytest -q tests/local_tests/test_ltx2_registry.py
```

### Buildkite Full Suite

- multi-stage preset smoke on LongCat and Hunyuan15
- server default-request merge behavior
- backward compatibility path for legacy CLI and Python kwargs
- real-server integration via [test_openai_api_integration.py](/home/william5lin/FastVideo/fastvideo/tests/entrypoints/test_openai_api_integration.py)

### Nightly

- LTX2 continuation/state roundtrip
- Hunyuan15 SR preset smoke
- LongCat distill/refine preset smoke

## PR Sequence

### PRs 0-7 (landed)

See "PR Landing History", "Landed Artifacts", and "Notable Decisions Carried Forward" above. Scope, tests, and merge criteria for the landed PRs are reflected in the linked files; the test suites at `fastvideo/tests/api/` + `fastvideo/tests/entrypoints/` cover the merge criteria. The detailed per-PR sections for PR 6 and PR 7 below are kept as historical context.

## PR 6: LTX2 Public Preset and Asset Wiring

### Scope

Upstream the public LTX2 two-stage preset shape from Dreamverse, but only for repo-visible/public FastVideo surfaces. Also colocate LTX2 config alongside its pipeline implementation.

Note: `ltx2_base` and `ltx2_distilled` presets already exist (`fastvideo/pipelines/basic/ltx2/presets.py`) and the old `configs/sample/ltx2.py` is gone along with the rest of `configs/sample/` (PR 4). This PR adds the two-stage preset and colocates PipelineConfig.

### Main Changes

- add `ltx2_two_stage` preset (two-stage refine)
- add LTX2 stage override types
- map repo-visible LTX2 init-time knobs into: preset selection, component config, preset overrides
- **typed replacements for internal `gpu_pool.py` flat kwargs** (load_kwargs at [gpu_pool.py:233-260](/home/william5lin/FastVideo-internal/ui/ltx2-streaming/server/gpu_pool.py)):
  - `ltx2_refine_enabled`, `ltx2_refine_upsampler_path`, `ltx2_refine_lora_path`, `ltx2_refine_add_noise` → `generator.pipeline.preset_overrides.refine.*` (init-time stage config)
  - `ltx2_refine_num_inference_steps`, `ltx2_refine_guidance_scale` → `request.stage_overrides.refine.*` (per-request)
  - `ltx2_vae_tiling` → typed field on `PipelineSelection` or similar
  - `torch_compile_kwargs` (`backend`, `fullgraph`, `mode`, `dynamic`) → first-class fields on `CompileConfig`, with `extras: dict[str, Any]` for uncommon kwargs
  - `FP4Config()` integration → typed quantization field path
- do not promise public compatibility for private Dreamverse-only aliases; those belong in the private adapter layer
- **config colocation**: move `configs/pipelines/ltx2.py` → `pipelines/basic/ltx2/pipeline_configs.py`; update `registry.py` imports
- optionally move `pipelines/stages/ltx2_*.py` (4 files) into `pipelines/basic/ltx2/stages/`

### Why the gpu_pool.py typed-replacement scope matters

PR 7.6 upstreams `gpu_pool.py` into the public repo. Without this PR 6 expansion, the public server perpetuates the flat-kwarg surface the refactor is eliminating. Every flat kwarg used by `gpu_pool.py` must have a typed home in `GeneratorConfig` / preset overrides / request stage overrides before PR 7.6 can land cleanly.

Same scoping rule applies to Dynamo. Draft PR ai-dynamo/dynamo#7544 had to reason about FastVideo-internal flat kwarg names to construct `VideoGenerator`; after this PR, Dynamo's `FastVideoArgGroup` can build a typed `GeneratorConfig` directly from its CLI with no awareness of legacy LTX2 names. Hard gate for PR 7.10.

### Commits

1. `feat(ltx2): add public ltx2 pipeline presets`
2. `feat(ltx2): add typed refine stage overrides for public ltx2 surfaces`
3. `feat(api): promote common torch.compile kwargs and ltx2 vae tiling to typed fields`
4. `refactor(ltx2): colocate pipeline config and stages with implementation`
5. `test(ltx2): cover preset config normalization, gpu_pool-style kwarg translation, compatibility`

### Required Tests

- new `fastvideo/tests/api/test_ltx2_presets.py`
- expand [test_ltx2_registry.py](/home/william5lin/FastVideo/tests/local_tests/test_ltx2_registry.py)
- expand [test_ltx2_pipeline_smoke.py](/home/william5lin/FastVideo/tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py)

### Merge Criteria

- public/repo-visible LTX2 setup can be represented by public config
- no new public compatibility promise is made for private-only Dreamverse field aliases
- LTX2 preset smoke passes on GPU CI
- LTX2 config no longer split across `configs/` and `pipelines/`

## PR 7: LTX2 Continuation State (Public Payload + Server Session)

### Scope

Upstream the continuation contract as a **hybrid** public API that supports both opaque client-round-trip payloads AND server-held session state. Both surfaces share one serialization format.

Decision rationale: [streaming-server-upstream-plan.md § Design Decision 1](.agents/exploration/streaming-server-upstream-plan.md).

### Main Changes

- add typed `LTX2ContinuationState` internal dataclass with:
  - trailing conditioning frames (or tensor blob ID)
  - audio latents (or tensor blob ID)
  - segment index / rollout position
  - audio sample rate, `video_position_offset_sec`, other metadata
- map request/response state into public `ContinuationState` with `kind="ltx2.v1"` and `payload` serialized from the typed state
- stop requiring app-level raw latent/audio condition fields in the public API
- keep `extensions` only for any remaining experimental fields
- add `SessionStore` interface in `fastvideo/entrypoints/streaming/session_store.py`:
  - `snapshot(session_id) -> ContinuationState`
  - `hydrate(state: ContinuationState) -> session_id`
  - default in-memory implementation; pluggable for redis/etc. later
- specify payload serialization: tensor fields use a blob store indirection when large (bandwidth optimization for the opaque form)

### Commits

1. `feat(ltx2): add typed ltx2 continuation state and public payload schema`
2. `feat(streaming): add SessionStore with snapshot and hydrate`
3. `refactor(ltx2): replace raw continuation kwargs with state roundtrip`
4. `test(ltx2): add continuation roundtrip, session snapshot, and compatibility coverage`

### Required Tests

- new `fastvideo/tests/api/test_ltx2_state.py`
- new `fastvideo/tests/entrypoints/streaming/test_session_store.py`
- integration tests: generate returns state when requested; next request accepts returned state; invalid state/preset combinations error cleanly; server-held session auto-chains between segments; snapshot + hydrate round-trip across session boundaries
- GPU smoke on Buildkite for state roundtrip

### Merge Criteria

- continuation state is a first-class public concept
- both opaque payload and server-held session flows work end to end
- server request translation is able to carry typed state without reverting to ad hoc kwargs
- `gpu_pool.py` style per-GPU implicit chaining is expressible via SessionStore
- `ContinuationState.payload` is JSON-serializable (or uses an opaque blob-ID indirection for large tensors) so it can round-trip through a Dynamo-style RPC for future disaggregated prefill/decode — aggregated Dynamo in PR 7.10 ignores continuation, but the contract must not drift

## PR 7.5: Streaming Server Skeleton

Status: open as #1251 on `will/api_7.5`.

### Scope

Upstream the minimum viable WebSocket streaming server from `FastVideo-internal/ui/ltx2-streaming/server/main.py`. Single generator (no GPU pool yet), typed `GenerationRequest` as JSON messages, fMP4 output. Prompt enhancer and pool come in later PRs.

### Main Changes

- `fastvideo/entrypoints/streaming/server.py`: FastAPI + WebSocket entry
- `fastvideo/entrypoints/streaming/session.py`: session lifecycle, state machine
- `fastvideo/entrypoints/streaming/protocol.py`: JSON WebSocket message schemas (`session_init_v2`, `segment_prompt_source`, `ltx2_segment_start`, `step_complete`, `media_init`, `media_segment_complete`, etc.)
- `fastvideo/entrypoints/streaming/stream.py`: fMP4 encoding via ffmpeg
- `fastvideo/entrypoints/streaming/session_init_image.py`: i2v init image
- Replace the `NotImplementedError` stub from PR 5.5 with the live server
- Wire into `fastvideo streaming-serve` CLI from PR 5.5
- `docs/design/server_contracts/streaming.md`: WebSocket protocol contract

### Commits (as shipped)

1. `feat(streaming): protocol schemas + session state machine`
2. `feat(streaming): fMP4 encoder + session init-image persistence`
3. `feat(streaming): single-generator WebSocket server entry`
4. `test(streaming): server lifecycle + protocol + fMP4 coverage`
5. `docs(streaming): server contract spec`
6. `fix(streaming): restore missing-streaming-block guard + retire stub-era test`
7. `simplify(streaming): review follow-ups (idle timeout via asyncio.wait_for, _send_error helper, _cleanup_session, Protocol-typed generator, cleanup-on-disconnect)`
8. `fix(streaming): enforce idle timeout on receive_json + flag generator-cancellation gap (TODO → PR 7.10)`

### Required Tests

- `fastvideo/tests/entrypoints/streaming/test_protocol.py`
- `fastvideo/tests/entrypoints/streaming/test_session.py`
- `fastvideo/tests/entrypoints/streaming/test_session_init_image.py`
- `fastvideo/tests/entrypoints/streaming/test_stream_encoder.py` (fMP4 encode)
- `fastvideo/tests/entrypoints/streaming/test_server.py` (in-process WebSocket lifecycle via `starlette.testclient`)

### Merge Criteria

- Single-user WebSocket session end-to-end: connect, init, submit prompts, receive fMP4 chunks, clean shutdown ✓
- All JSON messages validated against typed protocol schemas ✓

### Deferred to later PRs

- Per-step progress events (only a terminal `step_complete` is emitted today; per-step events need `generate_async` from PR 7.10).
- Mid-segment cancellation on client disconnect (TODO marker in `server.py` near `pool.run`; needs PR 7.10).

## PR 7.6: GPU Pool Upstream

Status: branch `will/api_7.6` rebased on `will/api_7.5`, not yet PR'd.

### Scope

Upstream `FastVideo-internal/ui/ltx2-streaming/server/gpu_pool.py` with a typed configuration boundary. One subprocess per GPU, one `VideoGenerator` per subprocess, job queue, session-to-GPU binding.

### Main Changes (as shipped on `will/api_7.6`)

- `fastvideo/entrypoints/streaming/gpu_pool.py`: `GpuPool` (ABC) + `InProcessGpuPool` + `SubprocessGpuPool` + `PoolAssignment` / `PoolHealth` / `PoolAcquireTimeout`
- `fastvideo/entrypoints/streaming/worker.py`: per-GPU worker entry (`worker_main`) and two-segment warmup helper
- subprocess startup uses typed `GeneratorConfig`, not flat kwargs
- session-to-GPU binding with timeout + queue for contention
- two-segment startup warmup per worker (segment 1 fresh + segment 2 with returned `ContinuationState` so both compile branches are primed)
- `SessionStore` (from PR 7) wired into the pool for per-GPU continuation cache
- `server.py` rerouted to `pool.acquire` / `pool.run` / `pool.release`; `build_app` accepts either a generator (wrapped in `InProcessGpuPool`) or a pre-built `pool=`

### Commits (as shipped)

1. `feat [7.6/n]: GPU pool manager with typed worker boundary`
2. `refactor [7.6/n]: route streaming server through GpuPool`
3. `test [7.6/n]: GPU pool coverage (in-process + subprocess)`
4. `fix(streaming): restore missing asyncio import in server` (rebase fixup)
5. `feat(streaming): extract worker.py and add two-segment warmup`

### Required Tests

- `fastvideo/tests/entrypoints/streaming/test_gpu_pool.py` (in-process + subprocess via thread-backed factory) ✓
- `fastvideo/tests/entrypoints/streaming/test_worker.py` (two-segment warmup dispatch + result-shape extractor) ✓
- GPU smoke on Buildkite for real multi-GPU pool — deferred (separate infra ask)

### Merge Criteria

- Single-user multi-segment WebSocket session via the pool path ✓ (covered by `test_server.py`'s `_MockGenerator` on `InProcessGpuPool`)
- No flat kwargs in the pool/worker construction path ✓ (worker_main takes `GeneratorConfig`)
- Session handoff across GPUs (snapshot + hydrate elsewhere) — `SessionStore` wired but no explicit handoff test yet
- Multi-user, multi-GPU WebSocket server end-to-end on real GPUs — deferred (Buildkite GPU smoke)

### Deferred to later PRs (with rationale)

- **Audio re-encode (`LTX2AudioEncoder`, `AudioProcessor`)** — internal `_re_encode_audio` runs *inside* the per-step streaming loop (`_stream_av_fmp4_events` / `do_step_ltx2`) to convert waveform → latents for continuation conditioning. The public LTX2 pipeline already populates `LTX2ContinuationState.audio_latents` via `ltx2_audio_decoding.py`, so the whole-segment `pool.run()` path doesn't need a re-encode shim. Re-encode integration is therefore a **per-step streaming concern that belongs with PR 7.10**'s `generate_async`.
- **Deprecate `VideoGenerator.from_pretrained(**flat_kwargs)`** — worker uses typed config, but no public deprecation warning yet. Belongs with PR 13 cleanup.
- **Multi-GPU Buildkite smoke** — separate infra ask; not blocking the upstream PR.

## PR 7.7: Prompt Enhancer Upstream

### Scope

Upstream `FastVideo-internal/ui/ltx2-streaming/server/prompt_enhancer.py` behind an `LLMProvider` abstraction. Ship built-in providers for cerebras and groq; leave the door open for users to add their own.

Decision rationale: [streaming-server-upstream-plan.md § Design Decision 3](.agents/exploration/streaming-server-upstream-plan.md).

### Main Changes

- `fastvideo/entrypoints/streaming/prompt/enhancer.py`: provider-agnostic prompt operations (enhance, auto-extend, rewrite)
- `fastvideo/entrypoints/streaming/prompt/providers/base.py`: `LLMProvider` protocol + `LLMRequest` / `LLMResponse` dataclasses
- `fastvideo/entrypoints/streaming/prompt/providers/cerebras.py`, `groq.py`: built-in implementations
- hot-reloadable system prompts via management endpoint
- fallback retry across providers in priority order
- wire `PromptEnhancerConfig` from PR 5.5 into server startup

### Commits

1. `feat(streaming): add LLMProvider protocol and cerebras providers`
2. `feat(streaming): add groq provider`
3. `feat(streaming): add provider-agnostic prompt enhancer orchestration`
4. `feat(streaming): add hot-reloadable system prompt management endpoint`
5. `test(streaming): cover enhancer operations and provider fallback`

### Required Tests

- `fastvideo/tests/entrypoints/streaming/prompt/test_providers.py`
- `fastvideo/tests/entrypoints/streaming/prompt/test_enhancer.py`
- mocked provider tests for fallback behavior

### Merge Criteria

- Enhancer works end-to-end with the WebSocket server from PR 7.5
- Third-party users can register their own `LLMProvider`
- Hot-reload of system prompts survives config changes

## PR 7.8: Streaming Auxiliaries

### Scope

Upstream the small auxiliary modules: prompt safety, rewrite payload builder, session logger, mock server.

### Main Changes

- `fastvideo/entrypoints/streaming/prompt/safety.py`: optional fasttext classifier; shipped as optional extra `pip install fastvideo[prompt-safety]`
- `fastvideo/entrypoints/streaming/prompt/rewrite.py`: rewrite payload builder (from internal `rewrite_prompt_payload.py`)
- `fastvideo/entrypoints/streaming/session_logger.py`: session JSONL logs
- `fastvideo/entrypoints/streaming/mock_server.py`: mock backend for dev/tests
- `pyproject.toml` optional-dependency entry for `prompt-safety`

### Commits

1. `feat(streaming): add optional fasttext prompt safety`
2. `feat(streaming): add rewrite payload builder and session logger`
3. `feat(streaming): add mock server for frontend development`
4. `test(streaming): cover safety, rewrite, mock server`

### Required Tests

- `fastvideo/tests/entrypoints/streaming/test_safety.py` (with and without fasttext installed)
- `fastvideo/tests/entrypoints/streaming/test_rewrite.py`
- `fastvideo/tests/entrypoints/streaming/test_mock_server.py`

### Merge Criteria

- Optional dependency gating works: safety silently disables when fasttext is not installed
- Mock server can replace the real server for frontend dev

## PR 7.9: Router Upstream

### Scope

Upstream `FastVideo-internal/ui/ltx2-streaming/router/main.py` — a multi-replica load balancer with health checks and WebSocket proxying.

**Open decision**: in-repo subpackage vs. separate package. Default assumption: in-repo at `fastvideo/entrypoints/streaming/router/`. If review concludes router is too orthogonal to inference, split into `fastvideo-router/` or `fastvideo/contrib/router/`.

### Main Changes

- `fastvideo/entrypoints/streaming/router/main.py`: router entry
- `fastvideo/entrypoints/streaming/router/registry.py`: replica registry (from internal `ReplicaRegistry` class)
- health check loop, failure threshold, primary/secondary semantics
- `fastvideo router-serve --config router.yaml` CLI subcommand
- typed `RouterConfig` in `fastvideo/api/schema.py`

### Commits

1. `feat(router): add typed RouterConfig and replica registry`
2. `feat(router): add health check loop and failure handling`
3. `feat(router): add websocket proxy and http passthrough`
4. `feat(cli): add router-serve subcommand`
5. `test(router): cover registry, health, and proxy behavior`

### Required Tests

- `fastvideo/tests/entrypoints/streaming/router/test_registry.py`
- `fastvideo/tests/entrypoints/streaming/router/test_main.py`
- integration test with two mock backends

### Merge Criteria

- Multi-backend deployment works end-to-end
- Health checks flip replicas in and out cleanly
- WebSocket sessions routed to healthy primary, fall back on failure

## PR 7.10: Dynamo Backend Contract

### Scope

Make FastVideo consumable as a native Dynamo backend (same tier as vllm, sglang, trtllm). **FastVideo does not host any Dynamo code** — the backend package (args.py, main.py, backend.py, register.py, health_check.py, adapter) lives entirely in the Dynamo repo at `components/src/dynamo/fastvideo/`, modeled on draft PR ai-dynamo/dynamo#7544.

This PR lands only the FastVideo-side **contract**: the public Python API that Dynamo's backend package imports. Zero new dependencies, zero `contrib/dynamo/`, zero Dynamo-specific CLI inside FastVideo.

Decision rationale: [streaming-server-upstream-plan.md § Design Decision 4](.agents/exploration/streaming-server-upstream-plan.md).

### Main Changes

- add `VideoGenerator.generate_async(request: GenerationRequest) -> AsyncGenerator[VideoEvent, None]`:
  - `VideoProgressEvent(step, total_steps, stage)`
  - `VideoPartialEvent(frames_ndarray, index)` (optional; emitted in the streaming path only)
  - `VideoFinalEvent(video_bytes_or_tensor, metadata, continuation_state: ContinuationState | None)`
- refactor `VideoGenerator.generate_video(request=...)` to run through `asyncio.run(generate_async(...))` internally, collecting events and returning the final result; preserve the public sync signature
- add `VideoGenerator.default_health_check_request() -> GenerationRequest` (256x256, 8 frames, 1 step) so a Dynamo-side health payload can be derived purely from the public API
- add typed `VideoEvent` hierarchy and `VideoResult` exports under `fastvideo.api`
- stabilize public re-exports: `from fastvideo import VideoGenerator`; `from fastvideo.api import GenerationRequest, SamplingConfig, ContinuationState, VideoResult, VideoEvent, VideoProgressEvent, VideoPartialEvent, VideoFinalEvent`
- PR 7.5's streaming server is rewired to consume `generate_async` directly (was a TODO in PR 7.5)
- audio re-encode integration (`LTX2AudioEncoder`, `AudioProcessor`) for per-step audio continuity inside the streaming fMP4 pipeline (deferred from PR 7.6 — the whole-segment `pool.run()` path doesn't need it; the per-step path does)
- mid-segment cancellation: client disconnect propagates an `asyncio.CancelledError` through `generate_async` so the GPU work stops instead of running to completion (the gap PR 7.5/7.6 flagged with TODO markers around `pool.run`)
- docs PR (PR 8) carries a reference mapping sketch the Dynamo repo's backend package can use as a template — but no code in FastVideo knows about Dynamo

### Why the async API matters for more than Dynamo

The same `generate_async` surface that Dynamo consumes also powers:

- **Streaming server fMP4 pipeline** (PR 7.5) — decoder pulls `VideoPartialEvent` frames directly, no intermediate disk spill
- **OpenAI stateless server** (PR 5) — can opt into progress events for future Server-Sent-Events extension without a second code path
- **Future disaggregation in Dynamo** — prefill/decode split can carry `ContinuationState` across workers using the same event shape

Building `generate_async` once prevents three near-duplicate progress loops from growing in the three adapters.

### Commits

1. `feat(api): add VideoEvent hierarchy and VideoResult public exports`
2. `feat(entrypoints): add VideoGenerator.generate_async event stream`
3. `refactor(entrypoints): reroute generate_video through generate_async internally`
4. `feat(entrypoints): add default_health_check_request helper`
5. `test(entrypoints): cover generate_async event stream and sync wrapper parity`
6. `test(entrypoints): contract test that mock dynamo-style handler wraps public api`

### Required Tests

- `fastvideo/tests/entrypoints/test_generate_async.py`:
  - event ordering (Progress* → Final)
  - sync `generate_video` returns the same result as collecting events from `generate_async`
  - cancellation mid-generation cleans up GPU state
- contract test that imports only the public surface (`from fastvideo import VideoGenerator`, `from fastvideo.api import ...`) and wraps it in a mock Dynamo-style handler signature (`async def generate(req: dict, ctx) -> AsyncGenerator[dict, None]`), verifying the wrap is possible without any FastVideo-internal imports
- health-check helper returns a valid `GenerationRequest` that passes `parse_config`

### Merge Criteria

- `generate_async` is the canonical execution API; `generate_video` remains for backward-compat but is a thin sync wrapper
- FastVideo's public surface alone is sufficient to construct a Dynamo backend — no flat LTX2 legacy kwargs, no FastVideo-internal imports required
- migration docs (PR 8) include the Dynamo integration skeleton so ai-dynamo/dynamo#7544 or its successor can be reopened and landed with a straightforward handler/adapter written in the Dynamo repo
- streaming server (PR 7.5) consumes `generate_async` directly, no wrapper duplication
- no Dynamo-related file, dep, or import path lives in FastVideo

## PR 8: Internal-UI ↔ Public-Server Contract + Dynamo Integration Reference

### Scope

Document and test the contract between the public FastVideo server stack (OpenAI HTTP, streaming WebSocket) and both of its major external consumers:

1. The private Dreamverse / internal-UI server stack that previously lived in `FastVideo-internal/ui/ltx2-streaming/`.
2. The Dynamo backend package at `ai-dynamo/dynamo/components/src/dynamo/fastvideo/`.

This PR does not ship runtime code beyond tests and docs — it locks down the surface so drift between FastVideo, the internal UI, and Dynamo can be caught at review time.

### Main Changes

- add WebSocket protocol reference doc (message schemas, state machine, error codes) for the streaming server from PR 7.5-7.9
- add OpenAI HTTP contract reference for the stateless server from PR 5
- add Dynamo integration reference:
  - end-to-end example showing how `components/src/dynamo/fastvideo/` should be laid out (args.py, main.py, backend.py, register.py, health_check.py) using the public FastVideo API from PR 7.10
  - contract tests that mimic Dynamo's `serve_endpoint(handler.generate, ...)` wrapping and verify FastVideo's `VideoGenerator` works inside it
- migration examples for Dreamverse-style inputs normalized through the typed public API without private-only compat kwargs
- keep translation from private-only fields in the private adapter rather than broadening the public FastVideo compatibility surface

### Commits

1. `docs(server): add websocket streaming protocol reference`
2. `docs(server): add openai stateless contract reference`
3. `docs(dynamo): add integration reference for native backend package`
4. `test(contract): add dreamverse-style and dynamo-style contract fixtures`

### Required Tests

- public-side tests only
- verify Dreamverse-style inputs can be normalized without private-only APIs
- mock Dynamo handler wraps `VideoGenerator.generate_async` without pulling internal modules — regression guard for PR 7.10

### Merge Criteria

- clear migration path exists for private server consumers
- clear reference path exists for Dynamo integrators (the next iteration of ai-dynamo/dynamo#7544 can reopen against this doc set)
- no new ad hoc public kwargs are added

## PR 9: LongCat Preset Migration

> May be combined with PR 10 if preset infrastructure from PR 4 is mature.

### Scope

Move LongCat's two-stage user flow behind a named preset and typed refine override surface. Colocate LongCat config — this family has the strongest case for colocation given its 9 model-specific stage files scattered in the shared `pipelines/stages/` directory.

Note: LongCat presets (`longcat_t2v`, `longcat_i2v`, `longcat_vc`) already exist and are wired via `default_preset` in registry (PR 4). LongCat never had SamplingParam subclasses. This PR adds the stage override types and colocates PipelineConfig + stages.

### Main Changes

- add `longcat_distill_refine` multi-stage preset
- add `LongCatRefineStageOverride`
- map current `refine_from`, `stage1_video`, `t_thresh`, `spatial_refine_only`, `num_cond_frames`
- update examples to use the preset path
- **config colocation**: move `configs/pipelines/longcat.py` → `pipelines/basic/longcat/pipeline_configs.py`; update `registry.py` imports
- **stage colocation**: move `pipelines/stages/longcat_*.py` (9 files) → `pipelines/basic/longcat/stages/`; update imports from pipeline classes

### Commits

1. `feat(longcat): add distill refine preset and override types`
2. `refactor(longcat): colocate pipeline config and stages with implementation`
3. `refactor(longcat): migrate examples to typed preset based api`
4. `test(longcat): add preset and compatibility coverage`

### Required Tests

- API/unit tests for LongCat preset parsing
- smoke coverage for preset dispatch
- if feasible, SSIM/nightly preset regression for LongCat refine path

### Merge Criteria

- LongCat public flow no longer requires hand-wired two-generator orchestration in examples
- LongCat config and model-specific stages colocated under `pipelines/basic/longcat/`

## PR 10: Hunyuan15 SR Preset Migration

> May be combined with PR 9 if preset infrastructure from PR 4 is mature.

### Scope

Represent Hunyuan15 SR as first-class presets rather than pipeline-class knowledge in examples. Colocate Hunyuan15 config alongside its pipeline implementation. This PR also serves as the proof-of-concept for migrating model-specific fields out of the shared schema (`height_sr`, `width_sr`, `num_inference_steps_sr` → SR stage overrides).

Note: Hunyuan15 presets (`hunyuan15_t2v_480p`, `hunyuan15_i2v_480p_distilled`, `hunyuan15_t2v_720p`, `hunyuan15_i2v_720p_distilled`, `hunyuan15_sr_1080p`) already exist and the old `configs/sample/hunyuan15.py` is gone along with the rest of `configs/sample/` (PR 4). This PR adds the SR stage override types and colocates PipelineConfig.

### Main Changes

- add `HunyuanSRStageOverride` typed override class
- add small stage override types where needed
- update examples and request defaults
- **config colocation**: move `configs/pipelines/hunyuan15.py` → `pipelines/basic/hunyuan15/pipeline_configs.py`; update `registry.py` imports
- **SR field migration POC**: declare `height_sr`, `width_sr`, `num_inference_steps_sr` as SR-stage-owned overrides in the preset rather than shared `SamplingConfig` fields

### Commits

1. `feat(hunyuan15): add sr presets with typed stage overrides`
2. `refactor(hunyuan15): colocate pipeline config with implementation`
3. `refactor(hunyuan15): route examples through preset selection`
4. `test(hunyuan15): add sr preset config and smoke coverage`

### Required Tests

- API parsing tests
- entrypoint tests for preset defaults
- GPU smoke or nightly SR preset checks

### Merge Criteria

- Hunyuan15 SR is discoverable and stable through named presets
- Hunyuan15 config colocated under `pipelines/basic/hunyuan15/`
- SR-specific fields validated through preset stage overrides

## PR 11: SSIM and Performance Test Migration

### Scope

Move golden-quality inference tests onto the typed/public API only after preset defaults and request-vs-pipeline override semantics have stabilized.

This PR should migrate regression suites that currently depend on legacy generation helpers such as `generate_video(..., **kwargs)` and test-only parameter bags.

### Main Changes

- migrate SSIM tests to the typed API or typed config path
- migrate performance tests to the typed API or typed config path
- make any remaining request-vs-pipeline compatibility mappings explicit before the migration
- keep output-quality baselines unchanged while swapping the invocation path

### Commits

1. `refactor(ssim): migrate golden inference tests to typed api`
2. `refactor(perf): migrate performance tests to typed api`
3. `test(regression): add old-vs-new invocation parity coverage where needed`

### Required Tests

- existing SSIM/nightly suites
- existing performance suites
- at least one parity check that runs the same canonical request through both the legacy and typed paths before the legacy path is removed

### Merge Criteria

- SSIM/performance tests no longer depend on legacy public inference helpers
- migrated tests do not need ad hoc compatibility exceptions beyond the typed schema/preset surface
- quality baselines remain stable after the invocation-path switch

## PR 12: Docs and Example Migration

> May be combined with PR 13 if scope is small.

### Scope

Move docs and examples onto the new API so the canonical surface is actually visible to users.

### Main Changes

- update inference docs
- add YAML config examples
- update basic examples to `GeneratorConfig` + `GenerationRequest`
- add migration guide from flat kwargs and flags

### Commits

1. `docs(api): add typed inference schema and yaml examples`
2. `docs(migration): map legacy kwargs and flags to new config`
3. `refactor(examples): migrate basic inference examples`

### Required Tests

- doc example smoke tests if feasible
- at minimum, keep example snippets synchronized with tested APIs

### Merge Criteria

- docs no longer present legacy kwargs as the preferred surface

## PR 13: Deprecation Warnings and Final Cleanup

> May be combined with PR 12 if scope is small.

### Scope

Make the old surface clearly transitional without removing it immediately.

### Main Changes

- add deprecation warnings for:
  - direct public `FastVideoArgs`
  - direct public `SamplingParam`
  - `generate_video(..., **kwargs)` legacy paths
  - `VideoGenerator.from_pretrained(**flat_kwargs)` (deferred from PR 7.6 — the typed `config=GeneratorConfig` path is already preferred)
  - legacy flat CLI flags where appropriate
- remove duplicated now-dead translation paths if any

### Commits

1. `chore(deprecation): warn on legacy inference api entrypoints`
2. `test(deprecation): cover warning paths and compat guarantees`

### Required Tests

- warning assertion tests
- ensure compatibility still works while warnings are emitted

### Merge Criteria

- users have a clear migration runway
- public docs point to the new surface

## Post-PR 13: Retire `fastvideo/api/compat.py`

After PR 13 puts deprecation warnings on the flat-kwarg public surface, the compat layer at `fastvideo/api/compat.py` (~370 lines) is the last remaining translation shim between the typed public API and the legacy internal types (`FastVideoArgs`, `SamplingParam`). It bridges five distinct jobs:

| Function / block | Job | Removable when |
|---|---|---|
| `legacy_from_pretrained_to_config` | forward: flat `from_pretrained(..., **kwargs)` → `GeneratorConfig` | all callers pass `config=GeneratorConfig` |
| `legacy_generate_call_to_request`, `_sampling_param_to_request_raw`, `_LEGACY_REQUEST_ALIASES` | forward: `generate_video(prompt, **kwargs)` → `GenerationRequest` | all callers pass `request=GenerationRequest` |
| `generator_config_to_fastvideo_args`, `_compile_config_to_torch_kwargs` | reverse: typed `GeneratorConfig` → legacy `FastVideoArgs` | runtime consumes `GeneratorConfig` directly |
| `request_to_sampling_param`, `explicit_request_updates` | reverse: typed `GenerationRequest` → legacy `SamplingParam`; supports the `ForwardBatch(**shallow_asdict(sampling_param), …)` spread in `entrypoints/video_generator.py` | `ForwardBatch`/stages read from `GenerationRequest` directly |
| `normalize_generator_config`, `normalize_generation_request`, `load_generator_config_from_file` | input normalization (dict / YAML / JSON → typed) | never — these move to `parser.py`, not deleted |

Can't be done in a single PR; each job has a different unlock condition. Kill order:

### PR 14: Strip forward translation

Prerequisite: flat-kwarg callers already migrated via PRs 11 (SSIM/performance tests), 12 (docs/examples), and 7.6 (realtime runtime upstream — `fastvideo/entrypoints/realtime/local_runtime.py:GPUSlot` currently calls `VideoGenerator.from_pretrained(**load_kwargs)` with the flat LTX-2 kwarg surface PR 6 has now typed).

- `VideoGenerator.from_pretrained` accepts `(model_path, config: GeneratorConfig | None = None)`; raises if called with extra `**kwargs`.
- `generate_video` accepts `(request: GenerationRequest, …)`; no legacy flat-kwarg path.
- Delete `legacy_from_pretrained_to_config`, `legacy_generate_call_to_request`, `_sampling_param_to_request_raw`, `_LEGACY_REQUEST_ALIASES`, `_LTX2_REFINE_FLAT_KEYS`, `_apply_request_field`.
- Verify with `grep -rn 'from_pretrained.*,\s*\*\*' fastvideo/ examples/ tests/` — expect zero hits outside the implementation itself.

Net: ~100 lines gone from compat.py; the file still exists for the reverse half.

### PR 15: `FastVideoArgs` becomes a view over `GeneratorConfig`

Today `FastVideoArgs` is a ~600-line god-object threaded through every stage, model loader, distributed setup, and worker. `compat.py:generator_config_to_fastvideo_args` exists solely to produce it from a typed config.

The minimum-disruption path: turn `FastVideoArgs` into a `@dataclass` that holds a `GeneratorConfig` and exposes the legacy field names as `@property` accessors backed by the nested typed fields. No more two-way conversion — one object, one source of truth.

- `generator_config_to_fastvideo_args` becomes identity / trivial wrapper; inlined at call sites where obvious.
- `FastVideoArgs.from_kwargs` (which silently filters unknown kwargs — see the text-encoder compile case in this session) either disappears or becomes equivalent to passing the GeneratorConfig through.
- All existing `fastvideo_args.dit_cpu_offload` etc. call sites keep working via the property accessors.

Alternative: sweep `grep -rn 'fastvideo_args\.' fastvideo/` (200+ hits) and rewrite every usage to read from `GeneratorConfig` directly. Cleaner end state, 3-5 PRs of grunt work; defer to opportunistic clean-as-you-go after PR 15.

Net: reverse-translation half of compat.py (~150 lines) becomes trivial or inlined.

### PR 16: `ForwardBatch` reads `GenerationRequest` by reference

Kills `request_to_sampling_param` and the `ForwardBatch(**shallow_asdict(sampling_param), …)` spread currently at `entrypoints/video_generator.py:576`.

- `ForwardBatch` gains `request: GenerationRequest` field.
- Stages that read `batch.num_inference_steps`, `batch.guidance_scale`, etc. rewrite to `batch.request.sampling.num_inference_steps` / `.guidance_scale`. Mostly mechanical; easily scripted.
- `explicit_request_updates` (used by the OpenAI server for operator-default merge) moves to `parser.py` — it's genuinely useful logic, just mis-filed in compat.
- `SamplingParam` either deleted or demoted to an external-facing convenience dataclass with no internal consumers.

Depends on PR 15 (or at least on having a typed config reachable from every stage that currently reaches `fastvideo_args`).

### PR 17: Dissolve `compat.py`

At this point the file has only the parsing helpers:

- `normalize_generator_config`, `normalize_generation_request` → move to `fastvideo/api/parser.py` next to `parse_config`.
- `load_generator_config_from_file` → move to `parser.py`.

Delete `fastvideo/api/compat.py`.

### Dependency chain

```
PR 13 (deprecation)
  ↓
PRs 11, 12, 7.6 (migrate callers)
  ↓
PR 14 (forward translation gone)      ─────  ~100 lines out of compat.py
  ↓
PR 15 (FastVideoArgs as view)         ─────  reverse-translation trivial
  ↓
PR 16 (ForwardBatch reads request)    ─────  SamplingParam demoted
  ↓
PR 17 (move normalizers, delete file)
```

PR 14 is reachable within the currently-planned sequence. PRs 15-17 touch training, distributed, and worker code in addition to the inference path; realistically 1-2 quarters of work beyond the current plan.

## Checkpoint: Post-PR 5.5 Review

Before starting PR 6, review:

- Whether PR 6 expansion (typed replacements for the flat LTX2 kwargs used by internal `gpu_pool.py`) is in reach
- Which model-specific fields remain in the shared `SamplingConfig`/`InputConfig`
- Whether the escape hatch pass-through in the compat layer has shrunk
- Whether config colocation (moving `configs/pipelines/<family>.py` into `pipelines/basic/<family>/`) can begin with PR 6, or needs a separate prep PR
- Whether PRs 9 and 10 can be combined
- Whether PRs 12 and 13 can be combined
- Whether model-specific stages in `pipelines/stages/` should move into family directories (especially LongCat with 9 model-specific stage files)
- Router: in-repo subpackage vs. separate package decision (defer to PR 7.9)
- Overall timeline and priority for the remaining PR tail

## Commit Hygiene Rules

Within each PR:

- keep structural moves and behavioral changes in separate commits
- keep test additions in separate commits where practical
- do not mix LTX2 private-upstream logic with unrelated CLI refactors
- every compatibility mapping change must have an equivalence test in the same PR

## Recommended Test Matrix By Layer

### Schema / Parser Layer

- public-field parity assertions
- YAML load
- JSON load
- dict load
- serialization roundtrip
- dotted override merge
- strict unknown-key rejection
- type error path reporting

### Python API Layer

- `from_config`
- `from_file`
- typed `generate`
- legacy kwargs translation
- batch prompt file behavior

### CLI Layer

- `generate --config`
- `serve --config`
- dotted nested overrides
- flat legacy flag compatibility
- precedence tests

### Preset Layer

- model/preset resolution
- preset version validation
- stage name validation
- stage override type validation

### Server Layer

- `ServeConfig` load
- default request merge
- per-request overrides
- stage override validation
- continuation state request/response path
- real-server health/generation integration in GPU/nightly CI

### GPU / Pipeline Layer

- LTX2 preset smoke
- LTX2 continuation roundtrip
- LongCat preset smoke
- Hunyuan15 SR preset smoke
- old-vs-new config equivalence for one or two canonical requests
- after preset/default stabilization, migrate SSIM and performance regression suites off legacy `generate_video(...)`

## Risk Management Notes

### Highest Risk Areas

- legacy kwargs translation subtly changing behavior
- server request translation bypassing validation
- LTX2 continuation semantics drifting from Dreamverse expectations
- preset defaults accidentally changing output quality

### Mitigations

- require normalized-config equivalence tests for legacy mapping
- gate server translation through typed parser only
- add LTX2 continuation roundtrip tests before private-server migration
- add smoke and nightly regression coverage for preset defaults

## Recommended Landing Order Summary

1. PR 0: schema parity inventory — **done**
2. PR 1: shared schema — **done**
3. PR 2: VideoGenerator typed path + compat — **done**
4. PR 3: CLI/YAML refactor — **done**
5. PR 4: preset registry + preset migration for all families — **done**
6. PR 5: stateless OpenAI server `default_request` merge (narrow) — **done**
7. PR 5.5: streaming server subpackage split (skeleton only, no behavior change) — **done**
8. PR 6: LTX2 public preset + config colocation + typed replacements for every flat LTX2 kwarg used by internal `gpu_pool.py` and consumed by the Dynamo backend
9. PR 7: LTX2 continuation state (hybrid: opaque payload + server session), JSON-serializable payloads
10. PR 7.5: streaming server skeleton (WebSocket + fMP4 + single generator)
11. PR 7.6: GPU pool upstream
12. PR 7.7: prompt enhancer upstream with `LLMProvider` abstraction
13. PR 7.8: streaming auxiliaries (prompt safety, session logger, mock server, rewrite)
14. PR 7.9: router upstream
15. PR 7.10: Dynamo backend **contract** — FastVideo adds `VideoGenerator.generate_async` + health-check helper; the Dynamo backend package itself lives entirely in the Dynamo repo
16. PR 8: internal-UI ↔ public-server contract docs + Dynamo integration reference
17. PR 9: LongCat preset migration + colocation
18. PR 10: Hunyuan15 SR preset migration + colocation + SR field migration POC
19. PR 11: SSIM/performance test migration
20. PR 12: docs/examples (includes streaming server + Dynamo)
21. PR 13: deprecation cleanup (includes flat LTX2 kwargs the internal gpu_pool used to consume)

This order keeps the API foundation first, lands server normalization before any private-server or Dynamo migration claim, builds one async execution substrate (PR 7.10) that serves both the streaming server and the Dynamo backend, and keeps broad multi-model cleanup only after the shared abstractions have settled.

## Post-PR 13 Cleanup Goal

The successful end state is:

- one public inference config surface under `fastvideo.api`
- one internal preset/default layer near each pipeline
- one internal execution-state layer

### Target directory structure per model family

Each family directory under `fastvideo/pipelines/basic/<family>/` should be self-contained:

```
fastvideo/pipelines/basic/<family>/
├── <family>_pipeline.py        # pipeline implementation(s)
├── presets.py                 # user-facing presets (defaults + stage topology)
├── pipeline_configs.py         # engine/arch config (from configs/pipelines/)
└── stages/                     # model-specific stages (optional, if >2 files)
```

### What should be gone

- `configs/pipelines/<family>.py` — moved to `pipelines/basic/<family>/`
- `configs/sample/` — directory removed entirely (**done** in PR 4); defaults absorbed by `presets.py`
- model-prefixed files in `pipelines/stages/` — moved to family `stages/` dirs

### What remains shared

- `configs/pipelines/base.py` — `PipelineConfig` base class
- `configs/models/` — architecture definitions (dits/, vaes/, encoders/)
- `pipelines/stages/` — shared stages only (denoising.py, encoding.py, etc.)

The refactor is not complete if the repo permanently keeps two equal, first-class public inference config systems, or if understanding a single model family requires visiting 4+ scattered directories.
