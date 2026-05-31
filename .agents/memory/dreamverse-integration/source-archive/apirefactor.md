# FastVideo API Refactor Design

## Related Documents
- [PR plan.md](PR%20plan.md) — PR-by-PR implementation plan for this design
- [.agents/exploration/streaming-server-upstream-plan.md](.agents/exploration/streaming-server-upstream-plan.md) — streaming-server upstream + Dynamo backend contract (shapes PRs 5.5-7.10)
- `../FastVideo-internal/.agents/exploration/rebase-upstream-fastvideo.md` — rebasing FastVideo-internal onto upstream (enables PRs 6-8)
- `../FastVideo-internal/ui/ltx2-streaming/` — source for the streaming server being upstreamed (PRs 7.5-7.9)
- `../dynamo/` — local clone of ai-dynamo/dynamo; `components/src/dynamo/sglang/` is the template for FastVideo's native backend landed in PR 7.10
- https://github.com/ai-dynamo/dynamo/pull/7544 — closed draft PR that establishes the Dynamo backend shape this design must satisfy

## Status

Design spec for the public inference API refactor. PRs 0-5.5 are landed; see [PR plan.md](PR%20plan.md) for rollout status and the PR 6+ roadmap. The typed schema, strict parser, preset system, typed VideoGenerator, typed CLI, and stateless OpenAI server default-request merge are all implemented. Streaming package skeleton + typed streaming config types are in place; live streaming server + Dynamo contract are the next milestones.

## Executive Summary

FastVideo should move to a single typed nested inference schema that is shared across:

- Python API
- CLI
- YAML/JSON config files
- OpenAI/server request translation

The core split is:

- `GeneratorConfig`: generator-instance lifetime settings
- `GenerationRequest`: per-call inputs, sampling, outputs, and continuation
- `InferencePreset`: model-owned named multi-stage defaults

The canonical user experience should be:

1. Choose a model.
2. Choose a pipeline preset.
3. Override a few typed fields.
4. Generate.

FastVideo should not make a raw free-form string dict the primary API. Dicts and YAML/JSON should be supported as serialization/interchange layers, but they must be parsed immediately into typed config objects with strict unknown-key validation.

The repo should also shift model-specific preset/default definitions closer to their pipeline implementations, while keeping the shared public schema and parsers centralized.

## Why This Refactor Is Needed

Today the public inference boundary is too flat and too forgiving.

- `VideoGenerator.from_pretrained(..., **kwargs)` mixes:
  - engine/runtime settings
  - pipeline init settings
  - component overrides
- `VideoGenerator.generate_video(..., **kwargs)` mixes:
  - prompt and inputs
  - sampling parameters
  - output settings
  - model-specific workflow knobs
- unknown or drifting keys can be silently filtered or merely logged instead of failing fast
- model-specific multi-stage behavior is exposed through ad hoc top-level flags instead of a stable preset/stage abstraction

This is already painful in LTX2/Dreamverse, and it will get worse as more multi-stage pipelines are upstreamed.

## Design Goals

- Keep the Python API typed and editor-friendly.
- Make YAML/JSON a first-class serialization of the same schema.
- Support CLI overrides cleanly without flattening the schema into hundreds of canonical flags.
- Separate init-time config from request-time config.
- Provide a stable public abstraction for multi-stage pipelines.
- Support LTX2 two-stage and continuation behavior cleanly.
- Keep the simple case simple.
- Co-locate model-owned defaults and stage topology with the relevant pipeline.
- Protect current public/server behavior with an explicit schema parity audit before freezing the new surface.
- Preserve backward compatibility long enough to migrate examples, internal users, and servers safely.

## Non-Goals

- Do not make Ray a structural dependency or copy its package layout.
- Do not make a raw free-form dict the primary Python API.
- Do not force all models into one universal `RefineConfig`.
- Do not expose stage indices as the primary user interface.
- Do not move every shared config class into per-model directories.

## External Inspiration

### Ray

Borrow only the ergonomic idea that user-facing config can be expressed as a string-keyed dict or YAML/JSON config. Do not copy Ray's structure into FastVideo.

### SGL Multimodal Gen

Useful ideas: split instance config from request config; allow dict input at the boundary; parse dicts immediately into typed request objects; merge request overrides onto model defaults; validate request params against pipeline/task type. Do not copy: request objects depending on server/engine config; broad weakly typed request bags as the canonical API.

### vLLM-Omni

Useful ideas: model-owned pipeline presets; explicit stage topology; per-stage default sampling params; clean separation between stage topology, engine defaults, and runtime overrides. Do not copy: positional `sampling_params_list` as the primary public API; serving-engine-oriented stage index semantics in the main Python interface.

## Core Decision

FastVideo should have:

1. A shared typed public schema.
2. Model-owned named pipeline presets.
3. Semantic stage overrides by stage name.
4. Optional advanced explicit plans for power users.
5. YAML-first config loading with dotted CLI overrides.

The public API should be stable at the schema level, while model-specific behavior should be contained in preset definitions and model-specific typed override classes.

## Schema Parity Requirement

Before the new schema is declared canonical, FastVideo should build a parity inventory across all current public inference surfaces (Python `VideoGenerator` kwargs, CLI flags, YAML/JSON config inputs, OpenAI/server request models, model-specific sampling/runtime fields). Each field must be marked: kept as-is, renamed, moved to a nested path, preset-owned, private-only adapter field, or intentionally dropped. No field should disappear implicitly.

For any public field that remains supported, there should be either a normalized-config equivalence test, or an explicit parser/translation test. Fields that exist only in private Dreamverse integration code should be handled by a private adapter layer, not quietly converted into public FastVideo compatibility guarantees.

Landed artifact: [inference_schema_parity_inventory.yaml](docs/design/inference_schema_parity_inventory.yaml) + guard [test_schema_parity_inventory.py](fastvideo/tests/api/test_schema_parity_inventory.py).

## Canonical Public Schema

The typed schema is implemented in [fastvideo/api/schema.py](fastvideo/api/schema.py). Envelope types:

- `RunConfig` — offline: `generator` (GeneratorConfig) + `request` (GenerationRequest)
- `ServeConfig` — serving: `generator` + `server` (ServerConfig) + `default_request` (GenerationRequest) + optional `streaming` (StreamingConfig)

Key nested types (summary; full fields in `schema.py`):

- `GeneratorConfig` → `model_path`, `revision`, `trust_remote_code`, `engine` (EngineConfig: parallelism/offload/compile/quantization/flags), `pipeline` (PipelineSelection: workload_type, preset, preset_version, components, preset_overrides, experimental)
- `GenerationRequest` → `prompt`, `negative_prompt`, `inputs` (InputConfig), `sampling` (SamplingConfig), `runtime` (RequestRuntimeConfig), `output` (OutputConfig), `stage_overrides`, `state` (ContinuationState), `plan` (GenerationPlan), `extensions`
- `ContinuationState` → opaque `{kind: str, payload: dict[str, Any]}`
- `GenerationPlan` → `{stages: list[PlannedStage], final_stage: str | None}`; advanced/escape-hatch only

### Important Semantics

- Dataclasses are canonical for Python users.
- Dict and YAML/JSON are parsed into these dataclasses immediately.
- Unknown keys must raise validation errors.
- Typed `GenerationRequest` defaults come from the public schema, not from model-specific `SamplingParam.from_pretrained(...)` defaults.
- Legacy `generate_video(...)` continues to inherit model-specific sampling defaults until the SSIM/performance migration lands (PR 11).
- The only open-ended escape hatches are:
  - `generator.pipeline.experimental`
  - `request.extensions`

That keeps the public contract strict without blocking experimental work.

### Request Mutation Tracking

When a `GenerationRequest` is parsed from a raw dict (YAML, JSON, or Python mapping), FastVideo records which fields the user explicitly provided versus which received schema defaults. This matters because `request_to_sampling_param()` must distinguish user-provided values (which should override model defaults) from schema defaults (which should NOT override model defaults).

The tracking contract:

- At parse time, the original raw dict and a baseline snapshot of the parsed object are stored on the request.
- Dataclass field mutations after parsing (e.g., `request.sampling.seed = 7`) are captured via lightweight `__setattr__` dirty-path recording.
- Dict-typed field mutations (e.g., `del request.stage_overrides["refine"]`) are detected at access time by diffing the current dict against the baseline snapshot.
- Setting a field to the schema default value IS captured as explicit, so it will override model defaults.
- The raw dict is reconciled lazily when `normalize_generation_request()` is called, not on every individual mutation.

### Schema Purity and Model-Specific Fields

The shared schema currently contains fields that are specific to one or two model families. These remain for backward compatibility during the initial migration (PRs 0-3) but should migrate to preset-owned typed override classes as the preset system lands (PRs 4-10).

**SamplingConfig fields to migrate:**

- `height_sr`, `width_sr`, `num_inference_steps_sr`: Hunyuan15 SR only. Target: `HunyuanSRStageOverride` in PR 10.
- `guidance_scale_2`, `boundary_ratio`: Wan2.2 and LingBotWorld only. Target: preset-owned overrides in the relevant model migration PR.

**InputConfig fields to migrate:**

- `mouse_cond`, `keyboard_cond`, `grid_sizes`: MatrixGame action control only. Target: `request.extensions` or a typed MatrixGame input config.
- `c2ws_plucker_emb`: LingBotWorld camera control only. Target: `request.extensions` or a typed LingBotWorld input config.
- `refine_from`, `stage1_video`: LongCat refinement only. Target: `LongCatRefineStageOverride` inputs or keep in `InputConfig` if they remain a public contract.

**Universal fields that stay in the shared schema:**

- `guidance_rescale`: used by multiple denoising stages across models, default 0.0. Universally applicable.
- `true_cfg_scale`: OpenAI adapter surface. Keep for protocol compatibility.

### Escape Hatch Sunset

`generator.pipeline.experimental` and `request.extensions` are intentional escape hatches for experimental and private work. They bypass strict validation by design.

Rules for escape hatch usage:

- New fields should not be added to `experimental` or `extensions` without a plan to either promote them to typed fields or remove them within two PR cycles.
- Each model migration PR (PRs 6-10) should review and shrink escape hatch usage for that model family.
- The compatibility layer currently routes unrecognized legacy kwargs into `experimental`. This pass-through should shrink as presets absorb model-specific fields.

## Public Python API

### New Canonical API

```python
from fastvideo import VideoGenerator
from fastvideo.api import (
    GeneratorConfig, GenerationRequest,
    EngineConfig, OutputConfig,
    PipelineSelection, SamplingConfig,
)

generator = VideoGenerator.from_pretrained(
    config=GeneratorConfig(
        model_path="/models/ltx2",
        engine=EngineConfig(num_gpus=1),
        pipeline=PipelineSelection(
            workload_type="t2v",
            preset="ltx2_two_stage",
        ),
    )
)

result = generator.generate(
    GenerationRequest(
        prompt="a fox running through snow",
        sampling=SamplingConfig(
            num_frames=121, height=1024, width=1536,
            num_inference_steps=8, seed=42,
        ),
        output=OutputConfig(save_video=True, return_state=True),
    )
)
```

### Accepted Construction Forms

Canonical:

```python
VideoGenerator.from_pretrained(config=GeneratorConfig(...))
VideoGenerator.from_config(GeneratorConfig(...))
VideoGenerator.from_file("run.yaml")
```

Stable convenience constructor:

```python
VideoGenerator.from_pretrained("model-id")
VideoGenerator.from_pretrained("model-id", num_gpus=2, use_fsdp_inference=False, ...)
```

Legacy compatibility:

```python
VideoGenerator.from_pretrained(model_path, **legacy_kwargs)
```

All constructor forms normalize through the same typed path. Stable convenience kwargs remain supported with no deprecation warning. Advanced model/pipeline-specific kwargs are accepted during migration but only as compatibility inputs that normalize into `GeneratorConfig`. The thing being deprecated over time is the unbounded legacy kwarg surface, not the `from_pretrained(...)` entrypoint itself.

### Generation Entry Point

Canonical: `generator.generate(request: GenerationRequest) -> GenerationResult`.

Compatibility alias: `generator.generate_video(prompt=..., **legacy_kwargs)` — converts legacy calls into a `GenerationRequest` and emits a deprecation warning.

During the compat period, `generate(request=...)` uses schema defaults while `generate_video(...)` preserves legacy model-default behavior. These paths intentionally differ until preset-owned defaults replace the remaining `SamplingParam` default logic (migrated in PR 11).

### Boundary Normalization Rule

Every public inference entrypoint normalizes into typed config objects before touching legacy internals. That includes Python constructors, generation calls, CLI `generate`, CLI `serve`, and OpenAI/server request translation. Legacy internals (`FastVideoArgs`, `SamplingParam`) may remain temporarily, but only behind a typed normalization boundary.

## Pipeline Presets

### Definition

An `InferencePreset` is a named model-owned preset that defines:

- workload selection
- stage topology
- per-stage defaults
- stage names
- allowed stage override types
- init-time feature requirements

The preset is not user-authored by default. It is supplied by the model integration.

### Why Presets Are The Right Abstraction

Users usually do not want to assemble a stage graph by hand. They want to say:

- use LongCat distill + refine
- use Hunyuan 1080p SR
- use LTX2 two-stage continuation mode

Presets provide a stable public noun for that behavior.

### Preset Naming Rules

- Use semantic names, not stage indices.
- Keep names stable across releases.
- If semantics change incompatibly, change `preset_version` or create a new preset name.

Examples: `ltx2_base`, `ltx2_two_stage`, `longcat_distill_refine`, `hunyuan15_sr_720p`, `hunyuan15_sr_1080p`.

### Preset-Owned Stage Names

Stage names are public and stable within a preset.

- LTX2: `base`, `refine`
- LongCat: `distill`, `refine`
- Hunyuan15: `base`, `sr_720p`, `sr_1080p`

Public overrides should reference these stage names, never stage indices.

## Stage Overrides

The main user override surface for multi-stage pipelines is:

```python
request.stage_overrides["refine"] = ...
```

Each model family should expose typed override classes for its stage names. Examples for the model families that land in PRs 6/9/10:

```python
@dataclass
class LTX2RefineStageOverride:
    enabled: bool | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    add_noise: bool | None = None
    image_crf: int | None = None
    video_position_offset_sec: float | None = None

@dataclass
class LongCatRefineStageOverride:
    t_thresh: float | None = None
    spatial_refine_only: bool | None = None
    num_cond_frames: int | None = None

@dataclass
class HunyuanSRStageOverride:
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
```

### Strictness Rules

- Stage names must exist in the selected preset.
- Override fields must be valid for that stage type.
- Unknown stage names and unknown fields must error.

## Advanced Explicit Plans

Presets should be the default API. `GenerationPlan` exists only for advanced composition or experimentation:

- building a custom workflow that is not yet standardized as a preset
- debugging or benchmarking stage combinations
- prototyping a future preset

Do not require `GenerationPlan` for normal users.

## Continuation State

Continuation must be a first-class part of the API.

### Public Contract

- `GenerationResult.state` may return a `ContinuationState`.
- `GenerationRequest.state` may accept a previously returned state.
- Most users should treat `state` as opaque and round-trip it back into the next request.

### Why This Matters

Dreamverse/LTX2 currently leaks continuation internals into app-level request fields like video conditions, audio clean latent, audio denoise mask, and segment offsets. Those should not remain top-level app-owned public API.

### State Design

Public surface:

```python
@dataclass
class ContinuationState:
    kind: str
    payload: dict[str, Any]
```

Internally, FastVideo should also define typed model-specific state subclasses, e.g. `LTX2ContinuationState` (PR 7) and `LongCatIntermediateState` if ever needed. Minimal stable surface: return state, pass state back in, validate that the state is compatible with the active preset.

Payload serialization: fields must be JSON-serializable or use an opaque blob-ID indirection for large tensors. This supports both the stateless OpenAI client round-trip AND future Dynamo prefill/decode disaggregation where prefill yields a state that decode hydrates across workers.

## YAML / JSON Design

YAML and JSON should be exact serializations of the typed schema, not a second unrelated config system. YAML is the primary documented format. JSON is accepted with the same schema.

### Run Config Example

```yaml
generator:
  model_path: /models/ltx2
  engine:
    num_gpus: 1
    parallelism: {tp_size: -1, sp_size: -1}
    offload: {dit: false, text_encoder: false, vae: false, pin_cpu_memory: true}
  pipeline:
    workload_type: t2v
    preset: ltx2_two_stage
    components:
      config_root: /models/ltx2-config
      upsampler_weights: /models/ltx2-refine
      lora_path: /models/ltx2-refine-lora
    preset_overrides:
      refine: {enabled: true, add_noise: true}

request:
  prompt: "a fox running through snow"
  sampling:
    num_frames: 121
    height: 1024
    width: 1536
    num_inference_steps: 8
    seed: 42
  output: {save_video: true, return_state: true}
  stage_overrides:
    refine: {num_inference_steps: 2, guidance_scale: 1.0}
```

### Serve Config Example

```yaml
generator:
  model_path: /models/ltx2
  engine: {num_gpus: 1}
  pipeline: {workload_type: t2v, preset: ltx2_two_stage}

server: {host: 0.0.0.0, port: 8000, output_dir: outputs/}

default_request:
  sampling: {num_frames: 121, height: 1024, width: 1536, num_inference_steps: 8}
  output: {save_video: false, return_frames: false}
```

### Validation Rules

- top-level schema must match `RunConfig` or `ServeConfig`
- unknown keys must fail
- dotted CLI overrides are applied to the nested config before typed parsing
- parse errors must include the exact nested path that failed

## CLI Design

Inference CLI reuses the best parts of the current training authoring flow (YAML-first authoring, dotted nested overrides, typed parsing after merge) but stays stricter than training at the public boundary because it is a user-facing API surface for Python, CLI, YAML/JSON, and serving.

### Canonical CLI Forms

```bash
fastvideo generate --config run.yaml
fastvideo generate --config run.yaml --request.sampling.seed 42
fastvideo generate --config run.yaml --generator.engine.num_gpus 2

fastvideo serve --config serve.yaml
fastvideo serve --config serve.yaml --server.port 8090
```

The CLI is config-only. Beyond `--config`, CLI input uses dotted override paths into the nested schema rather than maintaining a second flat flag surface.

Implementation: YAML/JSON is loaded into a nested dict, dotted CLI overrides are applied to the nested dict, then the result is parsed into typed config objects. Flat CLI flags are rejected so the nested schema stays canonical.

## OpenAI / Server Mapping

`fastvideo serve` loads `ServeConfig`. Incoming HTTP requests are translated into `GenerationRequest` by:

1. cloning `default_request`
2. applying API request fields onto that request
3. validating against the selected preset

This is similar in spirit to the SGL pattern of merging user overrides onto model defaults.

Rules:

- HTTP request translation must not bypass typed validation.
- multi-stage defaults should come from the preset and `default_request`, not from ad hoc server-local logic.
- stateful continuation requests should accept and return typed `ContinuationState` payloads.

Landed in PR 5 for the stateless OpenAI server at `fastvideo/entrypoints/openai/`. The streaming/session server (PRs 7.5-7.9) uses the same preset/default_request merge through `ServeConfig.streaming`.

## Streaming Server + Dynamo Backend

The typed public API is consumed by three server-class integrations. They must share one execution substrate so we don't grow three near-duplicate progress loops.

### The three consumers

| Consumer | Transport | Request shape | State |
|---|---|---|---|
| Stateless OpenAI (`fastvideo/entrypoints/openai/`) | HTTP POST | `GenerationRequest` merged onto `ServeConfig.default_request` | Stateless; continuation via opaque payload if needed |
| Streaming WebSocket (`fastvideo/entrypoints/streaming/`) | WebSocket JSON + binary fMP4 | `GenerationRequest` per segment, session-scoped | Server-held session (per-GPU continuation cache); snapshot on demand |
| Dynamo native backend (`ai-dynamo/dynamo/components/src/dynamo/fastvideo/`) | Dynamo RPC endpoint | `NvCreateVideoRequest` ↔ adapter ↔ `GenerationRequest` | Aggregated today; disaggregated prefill/decode later via `ContinuationState` |

### Shared execution substrate: `VideoGenerator.generate_async`

The OpenAI server, streaming server, and Dynamo backend all want the same thing: a typed async API that yields progress events and a typed final result. FastVideo exposes exactly one canonical entry point:

```python
async def generate_async(
    self,
    request: GenerationRequest,
) -> AsyncGenerator[VideoEvent, None]: ...
```

Events:

```python
@dataclass
class VideoProgressEvent:
    step: int
    total_steps: int
    stage: str  # "denoise" | "refine" | "decode" | ...

@dataclass
class VideoPartialEvent:
    frames: np.ndarray           # shape: (num_frames, H, W, 3)
    index: int                   # monotonic chunk index

@dataclass
class VideoFinalEvent:
    video_bytes: bytes | None    # mp4-encoded if requested
    tensor: torch.Tensor | None  # raw if requested
    metadata: dict[str, Any]
    continuation_state: ContinuationState | None

VideoEvent = VideoProgressEvent | VideoPartialEvent | VideoFinalEvent
```

The sync `generate_video(request=...) -> VideoResult` becomes a thin `asyncio.run` wrapper over `generate_async` that collects events and returns the final.

### Streaming server mapping

`fastvideo/entrypoints/streaming/` owns per-session state:

- `SessionStore.hydrate(state: ContinuationState) -> session_id`
- `SessionStore.snapshot(session_id) -> ContinuationState`
- Per-GPU implicit continuation cache (today's internal behavior) is wrapped as a `SessionStore` implementation.

Per-segment, the session writes a `GenerationRequest`, pipes the event stream to the WebSocket (progress → JSON messages, partial → fMP4 frames), and persists the final's `ContinuationState` into the session.

### Dynamo backend mapping

Dynamo's backend pattern (from `components/src/dynamo/sglang/`) is a pure Python import. FastVideo does not host a `fastvideo/entrypoints/dynamo/` subpackage; the integration lives in the Dynamo repo. FastVideo exposes a stable contract:

| Surface | Exposed as |
|---|---|
| Construction | `VideoGenerator.from_pretrained(model_path, **typed_kwargs)` |
| Execution (async) | `VideoGenerator.generate_async(request) -> AsyncGenerator[VideoEvent, None]` |
| Execution (sync) | `VideoGenerator.generate_video(request=...) -> VideoResult` |
| Typed request | `fastvideo.api.GenerationRequest`, `SamplingConfig`, `InputConfig` |
| Typed result | `fastvideo.api.VideoResult`, `VideoEvent`, `ContinuationState` |
| Health-check input | `VideoGenerator.default_health_check_request() -> GenerationRequest` |
| Config dump | `config_to_dict(cfg)` (already exists) |

Request/response mapping the Dynamo adapter must perform:

```
NvCreateVideoRequest          ->  fastvideo.api.GenerationRequest
  prompt                      ->    sampling.prompt
  size="WxH"                  ->    sampling.width, sampling.height
  seconds                     ->    seconds * nvext.fps -> sampling.num_frames
  input_reference             ->    input.image_path | input.video_path
  nvext.fps                   ->    sampling.fps
  nvext.num_frames            ->    sampling.num_frames (overrides seconds*fps)
  nvext.num_inference_steps   ->    sampling.num_inference_steps
  nvext.guidance_scale        ->    sampling.guidance_scale
  nvext.seed                  ->    sampling.seed
  nvext.negative_prompt       ->    sampling.negative_prompt
  response_format             ->    (handled at the adapter's output stage)

VideoFinalEvent               ->  NvVideosResponse
  video_bytes                 ->    data[0].b64_json  (if response_format=b64_json)
  uploaded URL                ->    data[0].url       (if response_format=url)
  metadata.inference_time_s   ->    inference_time_s
  continuation_state          ->    (reserved for future disaggregation)
```

All fields already exist (or will exist after PR 6's typed-kwarg expansion) on FastVideo's typed schema. **The adapter lives entirely in the Dynamo repo** at `components/src/dynamo/fastvideo/` — FastVideo does not host any Dynamo subpackage, dep, or CLI. The only FastVideo obligation is the stable public Python API listed above.

### Constraints this places on other sections

- **Continuation State** (see earlier section): `ContinuationState.payload` must be JSON-serializable or use an opaque blob-ID indirection for large tensors. This supports both the stateless OpenAI client round-trip *and* future Dynamo prefill/decode disaggregation, where prefill yields a state that decode hydrates across workers.
- **Typed GeneratorConfig** (see Public Python API): every flat legacy LTX2 kwarg currently used by the internal `gpu_pool.py` must have a typed home reachable from `GeneratorConfig`. Dynamo's `FastVideoArgGroup` builds the config from its CLI and must not have to know any legacy LTX2 name.
- **Public exports**: `from fastvideo import VideoGenerator`; `from fastvideo.api import GenerationRequest, SamplingConfig, ContinuationState, VideoResult, VideoEvent, VideoProgressEvent, VideoPartialEvent, VideoFinalEvent`.

## Repo Layout

### Shared Public API

`fastvideo/api/` contains the shared public API package. Current files:

- `schema.py` — `RunConfig`, `ServeConfig`, `ServerConfig`, `GeneratorConfig`, and all nested typed config dataclasses
- `sampling_param.py` — `SamplingParam` + `CacheParams` (canonical home since PR 4; former `configs/sample/base.py` location removed)
- `presets.py` — `InferencePreset`, `PresetStageSpec`, registry APIs
- `results.py` — `GenerationResult` / `VideoResult`
- `parser.py` — `from_dict`, `to_dict`, `load_yaml`, `load_json`, validation
- `overrides.py` — dotted override application
- `compat.py` — legacy Python kwargs translation
- `errors.py` — path-aware validation errors

May split further by concern in a future cleanup.

### Pipeline-Local Model-Owned Config

Model-owned presets and override types live next to the model pipeline:

```text
fastvideo/pipelines/basic/ltx2/
  ltx2_pipeline.py, presets.py, stage_overrides.py, continuation.py

fastvideo/pipelines/basic/longcat/
  longcat_pipeline.py, presets.py, stage_overrides.py

fastvideo/pipelines/basic/hunyuan15/
  hunyuan15_pipeline.py, hunyuan15_sr_pipeline.py, hunyuan15_2sr_pipeline.py,
  presets.py, stage_overrides.py
```

PR 4 landed `presets.py` for all 13 model families. Remaining colocation targets are `pipeline_configs.py` (moving `configs/pipelines/<family>.py`) and model-specific stages (moving `pipelines/stages/<family>_*.py`); see [PR plan.md](PR%20plan.md) "Pipeline Package Structure".

### Registry

Central registry (`fastvideo/registry.py`) registers preset providers rather than owning all model-specific defaults directly. It answers:

- which pipeline class corresponds to a model path
- which presets are available for that model family
- which override/state classes are valid for a selected preset

## Relationship To Current Internal Classes

This refactor does not require deleting current internals immediately.

- `FastVideoArgs` is an internal compatibility/input adapter, no longer the primary public inference type.
- `SamplingParam` now lives in `fastvideo/api/sampling_param.py` and gets model-specific defaults from presets via `_from_preset()`. All 12 `SamplingParam` subclasses have been removed and the former `fastvideo/configs/sample/` directory has been deleted entirely (PR 4). It remains an internal adapter between the preset system and the runtime.
- current `PipelineConfig` classes can remain temporarily as internal component config carriers
- the new public schema is the stable boundary above them

`VideoGenerator` accepts the new schema and translates down into current execution internals. Legacy `generate_video(..., **kwargs)` stays on the direct execution path during the compat period until SSIM/performance tests migrate in PR 11.

## Model-Specific Design

### LTX2 / Dreamverse

LTX2 needs both:

- init-time two-stage feature wiring
- request-time continuation/refine behavior

Expressed as:

- preset: `ltx2_two_stage`
- init-time fields: refine assets, optional config root, stage enablement
- request-time fields: stage override for refine behavior, optional returned continuation state

#### LTX2 Preset Example

```yaml
generator:
  pipeline:
    preset: ltx2_two_stage
    components:
      config_root: /models/ltx2-config
      upsampler_weights: /models/ltx2-refine
      lora_path: /models/ltx2-refine-lora
    preset_overrides:
      refine: {enabled: true, add_noise: true}
```

#### LTX2 Request Example

```yaml
request:
  prompt: "continue the previous sequence"
  state: ${previous_result.state}
  stage_overrides:
    refine:
      num_inference_steps: 2
      guidance_scale: 1.0
      image_crf: 18
  output:
    return_state: true
```

#### LTX2 Explicit Decisions

- `config_model_path` becomes `generator.pipeline.components.config_root`
- `ltx2_refine_*` stops being a pile of top-level kwargs
- continuation internals move into `ContinuationState`
- app-level code should pass `state`, not raw latent/audio condition payloads

### LongCat

LongCat should expose a named preset like `longcat_distill_refine` with stage topology `distill` and `refine`.

User-facing override knobs remain model-specific (`t_thresh`, `spatial_refine_only`, `num_cond_frames`) but live under:

```yaml
request:
  stage_overrides:
    refine:
      t_thresh: 0.5
      spatial_refine_only: false
      num_cond_frames: 8
```

### Hunyuan 1.5 SR

Hunyuan already behaves like an integrated multi-stage pipeline. Expose it via presets: `hunyuan15_sr_720p`, `hunyuan15_sr_1080p`. Users should not need to know the exact internal pipeline class split between base and SR stages. Per-stage override surface should stay small and mostly sampling-focused.

Hunyuan15 presets (`hunyuan15_t2v_480p`, `hunyuan15_i2v_480p_distilled`, `hunyuan15_t2v_720p`, `hunyuan15_i2v_720p_distilled`, `hunyuan15_sr_1080p`) are implemented (PR 4). The `Hunyuan15_*_SamplingParam` subclasses have been removed; defaults (including precomputed sigmas) come from preset `defaults` dicts. Remaining work: adding typed `HunyuanSRStageOverride` classes and colocating PipelineConfig (PR 10).

## Exact Compatibility Mapping

Intended translation layer for common current fields.

| Legacy Field | New Path |
| --- | --- |
| `model_path` | `generator.model_path` |
| `revision` | `generator.revision` |
| `trust_remote_code` | `generator.trust_remote_code` |
| `workload_type` | `generator.pipeline.workload_type` |
| `num_gpus` | `generator.engine.num_gpus` |
| `tp_size` | `generator.engine.parallelism.tp_size` |
| `sp_size` | `generator.engine.parallelism.sp_size` |
| `dit_cpu_offload` | `generator.engine.offload.dit` |
| `dit_layerwise_offload` | `generator.engine.offload.dit_layerwise` |
| `text_encoder_cpu_offload` | `generator.engine.offload.text_encoder` |
| `image_encoder_cpu_offload` | `generator.engine.offload.image_encoder` |
| `vae_cpu_offload` | `generator.engine.offload.vae` |
| `pin_cpu_memory` | `generator.engine.offload.pin_cpu_memory` |
| `enable_torch_compile` | `generator.engine.compile.enabled` |
| `torch_compile_kwargs` | split across `generator.engine.compile.backend`, `.fullgraph`, `.mode`, `.dynamic`; uncommon keys land in `.extras` |
| `enable_torch_compile_text_encoder` | `generator.engine.compile.text_encoder_enabled` |
| `enable_stage_verification` | `generator.engine.enable_stage_verification` |
| `prompt_txt` | `request.inputs.prompt_path` |
| `prompt` | `request.prompt` |
| `negative_prompt` | `request.negative_prompt` |
| `image_path` | `request.inputs.image_path` |
| `video_path` | `request.inputs.video_path` |
| `output_path` | `request.output.output_path` |
| `output_video_name` | `request.output.output_video_name` |
| `save_video` | `request.output.save_video` |
| `return_frames` | `request.output.return_frames` |
| `num_videos_per_prompt` | `request.sampling.num_videos_per_prompt` |
| `seed` | `request.sampling.seed` |
| `num_frames` | `request.sampling.num_frames` |
| `height` | `request.sampling.height` |
| `width` | `request.sampling.width` |
| `fps` | `request.sampling.fps` |
| `num_inference_steps` | `request.sampling.num_inference_steps` |
| `guidance_scale` | `request.sampling.guidance_scale` |
| `guidance_scale_2` | `request.sampling.guidance_scale_2` |
| `guidance_rescale` | `request.sampling.guidance_rescale` |
| `true_cfg_scale` | `request.sampling.true_cfg_scale` |
| `boundary_ratio` | `request.sampling.boundary_ratio` |
| `sigmas` | `request.sampling.sigmas` |
| `enable_teacache` | `request.runtime.enable_teacache` |
| `return_trajectory_latents` | `request.runtime.return_trajectory_latents` |
| `return_trajectory_decoded` | `request.runtime.return_trajectory_decoded` |

### Private Dreamverse Adapter Mapping

The mappings below are useful for private Dreamverse migration, but they should not be treated as a public FastVideo backward-compatibility promise unless and until those fields actually exist in the public repo surfaces.

| Private Adapter Field | New Path |
| --- | --- |
| `config_model_path` | `generator.pipeline.components.config_root` |
| `ltx2_refine_enabled` | `generator.pipeline.preset_overrides.refine.enabled` |
| `ltx2_refine_upsampler_path` | `generator.pipeline.components.upsampler_weights` |
| `ltx2_refine_lora_path` | `generator.pipeline.components.lora_path` |
| `ltx2_refine_num_inference_steps` | `request.stage_overrides.refine.num_inference_steps` |
| `ltx2_refine_guidance_scale` | `request.stage_overrides.refine.guidance_scale` |
| `ltx2_refine_add_noise` | `generator.pipeline.preset_overrides.refine.add_noise` |
| `ltx2_image_crf` | `request.stage_overrides.refine.image_crf` |
| `return_continuation_state` | `request.output.return_state` |

### LongCat Legacy Mapping

| Legacy Field | New Path |
| --- | --- |
| `refine_from` | `request.inputs.refine_from` |
| `stage1_video` | `request.inputs.stage1_video` |
| `t_thresh` | `request.stage_overrides.refine.t_thresh` |
| `spatial_refine_only` | `request.stage_overrides.refine.spatial_refine_only` |
| `num_cond_frames` | `request.stage_overrides.refine.num_cond_frames` |

## Validation and Error Handling

### Strict by Default

All structured inputs should be strict by default: unknown keys error, wrong types error, invalid stage names error, incompatible state/preset combinations error.

### Exceptions

The only intentionally open-ended fields are `generator.pipeline.experimental` and `request.extensions`. These must be clearly documented as unstable and unsupported for long-term API compatibility.

### Error Quality

Validation errors should include the full nested path, expected type or valid choices, and preset/stage context when relevant:

```text
Invalid field: request.stage_overrides.refine.num_inference_steps
Expected int, got "two"
Preset: ltx2_two_stage
Stage: refine
```

## Implementation Plan

### Phases 0-5: Landed

- Phase 0 — Schema Parity Inventory: inventory complete; field classifications live in `docs/design/inference_schema_parity_inventory.yaml`; parity test guard in `fastvideo/tests/api/test_schema_parity_inventory.py`.
- Phase 1 — Shared Schema: `fastvideo/api/` with typed dataclasses, parser, validation, dotted overrides, `RunConfig`/`ServeConfig`.
- Phase 2 — VideoGenerator Compat: `from_config`, `from_file`, `generate(request=...)`, legacy `from_pretrained(..., **kwargs)` and `generate_video(..., **kwargs)` as compat shims routed through typed normalization.
- Phase 3 — CLI Refactor: `fastvideo generate` and `fastvideo serve` parse nested YAML/JSON with training-style dotted overrides; flat flag expansion removed as the canonical path.
- Phase 4 — Preset System: shared registry + pipeline-local `presets.py` for all 13 families; all 12 `SamplingParam` subclasses removed; `SamplingParam` moved to `fastvideo/api/sampling_param.py`.
- Phase 5 — Server Request Translation: `fastvideo serve` loads `ServeConfig`; stateless OpenAI endpoint clones `default_request` and merges validated user overrides.

### Remaining Phases

- **Phase 6 — LTX2 Public Upstream Path** (PR 6): upstream `ltx2_two_stage` preset; upstream continuation-state contract; upstream only repo-visible/public LTX2 surfaces into FastVideo.
- **Phase 7 — Dreamverse Adapter Migration** (PR 7 + private repo work): translate private Dreamverse-only request/config fields in a private adapter; replace raw app-owned continuation kwargs with `state` in the private server; do not expand the public FastVideo compatibility promise just to match private adapter fields.
- **Phase 7.5-7.10 — Streaming Server and Dynamo Contract** (PRs 7.5-7.10): upstream the streaming server (skeleton, GPU pool, prompt enhancer, auxiliaries, router) consuming `generate_async`; land the Dynamo backend contract (`VideoGenerator.generate_async`, health-check helper) with the Dynamo backend package itself living in the Dynamo repo.
- **Phase 8 — Model Migration and Docs** (PRs 9-10, 12): colocate `configs/pipelines/<family>.py` with pipeline implementations; add typed stage override classes for multi-stage models; update basic examples to the new API; document YAML-first inference config and migration guidance.
- **Phase 8.5 — Golden-Test Migration** (PR 11): keep SSIM/performance regression tests on legacy Python generation while preset defaults are still settling; one dedicated migration pass after the preset system and model-default behavior are stable; complete this migration before removing legacy Python inference entrypoints or kwargs.
- **Phase 9 — Deprecation and Cleanup** (PR 13): deprecate direct public use of `FastVideoArgs`; deprecate direct public use of `SamplingParam`; gradually reduce public documentation for flat flags; eventually remove legacy kwargs after downstream migration is complete.

## Final Recommendation

The public FastVideo inference API is being rebuilt around:

- typed nested configs
- model-owned named presets
- semantic stage overrides
- first-class continuation state
- YAML-first CLI with dotted overrides

The primary abstraction is `InferencePreset`, not raw kwargs and not a fully manual stage graph.

The repo is moving model-specific defaults closer to each pipeline, while keeping the public schema and parsing logic centralized.

Regression and quality tests follow the rollout. Unit/entrypoint tests migrated to the typed API early, but SSIM/performance suites only move once the typed path can express all current knobs without compatibility exceptions and produces stable defaults through presets (PR 11).

End state:

- stable Python typing
- clean YAML/JSON support
- a much better CLI story
- a sane path for Dreamverse/LTX2
- a unified abstraction for LongCat, Hunyuan, and future multi-stage models
