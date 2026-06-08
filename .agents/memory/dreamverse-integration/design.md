# Design — Typed Public Inference API

Synthesis of the FastVideo public inference API refactor design philosophy.
For PR-by-PR execution see [pr-roadmap.md](pr-roadmap.md). For the streaming
extension see [streaming-server.md](streaming-server.md).

**Last updated:** 2026-05-03.

## Why the refactor

The pre-refactor public boundary mixed three concerns through `**kwargs`:

- `VideoGenerator.from_pretrained(..., **kwargs)` mixed engine/runtime,
  pipeline init, and component overrides.
- `VideoGenerator.generate_video(..., **kwargs)` mixed prompt+inputs,
  sampling, output, and model-specific workflow knobs.
- Unknown keys silently filtered or merely logged → API drift hard to detect.
- Multi-stage models (LTX-2 two-stage, Hunyuan15 SR, LongCat distill+refine)
  exposed via ad hoc top-level flags.

This was already painful for LTX2/Dreamverse and would worsen as more
multi-stage pipelines came in.

## Core decision

FastVideo has:

1. **Typed nested public schema** — `RunConfig`, `ServeConfig`,
   `GeneratorConfig`, `GenerationRequest`, `ContinuationState`.
2. **Model-owned named pipeline presets** — `ltx2_two_stage`,
   `longcat_distill_refine`, `hunyuan15_sr_1080p`, etc. All 13 model families
   landed presets in PR 4.
3. **Semantic stage overrides by stage name** —
   `request.stage_overrides["refine"] = LTX2RefineStageOverride(...)`.
4. **Optional advanced explicit plans** for power users — `GenerationPlan`
   (escape hatch only; not the canonical surface).
5. **YAML-first CLI** with dotted overrides —
   `fastvideo generate --config run.yaml --request.sampling.seed 42`.

The canonical user experience: choose a model → choose a preset → override
a few typed fields → generate. Dicts/YAML/JSON are supported as
serialization, but parse immediately into typed objects with strict
unknown-key validation.

## Schema surface

Implemented in [`fastvideo/api/`](file:///home/william5lin/FastVideo/fastvideo/api/):

| Type | Role |
|---|---|
| `RunConfig` | Offline envelope: `generator` + `request` |
| `ServeConfig` | Serving envelope: `generator` + `server` + `default_request` + optional `streaming` |
| `GeneratorConfig` | `model_path`, `revision`, `trust_remote_code`, `engine`, `pipeline` |
| `EngineConfig` | parallelism / offload / compile / quantization / flags |
| `PipelineSelection` | `workload_type`, `preset`, `preset_version`, `components`, `preset_overrides`, `experimental` |
| `GenerationRequest` | `prompt`, `negative_prompt`, `inputs`, `sampling`, `runtime`, `output`, `stage_overrides`, `state`, `plan`, `extensions` |
| `ContinuationState` | Opaque envelope `{kind: str, payload: dict[str, Any]}` |
| `GenerationPlan` | Advanced/escape-hatch only; `{stages: list[PlannedStage], final_stage: str|None}` |

Files:

| File | Role |
|---|---|
| [`schema.py`](file:///home/william5lin/FastVideo/fastvideo/api/schema.py) | All public dataclasses |
| [`parser.py`](file:///home/william5lin/FastVideo/fastvideo/api/parser.py) | `from_dict`, `to_dict`, `load_yaml`, `load_json`, validation |
| [`overrides.py`](file:///home/william5lin/FastVideo/fastvideo/api/overrides.py) | Dotted override application |
| [`compat.py`](file:///home/william5lin/FastVideo/fastvideo/api/compat.py) | Legacy kwargs translation (~370 lines, scheduled for death PRs 14-17) |
| [`presets.py`](file:///home/william5lin/FastVideo/fastvideo/api/presets.py) | Preset registry |
| [`sampling_param.py`](file:///home/william5lin/FastVideo/fastvideo/api/sampling_param.py) | Internal `SamplingParam` adapter (canonical home since PR 4) |
| [`results.py`](file:///home/william5lin/FastVideo/fastvideo/api/results.py) | `GenerationResult` / `VideoResult` |
| [`errors.py`](file:///home/william5lin/FastVideo/fastvideo/api/errors.py) | Path-aware validation errors |

## Boundary normalization rule

Every public inference entrypoint normalizes into typed config objects
before touching legacy internals (`FastVideoArgs`, `SamplingParam`).
Includes Python constructors, `generate*` calls, CLI `generate`, CLI
`serve`, OpenAI server request translation, streaming server request
translation.

Legacy internals (`FastVideoArgs`, `SamplingParam`) may remain temporarily,
but only behind a typed normalization boundary.

## Strict-by-default validation

All structured inputs are strict:

- Unknown keys → error
- Wrong types → error
- Invalid stage names → error
- Incompatible state/preset combinations → error

The only intentional escape hatches:

- `generator.pipeline.experimental` — for in-flight features without typed home
- `request.extensions` — same, request-side

These bypass validation by design. Intent: shrink as presets absorb
model-specific fields. New fields should not land in `experimental` /
`extensions` without a plan to either promote them to typed fields or
remove them within two PR cycles.

Error format includes nested path:

```
Invalid field: request.stage_overrides.refine.num_inference_steps
Expected int, got "two"
Preset: ltx2_two_stage
Stage: refine
```

## Request mutation tracking

When a `GenerationRequest` is parsed from raw dict (YAML/JSON/Python),
FastVideo tracks which fields the user explicitly provided vs. which got
schema defaults. Matters for `request_to_sampling_param()` — explicit
values override model defaults; schema defaults do NOT.

Mechanics:

- At parse time, original raw dict + baseline snapshot stored on the request.
- Dataclass field mutations (e.g. `request.sampling.seed = 7`) captured via
  lightweight `__setattr__` dirty-path recording.
- Dict-typed field mutations (e.g. `del request.stage_overrides["refine"]`)
  detected at access time by diffing current dict vs. baseline.
- Setting a field to its schema default value IS captured as explicit, so
  it overrides model defaults.
- Raw dict reconciled lazily when `normalize_generation_request()` is called.

## Schema purity (model-specific fields still in shared schema)

Remain for back-compat during initial migration; targeted for migration
into preset-owned typed override classes:

| Field | Owner | Migration target |
|---|---|---|
| `SamplingConfig.height_sr` / `width_sr` / `num_inference_steps_sr` | Hunyuan15 SR | `HunyuanSRStageOverride` (PR 10) |
| `SamplingConfig.guidance_scale_2`, `boundary_ratio` | Wan2.2, LingBotWorld | preset-owned (per-family PR) |
| `InputConfig.mouse_cond`, `keyboard_cond`, `grid_sizes` | MatrixGame | `request.extensions` or typed input config |
| `InputConfig.c2ws_plucker_emb` | LingBotWorld | `request.extensions` or typed input config |
| `InputConfig.refine_from`, `stage1_video` | LongCat | `LongCatRefineStageOverride` inputs (PR 9) |

LTX-2 multi-modal CFG knobs (`ltx2_modality_scale_video/_audio`,
`ltx2_rescale_scale`, `ltx2_stg_scale_video/_audio`,
`ltx2_stg_blocks_video/_audio`) still leak into shared `SamplingParam` but
only LTX-2 reads them today. Migration to typed `LTX2SamplingOverride` is
deferred to per-model migration sweep.

**LTX-2 CFG-force fix landed in PR 6**: defaults moved from `3.0/7.0` to
`1.0/1.0` to stop force-enabling CFG for non-LTX-2 families.
`ltx2_base` preset still sets `3.0/7.0` explicitly. Regression guard:
`test_presets.py::TestPresetDefaultTypes::test_ltx2_cfg_defaults_are_off`.

## Continuation state

Public surface:

```python
@dataclass
class ContinuationState:
    kind: str            # e.g. "ltx2.v1"
    payload: dict[str, Any]
```

Internally, model-specific typed subclasses (e.g. `LTX2ContinuationState`
at [`fastvideo/pipelines/basic/ltx2/continuation.py`](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/continuation.py)).

Payload must be JSON-serializable or use opaque blob-ID indirection for
large tensors — supports both stateless OpenAI client round-trip AND
future Dynamo prefill/decode disaggregation.

Hybrid model: server-held for streaming WS, client-round-trip for
stateless HTTP. See [streaming-server.md](streaming-server.md) D-1.

## Pipeline package structure (target)

Per-family colocation under `pipelines/basic/<family>/`:

```
fastvideo/pipelines/basic/<family>/
├── <family>_pipeline.py      # pipeline implementation(s)
├── presets.py                # user-facing presets (DONE in PR 4)
├── pipeline_configs.py       # engine/arch config (from configs/pipelines/)
└── stages/                   # model-specific stages (optional, if >2 files)
```

What stays shared:

- `configs/pipelines/base.py` — `PipelineConfig` base class
- `configs/models/` — architecture defs (dits/, vaes/, encoders/)
- `pipelines/stages/` — shared stages only (denoising, encoding, decoding,
  text_encoding, timestep_preparation, ...)

What's gone (PR 4):

- `fastvideo/configs/sample/` — directory removed entirely; defaults
  absorbed into per-family `presets.py`.
- All 12 `*_SamplingParam` subclass files — `SamplingParam` lives at
  `fastvideo/api/sampling_param.py`; defaults flow through
  `SamplingParam.from_pretrained()` → `_from_preset()`.

What's pending: `configs/pipelines/<family>.py` colocation, optional
`pipelines/stages/<family>_*.py` colocation. Per-model migration PRs
(6/9/10) include the colocation step for that family.

## YAML examples

### Run config

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
  sampling: {num_frames: 121, height: 1024, width: 1536, num_inference_steps: 8, seed: 42}
  output: {save_video: true, return_state: true}
  stage_overrides:
    refine: {num_inference_steps: 2, guidance_scale: 1.0}
```

### Serve config

See [`Dreamverse/serve_configs/streaming_demo.yaml`](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml)
for a canonical example matching internal/ui defaults (LTX-2 distilled,
NVFP4, 121 frames @ 1088×1920 24fps, 5 inference steps, 2-step refine).

## Compatibility mapping (legacy → typed)

| Legacy field | New path |
|---|---|
| `model_path` | `generator.model_path` |
| `num_gpus` | `generator.engine.num_gpus` |
| `tp_size` / `sp_size` | `generator.engine.parallelism.{tp_size,sp_size}` |
| `dit_cpu_offload` | `generator.engine.offload.dit` |
| `enable_torch_compile` | `generator.engine.compile.enabled` |
| `torch_compile_kwargs` | split: `generator.engine.compile.{backend,fullgraph,mode,dynamic}` + `.extras` |
| `enable_torch_compile_text_encoder` | `generator.engine.compile.text_encoder_enabled` |
| `prompt_txt` | `request.inputs.prompt_path` |
| `image_path` / `video_path` | `request.inputs.{image_path,video_path}` |
| `output_path` / `save_video` / `return_frames` | `request.output.*` |
| `seed` / `num_frames` / `height` / `width` / `fps` / `num_inference_steps` / `guidance_scale` | `request.sampling.*` |
| `enable_teacache` / `return_trajectory_*` | `request.runtime.*` |

LTX-2 specific (private adapter, NOT public compat promise):

| Legacy LTX-2 field | New path |
|---|---|
| `config_model_path` | `generator.pipeline.components.config_root` |
| `ltx2_refine_enabled` | `generator.pipeline.preset_overrides.refine.enabled` |
| `ltx2_refine_upsampler_path` | `generator.pipeline.components.upsampler_weights` |
| `ltx2_refine_lora_path` | `generator.pipeline.components.lora_path` |
| `ltx2_refine_num_inference_steps` | `request.stage_overrides.refine.num_inference_steps` |
| `ltx2_refine_guidance_scale` | `request.stage_overrides.refine.guidance_scale` |
| `ltx2_refine_add_noise` | `generator.pipeline.preset_overrides.refine.add_noise` |
| `ltx2_image_crf` | `request.stage_overrides.refine.image_crf` |
| `return_continuation_state` | `request.output.return_state` |

LongCat:

| Legacy | New |
|---|---|
| `refine_from` / `stage1_video` | `request.inputs.{refine_from,stage1_video}` |
| `t_thresh` / `spatial_refine_only` / `num_cond_frames` | `request.stage_overrides.refine.*` |

## External inspirations (and limits)

| Source | Useful idea | Don't copy |
|---|---|---|
| Ray | YAML-first config interchange | Ray's package layout |
| SGL `multimodal_gen` | Split instance/request config; dict input parsed into typed objects; merge user overrides on model defaults | `SamplingParams._adjust(ServerArgs)` (request depending on engine config); broad weakly-typed request bags |
| vLLM-Omni | Model-owned pipeline presets; explicit stage topology; per-stage default sampling | Positional `sampling_params_list`; serving-engine stage-index semantics in primary Python API |

## Naming guidance

- Public schema names namespaced under `fastvideo.api`
- Don't export from top-level `fastvideo/__init__.py` until migration further along
- `RunConfig` / `ServeConfig` get sufficient disambiguation from training
  config via the namespace
- Future rename to `EngineQuantizationConfig` reserved if a collision
  arises (deferred)

## Public Python API (canonical form)

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
        pipeline=PipelineSelection(workload_type="t2v", preset="ltx2_two_stage"),
    )
)

result = generator.generate(
    GenerationRequest(
        prompt="a fox running through snow",
        sampling=SamplingConfig(num_frames=121, height=1024, width=1536,
                                 num_inference_steps=8, seed=42),
        output=OutputConfig(save_video=True, return_state=True),
    )
)
```

Accepted constructor forms:

```python
VideoGenerator.from_pretrained(config=GeneratorConfig(...))
VideoGenerator.from_config(GeneratorConfig(...))
VideoGenerator.from_file("run.yaml")
VideoGenerator.from_pretrained("model-id", num_gpus=2, ...)  # stable convenience
VideoGenerator.from_pretrained(model_path, **legacy_kwargs)   # compat (deprecated PR 13)
```
