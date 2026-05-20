# FastVideo Streaming Server Upstream — Design & Plan

## Status
Exploration / design draft. Captures the re-evaluation triggered by the
decision to upstream `FastVideo-internal/ui/ltx2-streaming/server/` into
the public repo. Not yet approved for execution.

## Related Documents
- [PR plan.md](../../PR%20plan.md) — PR-by-PR implementation plan for the API refactor
- [apirefactor.md](../../apirefactor.md) — design spec this plan implements
- `../../../FastVideo-internal/ui/ltx2-streaming/` — upstream source (server side)
- `../../../dynamo/` — local clone of ai-dynamo/dynamo; backend patterns at
  `components/src/dynamo/{vllm,sglang,trtllm}/` and `CLAUDE.md` files
- https://github.com/ai-dynamo/dynamo/pull/7544 — draft PR that promotes
  FastVideo to a native Dynamo backend (CLOSED, superseded — but establishes
  the integration shape)

## Context

The internal `FastVideo-internal/ui/ltx2-streaming/` directory contains a
complete LTX2 streaming service. The user has decided:

- **Frontend clients** (`client/`, `prod-ui/`) stay in the internal repo
- **Everything server-side** — FastAPI/WebSocket server, GPU pool, prompt
  enhancer, router, auxiliaries — will be upstreamed to FastVideo

In parallel, FastVideo is becoming a **first-class Dynamo backend** (same
tier as vllm, sglang, trtllm). The refactor must produce an API that
Dynamo's `components/src/dynamo/fastvideo/` package can consume as a
pure Python import, without re-introducing the legacy flat-kwarg
surface. Draft PR ai-dynamo/dynamo#7544 defines the concrete integration
shape we need to support.

This materially changes the tail of the API refactor plan. The current
PR 5 ("wire `ServeConfig.default_request` into the OpenAI-compatible
HTTP server") addresses only the stateless endpoint; the real upstream
target is a much larger, session-based stack **plus** a clean Dynamo
backend contract.

This document captures:
- what's being upstreamed and where it lands
- four design decisions that shape the upstream (continuation model,
  streaming server layout, LLM provider abstraction, Dynamo backend
  integration)
- a revised PR sequence for the tail of the refactor

## What's being upstreamed

| Internal path | Size | Role | Upstream target |
|---|---|---|---|
| `server/main.py` | 94KB | FastAPI + WebSocket, session lifecycle, segment orchestration | `fastvideo/entrypoints/streaming/server.py` + handlers |
| `server/gpu_pool.py` | 66KB | GPU orchestration, subprocess workers | `fastvideo/entrypoints/streaming/gpu_pool.py` |
| `server/prompt_enhancer.py` | 69KB | LLM orchestration (cerebras_ifm, cerebras, groq) | `fastvideo/entrypoints/streaming/prompt/` package |
| `server/mock_server.py` | 45KB | Mock backend for dev/tests | `fastvideo/entrypoints/streaming/mock_server.py` |
| `server/prompt_safety.py` | 7KB | Optional fasttext-gated prompt safety | `fastvideo/entrypoints/streaming/prompt/safety.py` |
| `server/session_init_image.py` | 3KB | i2v init image handling | `fastvideo/entrypoints/streaming/session_init_image.py` |
| `server/rewrite_prompt_payload.py` | 3KB | Rewrite flow payload builder | `fastvideo/entrypoints/streaming/prompt/rewrite.py` |
| `server/session_logger.py` | 1KB | Session JSONL logs | `fastvideo/entrypoints/streaming/session_logger.py` |
| `server/config.py` | 9KB | Env-driven server config | Typed `ServeConfig` extensions |
| `router/main.py` | 27KB | Multi-replica load balancer + WS proxy | `fastvideo/entrypoints/streaming/router/` (or separate package) |
| `slurm/` | — | Deployment scripts | Likely stays internal |

## FastVideo contact surface today

Direct calls from the internal stack into FastVideo, all in `gpu_pool.py`:

| Location | Call | Notes |
|---|---|---|
| `gpu_pool.py:164` | `from fastvideo.entrypoints.video_generator import VideoGenerator` | Subprocess-level import, post-`CUDA_VISIBLE_DEVICES` setup |
| `gpu_pool.py:230` | `PipelineConfig.from_pretrained(config_model_path)` | Direct access to legacy `PipelineConfig` |
| `gpu_pool.py:231` | `pipeline_config.dit_config.quant_config = FP4Config()` | Direct internals mutation |
| `gpu_pool.py:264-267` | `VideoGenerator.from_pretrained(model_root, **load_kwargs)` | Flat legacy kwargs |
| `gpu_pool.py:837` | `generator.generate_video(**request_kwargs)` | Per-segment flat kwargs |
| `gpu_pool.py:282-288` | `LTX2AudioEncoder`, `AudioProcessor`, `get_diffusers_config` | Audio re-encode path |

`load_kwargs` at `gpu_pool.py:233-260` contains:
`ltx2_refine_enabled`, `ltx2_refine_upsampler_path`, `ltx2_refine_lora_path`,
`ltx2_refine_num_inference_steps`, `ltx2_refine_guidance_scale`,
`ltx2_refine_add_noise`, `pipeline_config`, `torch_compile_kwargs`,
`dit_cpu_offload`, `dit_layerwise_offload`, `vae_cpu_offload`,
`text_encoder_cpu_offload`, `pin_cpu_memory`, `ltx2_vae_tiling`,
`use_fsdp_inference`, `enable_torch_compile`.

`request_kwargs` at `gpu_pool.py:837` includes:
`ltx2_audio_clean_latent`, `ltx2_audio_denoise_mask`,
`ltx2_video_conditions`, `video_position_offset_sec`, standard sampling
fields.

**Implication**: upstreaming `gpu_pool.py` as-is perpetuates the flat
kwarg surface inside the public server. We need a typed translation
(PR 6 expansion) at the worker boundary before, or as part of, the
gpu_pool upstream.

## Session / continuation semantics today

Per-session state (in `server/main.py`):
- `locked_segment_prompts`, `curated_prompts`, `segment_idx`,
  `generated_segment_count`, `loop_iteration`

Per-**GPU** (not per-session) continuation cache (in `gpu_pool.py`):
- `ltx2_continuation_images` — last 9 decoded frames for clip conditioning
- `ltx2_continuation_audio_latents` — denoised audio latents for audio conditioning

Segment N+1 automatically conditions on segment N's trailing frames and
audio. On session reset or handoff (`USER_JOIN`), the per-GPU cache is
cleared. There is currently **no way for a client to serialize and
resume continuation state elsewhere** — it lives on the GPU only.

## Design Decision 1: Continuation model

### Options

**A. Opaque client-round-trip payload** (current plan PR 7 design)
- Server returns `ContinuationState(kind, payload)`; client sends it back.
- Pro: stateless server, trivially load-balanceable, survives disconnects.
- Con: large payloads (frames + audio latents) over every request hop;
  bandwidth heavy on multi-segment WebSocket sessions.

**B. Server-held session state** (internal reality)
- Continuation lives per-GPU; implicit between adjacent segments.
- Pro: zero client bandwidth; fast; matches today.
- Con: needs GPU affinity, no resume after disconnect, harder to scale horizontally.

**C. Hybrid** (recommended)
- Server-held is the default for streaming WebSocket sessions.
- Server exposes a `snapshot_state` message that returns the opaque
  payload form for migration/retry.
- Stateless HTTP endpoints always use round-trip opaque payloads.
- One serialization format underlies both surfaces.

### Decision: **C (Hybrid)**

Rationale: matches both internal streaming use (server-held, fast) and
stateless API use (client-round-trip, resumable). Cost is one serialization
layer that serves both.

### Implications
- `ContinuationState.kind` identifies the payload schema
  (e.g. `"ltx2.v1"`).
- `ContinuationState.payload` must cover:
  - trailing conditioning frames (or a tensor reference)
  - audio latents (or a tensor reference)
  - segment index / rollout position
  - any model-specific conditioning metadata (e.g. audio sample rate,
    `video_position_offset_sec`)
- For large tensors, payload may reference a server-side blob by ID
  rather than inline everything.
- Streaming server has a `SessionStore` keyed by session ID that holds
  a typed `LTX2ContinuationState` object.
- `SessionStore.snapshot(session_id) -> ContinuationState` serializes
  the current state for export.
- `SessionStore.hydrate(state: ContinuationState) -> session_id` loads
  a state into a new session.
- Plan PR 7 expands to cover both surfaces and define the payload schema.

## Design Decision 2: Streaming server layout

### Options

- **A. `fastvideo/entrypoints/streaming/`** — parallel to
  `fastvideo/entrypoints/openai/`
- **B. `fastvideo/entrypoints/server/{stateless,streaming}/`** — reorg both
- **C. `fastvideo/streaming/`** — top-level package, not under entrypoints

### Decision: **A (parallel subpackage)**

Rationale: lowest-friction, no existing code moves, both servers share
the same `fastvideo/entrypoints/*` namespace and import style. Shared
utilities can be factored into `fastvideo/entrypoints/server_common/`
later if needed. Option B creates churn across every openai/ import for
marginal organizational win.

### Target layout

```text
fastvideo/entrypoints/
├── openai/                        # existing: stateless HTTP POST
│   ├── api_server.py
│   ├── video_api.py
│   ├── image_api.py
│   ├── common_api.py
│   ├── protocol.py
│   ├── state.py
│   ├── stores.py
│   └── utils.py
├── streaming/                     # NEW: session WebSocket
│   ├── server.py                  # FastAPI + WebSocket entry
│   ├── session.py                 # session lifecycle, state machine
│   ├── session_store.py           # typed session state + snapshot/hydrate
│   ├── protocol.py                # JSON WebSocket message schemas
│   ├── stream.py                  # fMP4 encoding (av_fmp4 mode)
│   ├── gpu_pool.py                # subprocess workers
│   ├── worker.py                  # per-GPU worker loop
│   ├── continuation.py            # typed LTX2 state payload
│   ├── session_init_image.py
│   ├── session_logger.py
│   ├── mock_server.py
│   ├── prompt/
│   │   ├── enhancer.py            # provider-agnostic prompt ops
│   │   ├── rewrite.py
│   │   ├── safety.py              # optional fasttext
│   │   ├── payload.py             # rewrite payload builder
│   │   └── providers/
│   │       ├── base.py            # LLMProvider protocol
│   │       ├── cerebras.py
│   │       ├── cerebras_ifm.py
│   │       └── groq.py
│   └── router/                    # or separate top-level package
│       ├── main.py
│       └── registry.py
├── cli/                           # existing
└── video_generator.py             # existing
```

### Config integration

`ServeConfig` gets an optional `streaming: StreamingConfig | None` field:

```python
@dataclass
class StreamingConfig:
    session_timeout_seconds: int = 300
    generation_segment_cap: int = 6
    stream_mode: Literal["av_fmp4", "legacy_jpeg"] = "av_fmp4"
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    pool: GpuPoolConfig = field(default_factory=GpuPoolConfig)
    prompt: PromptEnhancerConfig | None = None
    safety: PromptSafetyConfig | None = None

@dataclass
class GpuPoolConfig:
    num_workers: int | None = None  # default: CUDA_VISIBLE_DEVICES count
    enable_audio_reencode: bool = True
    conditioning_num_frames: int = 9
    conditioning_end_offset: int = 0

@dataclass
class PromptEnhancerConfig:
    provider: Literal["cerebras_ifm", "cerebras", "groq"] = "cerebras_ifm"
    model: str = "gpt-oss-120b"
    timeout_ms: int = 20000
    system_prompt_dir: str | None = None  # hot-reloadable system prompts

@dataclass
class PromptSafetyConfig:
    enabled: bool = False
    classifier_path: str | None = None
```

## Design Decision 3: LLM provider abstraction

### Problem

`prompt_enhancer.py` (69KB) hard-codes three providers (cerebras_ifm,
cerebras, groq) with provider-specific request/response handling
scattered throughout. Upstreaming as-is locks FastVideo to those three
providers and couples the prompt operations to their response shapes.

### Shape

Introduce an `LLMProvider` protocol:

```python
from typing import Protocol, AsyncIterator, Literal
from dataclasses import dataclass

@dataclass
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    timeout_ms: int | None = None

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    latency_ms: float
    fallback_used: bool = False

class LLMProvider(Protocol):
    name: str
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
```

### Decision: **Protocol + built-in implementations for cerebras, cerebras_ifm, groq**

Rationale: keeps the prompt enhancer free of provider-specific branching;
users (and future OpenAI/Anthropic/local additions) can register their
own provider without modifying FastVideo. Each built-in provider is
100-200 LOC; the enhancer becomes provider-agnostic prompt orchestration.

### Implications
- `prompt_enhancer.py` splits into `enhancer.py` (prompt operations) +
  `providers/` (IO).
- Config moves from scattered env vars to typed `PromptEnhancerConfig`
  under `ServeConfig.streaming.prompt`.
- Hot-reloadable system prompts stay — exposed as a management endpoint
  on the streaming server.
- Fallback behavior (retry across providers in priority order) moves
  into the enhancer layer, orthogonal to provider implementations.

## Design Decision 4 preamble: what Dynamo expects from FastVideo

Dynamo's backend pattern (observed in
`dynamo/components/src/dynamo/sglang/` and confirmed by PR #7544) is a
**pure Python import** pattern. Dynamo owns the backend subpackage in its
own repo; FastVideo only needs to expose a stable, typed, aggregated
and (later) streaming generation surface.

### Contract surface Dynamo consumes

| Surface | Shape | Notes |
|---|---|---|
| Constructor | `VideoGenerator.from_pretrained(model_path, **typed_kwargs)` | Already exists; `typed_kwargs` must be a stable subset from `GeneratorConfig` — no flat LTX2 legacy kwargs. |
| Sync execution | `generator.generate_video(request: GenerationRequest) -> VideoResult` | Aggregated mode; Dynamo wraps in `asyncio.to_thread` under an `asyncio.Lock`. |
| Async execution | `generator.generate_async(request: GenerationRequest) -> AsyncGenerator[VideoEvent, None]` | Needed for: (a) streaming server fMP4 chunks; (b) future Dynamo disaggregation. Events: `Progress`, `Partial?`, `Final`. |
| Typed request | `fastvideo.api.GenerationRequest`, `SamplingConfig`, `InputConfig` | Stable import path; Dynamo's adapter builds this from `NvCreateVideoRequest` + `VideoNvExt`. |
| Typed result | `VideoResult` with `video_bytes` or tensor frames, plus `ContinuationState?` | Must be picklable / JSON-serializable enough for Dynamo RPC. |
| Continuation | `ContinuationState(kind, payload)` with schema-versioned payloads | Used by FastVideo's session store today; tomorrow by Dynamo disaggregated workers. |
| Health check input | `VideoGenerator.default_health_check_request() -> GenerationRequest` | Minimal 256x256 / 8 frames / 1 step; lets Dynamo's `FastVideoHealthCheckPayload.to_dict()` produce the Dynamo `health_check_payload` kwarg without knowledge of FastVideo internals. |
| Config dump | `GeneratorConfig.to_dict()` / `ServeConfig.to_dict()` | Dynamo calls `dynamo.common.config_dump.dump_config(path, config)` at worker start; we already have `config_to_dict()`. |

### Request/response mapping (Dynamo ↔ FastVideo)

Dynamo's video protocol (`NvCreateVideoRequest` / `NvVideosResponse`):

```
NvCreateVideoRequest          ->  fastvideo.api.GenerationRequest
  prompt                      ->    sampling.prompt
  size="WxH"                  ->    sampling.width, sampling.height
  seconds                     ->    (seconds * nvext.fps) -> sampling.num_frames
  input_reference             ->    input.image_path / input.video_path
  nvext.fps                   ->    sampling.fps
  nvext.num_frames            ->    sampling.num_frames (overrides seconds*fps)
  nvext.num_inference_steps   ->    sampling.num_inference_steps
  nvext.guidance_scale        ->    sampling.guidance_scale
  nvext.seed                  ->    sampling.seed
  nvext.negative_prompt       ->    sampling.negative_prompt
  response_format             ->    (handled by adapter at output)

VideoFinalEvent               ->  NvVideosResponse
  video_bytes                 ->    data[0].b64_json  (if response_format=b64_json)
  video_url (after upload)    ->    data[0].url       (if response_format=url)
  metadata.inference_time_s   ->    inference_time_s
```

All fields already exist (or will exist after PR 6 expansion) on
FastVideo's typed schema. No FastVideo changes required beyond what the
rest of this plan already covers **except**:

1. `generate_async` must exist (new in PR 7.10).
2. `default_health_check_request()` helper (new in PR 7.10).
3. The sync `generate_video(request=...)` path must be reachable without
   extra wrapping (exists since PR 2; confirm stability).

### Where the Dynamo subpackage lives

The Dynamo-side integration (`FastVideoHandler`, `register_fastvideo_model`,
`FastVideoHealthCheckPayload`, args parsing, main.py, Dockerfile,
request/response mapping) lives **entirely in the Dynamo repo** at
`components/src/dynamo/fastvideo/`, matching the pattern used by vllm
and sglang. FastVideo does **not** host any Dynamo-related subpackage,
Dynamo dependency, or Dynamo-specific CLI. FastVideo's only obligation
is to expose a clean, stable, typed Python API that Dynamo's backend
package can import.

## Design Decision 4: Dynamo as first-class backend target

### Problem

PR #7544 (closed) shows two frictions with the pre-refactor API:

1. **Flat legacy kwargs** — the Dynamo handler had to know about
   LTX2-specific flat names.
2. **Sync-only generation** — Dynamo's async handler wrapped
   `generator.generate(...)` in `asyncio.to_thread` under a lock; no
   progress streaming, no disaggregation path.

The refactor's stateless OpenAI server, WebSocket streaming server, and
Dynamo backend all want the same thing: **a typed async API that yields
progress events and a typed final result**. If we build it once in
`VideoGenerator`, all three adapters become thin.

### Options

**A. Keep sync-only, each adapter wraps**
- Simple; matches PR #7544.
- Con: streaming server needs its own async runner; Dynamo loses progress
  streaming; no path to disaggregation.

**B. Add async event stream to `VideoGenerator`**
- `generate_async(request) -> AsyncGenerator[VideoEvent, None]`.
- Sync `generate_video` becomes a thin `asyncio.run` wrapper internally.
- Pro: one canonical execution API; streaming server, OpenAI server,
  and Dynamo all consume events directly.
- Con: larger delta in `VideoGenerator` — must thread async through the
  pipeline step loop.

**C. Queue-based `generate(request, event_cb)` callback**
- Middle ground; callback receives events.
- Pro: no async rewrite needed.
- Con: callers have to invert control; awkward for Dynamo's async
  handler.

### Decision: **B (async event stream)**

Rationale: one substrate serves all three consumers. The cost is a
`generate_async` implementation that runs the pipeline step loop in a
thread and bridges events back via an asyncio queue — standard pattern,
limited surface area.

### Implications

- New PR 7.10 adds `generate_async` on `VideoGenerator` with three event
  types: `VideoProgressEvent(step, total_steps, stage)`,
  `VideoPartialEvent(frames_ndarray, index)` (optional; emitted only in
  the streaming path), `VideoFinalEvent(video_bytes_or_tensor, metadata,
  continuation_state?)`.
- Sync `generate_video(request=...)` becomes `asyncio.run(...)` over
  `generate_async`, collecting events and returning the final.
- Streaming server's fMP4 encoder consumes `VideoPartialEvent` frames
  directly, never re-decoding through disk.
- Dynamo adapter consumes `generate_async` and yields one
  `NvVideosResponse` per `VideoFinalEvent` (aggregated mode; ignores
  intermediate events today; can surface progress via Dynamo's
  status/progress fields in the future).
- `ContinuationState` can be attached to `VideoFinalEvent.metadata`,
  giving Dynamo a first-class way to surface state for disaggregation
  later.
- Stable public exports: `from fastvideo import VideoGenerator`;
  `from fastvideo.api import GenerationRequest, SamplingConfig,
  ContinuationState, VideoResult, VideoEvent`.
- No Dynamo subpackage, dep, or CLI lives in FastVideo. The adapter
  (`NvCreateVideoRequest ↔ GenerationRequest` mapping, handler,
  registration) lives entirely in the Dynamo repo at
  `components/src/dynamo/fastvideo/`.

### Constraints this adds to earlier PRs

- **PR 6** (typed LTX2 kwargs): every flat kwarg must have a typed home
  **reachable from `GeneratorConfig`**, so Dynamo can construct the
  generator without importing internal compat paths.
- **PR 7** (continuation state): `ContinuationState.payload` must be
  JSON/YAML serializable (no raw torch tensors inline; use blob
  indirection) so it survives Dynamo RPC transport.
- **PR 7.5** (streaming skeleton): consume `generate_async` rather than
  re-implementing a progress loop around `generate_video`.
- **PR 2/3/4 already landed**: the typed request shape is fixed and
  matches Dynamo's mapping needs — no backtracking required.

## Revised PR sequence (PR 5 onwards)

PRs 0-4 are unchanged and already landed. PR 5 is narrowed; PRs 5.5-7.9
are new inserts; PRs 8-13 are reshaped or kept.

| # | Title | Change | Key deliverables |
|---|---|---|---|
| **5** | Stateless `ServeConfig.default_request` merge | **Narrowed.** Wire typed default-request into `fastvideo/entrypoints/openai/`. | `_merge_default_request` helper, validated-against-preset, tests for default+user-override precedence |
| **5.5** | Server architecture split | **NEW.** Introduce `fastvideo/entrypoints/streaming/` subpackage skeleton. No behavior change. | Empty subpackage + stub server.py; CLI subcommand `fastvideo streaming-serve` (raises NotImplementedError); doc on layout |
| **6** | LTX2 public preset + stage overrides + config colocation | **Expanded.** Also add typed replacements for every flat kwarg used by internal `gpu_pool.py`. | `ltx2_two_stage` preset, `LTX2RefineStageOverride`, `CompileConfig` field types, typed `FP4Config` integration, colocation |
| **7** | Continuation state (public + session) | **Expanded.** Define both opaque payload AND server-held session store. | `ContinuationState.payload` schema, `LTX2ContinuationState` typed subclass, `SessionStore` interface, snapshot/hydrate APIs |
| **7.5** | Streaming server skeleton | **NEW.** Minimum viable WebSocket server: session lifecycle, JSON messages, fMP4 output, single-generator. | `server.py`, `session.py`, `protocol.py`, `stream.py` (fMP4), typed `StreamingConfig` |
| **7.6** | GPU pool upstream | **NEW.** Upstream `gpu_pool.py` with typed config boundary. | `gpu_pool.py`, `worker.py`, job queue, session-to-GPU binding, session timeout handling |
| **7.7** | Prompt enhancer upstream | **NEW.** Upstream `prompt_enhancer.py` with `LLMProvider` abstraction. | `prompt/enhancer.py`, `prompt/providers/{base,cerebras,cerebras_ifm,groq}.py`, hot-reloadable system prompts |
| **7.8** | Streaming auxiliaries | **NEW.** Small, isolated. | `prompt/safety.py`, `session_init_image.py`, `prompt/rewrite.py`, `session_logger.py`, `mock_server.py` |
| **7.9** | Router upstream | **NEW.** Multi-replica load balancer + WS proxy. | `streaming/router/` (or separate top-level package), health checks, WS proxy |
| **7.10** | Dynamo backend contract | **NEW.** Add `VideoGenerator.generate_async` event stream + `default_health_check_request()` helper. FastVideo exposes the async API only; the Dynamo backend package (handler, adapter, registration) lives entirely in the Dynamo repo at `components/src/dynamo/fastvideo/`. Streaming server (PR 7.5) and Dynamo backend both consume the same async API. | `generate_async` with `VideoProgressEvent`/`VideoPartialEvent`/`VideoFinalEvent`; sync `generate_video` becomes a thin wrapper; contract tests against a mock Dynamo-style handler that imports only public FastVideo APIs |
| **8** | Internal-UI ↔ public-server contract docs & tests | **Reframed.** Was "Dreamverse Server Adaptation Layer." Also covers Dynamo integration reference. | WebSocket protocol reference, contract tests, migration examples, Dynamo adapter example that upstream PR can copy verbatim |
| **9** | LongCat preset migration + colocation | **Keep.** | Stage overrides, colocation |
| **10** | Hunyuan15 SR preset migration + colocation | **Keep.** | Stage overrides, SR field migration POC, colocation |
| **11** | SSIM / perf test migration | **Keep.** Now blocked on PR 6 expansion. | Typed API migration of golden tests |
| **12** | Docs + examples | **Keep, expand.** | Streaming server docs now part of scope |
| **13** | Deprecation + cleanup | **Keep, expand.** | Also deprecate flat kwargs that internal gpu_pool uses today |

Total PR count: 13 → ~20 (13 original + 5 streaming-upstream inserts +
1 architecture split + 1 Dynamo contract). Each new PR is small and
self-contained because the streaming components are already cleanly
separated in the internal repo, and the Dynamo contract rides on top of
the async API that the streaming server already needs.

## Open questions

1. **Router: in-repo or separate package?** — It's orthogonal to inference;
   in-repo couples deploy cycles, separate leaves FastVideo cleaner.
   Recommendation: separate package `fastvideo-router/` or
   `fastvideo/contrib/router/`; defer final call to PR 7.9.
2. **Session ID authority** — internal uses ad-hoc client IDs.
   Recommendation: server-generated UUID, accept externally provided
   session ID only for resume flows.
3. **Torch compile kwargs typing** — `CompileConfig.kwargs: dict[str, Any]`
   today accepts `mode`, `backend`, `fullgraph`, `dynamic`. Options: keep
   as opaque dict; fully type; hybrid (type the common four + allow
   extras). Recommendation: hybrid, type common fields.
4. **Prompt safety / fasttext dependency** — heavy for users who don't
   need it. Recommendation: ship as optional extra
   `pip install fastvideo[prompt-safety]`.
5. **Audio-specific tensor payloads** — `ltx2_audio_clean_latent`,
   `ltx2_audio_denoise_mask`, `ltx2_audio_latents` are not in the current
   public schema. PR 7 should classify them (probably as opaque fields
   inside `LTX2ContinuationState.payload`, not top-level sampling fields).
6. **Batching behavior** — internal `test_batching.py` suggests batching
   is exercised. Scope this into PR 7.5 or defer to a post-cleanup perf PR?
7. ~~**Dynamo subpackage home**~~ — **Resolved.** No Dynamo code lives
   in FastVideo. The full backend package (handler, adapter,
   registration, health check) is owned by the Dynamo repo at
   `components/src/dynamo/fastvideo/`, same pattern as vllm/sglang.
   FastVideo only guarantees the public API contract listed above.
8. **Disaggregation readiness** — PR #7544 is aggregated-only. Our
   `ContinuationState` hybrid already supports a future prefill/decode
   split (prefill yields state; decode hydrates it). Should PR 7.10
   explicitly validate that `ContinuationState` survives round-trip
   through a Dynamo-style RPC (pickle or JSON), even though Dynamo
   isn't using it today? Recommendation: yes; cheap contract test that
   prevents drift.
9. **Dynamo progress/status passthrough** — `NvVideosResponse` has
   `status` and `progress` fields. Should PR 7.10's handler contract
   emit intermediate `NvVideosResponse` chunks keyed off
   `VideoProgressEvent`, or stay aggregated-final-only to match PR
   #7544? Recommendation: stay aggregated-final for PR 7.10; revisit
   after Dynamo clarifies their streaming/progress semantics.

## Immediate path forward

1. Land `will/api_5` cleanup commits — **done** (`e03ca7d9`, `41f93179`
   force-pushed without Claude co-author).
2. Review this plan with a human — commit the doc to capture the state.
3. Execute PR 5 (narrow stateless merge) and PR 5.5 (subpackage split)
   in parallel. Both small; both unblock the streaming upstream that
   follows.
4. Start PR 6 expansion (typed replacements for flat LTX2 kwargs) as the
   critical path for PR 7.6 (gpu_pool upstream).
