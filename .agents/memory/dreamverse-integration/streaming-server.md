# Streaming Server Upstream — PRs 5.5 → 7.10

The `FastVideo-internal/ui/ltx2-streaming/server/` stack is being
upstreamed into public FastVideo at `fastvideo/entrypoints/streaming/`.
In parallel, FastVideo is becoming a first-class Dynamo backend (same
tier as vllm, sglang, trtllm). This file covers both threads since they
share `generate_async` as the substrate.

For PR sequence/status see [pr-roadmap.md](pr-roadmap.md). For the
Dreamverse-side adoption see [cross-repo-surfaces.md](cross-repo-surfaces.md).

**Last updated:** 2026-05-03.

## What's being upstreamed

| Internal path | Size | Role | Public target |
|---|---|---|---|
| `server/main.py` | 94 KB | FastAPI + WebSocket, session lifecycle, segment orchestration | `fastvideo/entrypoints/streaming/server.py` + handlers |
| `server/gpu_pool.py` | 66 KB | GPU orchestration, subprocess workers | `fastvideo/entrypoints/streaming/gpu_pool.py` |
| `server/prompt_enhancer.py` | 69 KB | LLM orchestration (cerebras_ifm, cerebras, groq) | `fastvideo/entrypoints/streaming/prompt/` package |
| `server/mock_server.py` | 45 KB | Mock backend for dev/tests | `fastvideo/entrypoints/streaming/mock_server.py` |
| `server/prompt_safety.py` | 7 KB | Optional fasttext-gated prompt safety | `prompt/safety.py` |
| `server/session_init_image.py` | 3 KB | i2v init image handling | `streaming/session_init_image.py` (PR 7.5, already public) |
| `server/rewrite_prompt_payload.py` | 3 KB | Rewrite flow payload builder | `prompt/rewrite.py` |
| `server/session_logger.py` | 1 KB | Session JSONL logs | `streaming/session_logger.py` |
| `server/config.py` | 9 KB | Env-driven server config | typed `ServeConfig.streaming` extensions |
| `router/main.py` | 27 KB | Multi-replica load balancer + WS proxy | `fastvideo/entrypoints/streaming/router/` |
| `slurm/` | — | Deployment scripts | Stays internal |

Frontend clients (`client/`, `prod-ui/`) stay in the internal repo.

## Four design decisions that shape the upstream

### D-1: Continuation model — Hybrid (server-held + opaque client-round-trip)

Streaming WebSocket sessions hold continuation per-GPU (matches today's
internal behavior, fast, zero client bandwidth). Stateless HTTP endpoints
use opaque round-trip payloads. Server exposes a `snapshot_state` message
that returns the opaque form for migration/retry.

One serialization layer underlies both surfaces.

Implementation: `SessionStore` (in-memory default, pluggable for
redis/etc.) keyed by session ID, holds typed `LTX2ContinuationState`.
- `snapshot(session_id) -> ContinuationState` exports for migration
- `hydrate(state: ContinuationState) -> session_id` loads state into new session

Payload schema covers: trailing conditioning frames (or tensor-blob ID),
audio latents (or blob ID), segment index, audio sample rate,
`video_position_offset_sec`, model-specific metadata.

Landed in PR 7. See [cross-repo-surfaces.md](cross-repo-surfaces.md) for
the full wire format.

### D-2: Streaming server layout — Parallel subpackage `fastvideo/entrypoints/streaming/`

Sits next to `fastvideo/entrypoints/openai/`. No existing code moves.
Both servers share the `entrypoints/*` namespace. Shared utilities can be
factored into `fastvideo/entrypoints/server_common/` later if needed.

### D-3: LLM provider abstraction — `LLMProvider` protocol + built-in providers

`prompt_enhancer.py` (69 KB) hard-coded three providers (cerebras_ifm,
cerebras, groq) with provider-specific request/response handling
scattered throughout. Upstreaming as-is would lock FastVideo to those
providers.

Protocol shape:

```python
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

PR 7.7 ships built-in providers for cerebras, groq. **Public Literal
currently restricts to `Literal["cerebras", "groq"]`** — `cerebras_ifm`
is internal-only and remains environment-driven on `dreamverse-server`.
See [open-threads.md](open-threads.md) follow-up #3.

Hot-reloadable system prompts via management endpoint. Sequential
fallback across providers in priority order — race-based fallback (the
internal optimization) deferred per [decisions-log.md](decisions-log.md)
D-3.

### D-4: Dynamo as first-class backend target — async event stream

PR ai-dynamo/dynamo#7544 (closed draft) showed two frictions:

1. Flat legacy kwargs — Dynamo handler had to know LTX-2-specific names.
2. Sync-only generation — Dynamo wrapped `generator.generate(...)` in
   `asyncio.to_thread` under a lock; no progress streaming, no
   disaggregation path.

Decision: **add `generate_async`** as the canonical execution API.

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
    frames: np.ndarray           # (num_frames, H, W, 3)
    index: int                   # monotonic chunk index

@dataclass
class VideoFinalEvent:
    video_bytes: bytes | None
    tensor: torch.Tensor | None
    metadata: dict[str, Any]
    continuation_state: ContinuationState | None
```

The sync `generate_video(request=...) -> VideoResult` becomes a thin
`asyncio.run` wrapper over `generate_async` that collects events and
returns the final.

**Three consumers, one substrate:**

| Consumer | Transport | Request shape | State |
|---|---|---|---|
| Stateless OpenAI (`fastvideo/entrypoints/openai/`) | HTTP POST | `GenerationRequest` merged onto `ServeConfig.default_request` | Stateless; opaque payload |
| Streaming WebSocket (`fastvideo/entrypoints/streaming/`) | WebSocket JSON + binary fMP4 | `GenerationRequest` per segment, session-scoped | Server-held; per-GPU continuation cache |
| Dynamo native backend (`ai-dynamo/dynamo/components/src/dynamo/fastvideo/`) | Dynamo RPC endpoint | `NvCreateVideoRequest` ↔ adapter ↔ `GenerationRequest` | Aggregated today; future disaggregated via `ContinuationState` |

**FastVideo does NOT host any Dynamo code.** The full backend package
(`args.py`, `main.py`, `backend.py`, `register.py`, `health_check.py`)
lives entirely in the Dynamo repo at `components/src/dynamo/fastvideo/`,
matching the vllm/sglang pattern. FastVideo's only obligation is the
stable public Python API.

PR 7.10 lands the FastVideo-side contract. Dynamo backend code lives in
ai-dynamo/dynamo (next iteration of #7544 reopens against PR 8 reference
docs).

## Target package layout

```
fastvideo/entrypoints/
├── openai/                        # existing: stateless HTTP POST
├── streaming/                     # NEW: session WebSocket
│   ├── server.py                  # FastAPI + WebSocket entry
│   ├── session.py                 # session lifecycle, state machine
│   ├── session_store.py           # typed session state + snapshot/hydrate
│   ├── protocol.py                # JSON WebSocket message schemas
│   ├── stream.py                  # fMP4 encoding (av_fmp4 mode)
│   ├── gpu_pool.py                # subprocess workers (PR 7.6)
│   ├── worker.py                  # per-GPU worker loop
│   ├── continuation.py            # typed LTX2 state payload
│   ├── session_init_image.py
│   ├── session_logger.py
│   ├── mock_server.py
│   ├── prompt/
│   │   ├── enhancer.py            # provider-agnostic prompt ops
│   │   ├── rewrite.py
│   │   ├── safety.py              # optional fasttext
│   │   └── providers/
│   │       ├── base.py            # LLMProvider protocol
│   │       ├── cerebras.py
│   │       ├── cerebras_ifm.py
│   │       └── groq.py
│   └── router/
│       ├── main.py
│       └── registry.py
├── cli/
└── video_generator.py
```

## Typed config integration

`ServeConfig` gets an optional `streaming: StreamingConfig | None`:

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
    provider: Literal["cerebras", "groq"] = "cerebras"  # cerebras_ifm pending
    model: str = "gpt-oss-120b"
    timeout_ms: int = 20000
    system_prompt_dir: str | None = None  # hot-reloadable

@dataclass
class PromptSafetyConfig:
    enabled: bool = False
    classifier_path: str | None = None
```

## `build_app` route contract — open follow-up

Today `fastvideo.entrypoints.streaming.server.build_app` exposes only:

- `GET /health`
- `WS /v1/stream`

The Dreamverse Next.js shell expects these additional routes that the
upstream plan (and Dreamverse FE today) require:

| Route | Owner per upstream plan | Status |
|---|---|---|
| `GET /healthz` | Streaming-server-side health (FastVideo) | 🔴 NOT YET MIGRATED |
| `GET /readyz` | Streaming-server-side health (FastVideo) | 🔴 NOT YET MIGRATED |
| `GET /status` | Streaming-server-side health (FastVideo) | 🔴 NOT YET MIGRATED |
| `GET /curated-presets` | Operator-side surface (Dreamverse) | 🟡 stays in Dreamverse, FE feature-detects |
| `POST /curated-presets/append` | Operator-side surface (Dreamverse) | 🟡 stays in Dreamverse |
| `GET /prompt-system-config` | Operator-side surface (Dreamverse) | 🟡 stays in Dreamverse |
| Devtools routes | Dreamverse-only | 🟡 stays in Dreamverse |

Until the three health routes migrate into FastVideo's `build_app`, the
`BE_FLAVOR=fastvideo` flavor of `launch_demo.sh` is a "diagnostic" flavor
only (verifies typed serve-config path) — not FE-compatible. See
[open-threads.md](open-threads.md) follow-up #1.

The streaming-upstream plan listed `/healthz`, `/readyz`, `/status`,
`/ws` as the contract that the upstream of `realtime/` → `streaming/`
must preserve. They were deferred from PR 7.5's MVP.

## PR 7.5 status — open as #1251

Single-generator WebSocket end-to-end shipped (8 commits):

1. `feat(streaming): protocol schemas + session state machine`
2. `feat(streaming): fMP4 encoder + session init-image persistence`
3. `feat(streaming): single-generator WebSocket server entry`
4. `test(streaming): server lifecycle + protocol + fMP4 coverage`
5. `docs(streaming): server contract spec`
6. `fix(streaming): restore missing-streaming-block guard + retire stub-era test`
7. `simplify(streaming): review follow-ups (idle timeout via asyncio.wait_for, _send_error helper, _cleanup_session, Protocol-typed generator, cleanup-on-disconnect)`
8. `fix(streaming): enforce idle timeout on receive_json + flag generator-cancellation gap (TODO → PR 7.10)`

Deferred TODOs (intentionally) blocking on PR 7.10:

- **Per-step progress events** — only terminal `step_complete` today;
  needs `generate_async` for per-step `VideoProgressEvent` emission.
- **Mid-segment cancellation on client disconnect** — TODO marker in
  `server.py` near `pool.run`. Needs `generate_async`'s cancellation
  propagation.

## PR 7.6 status — branch ready, not yet PR'd

`will/api_7.6` (5 commits, rebased on 7.5):

1. `feat [7.6/n]: GPU pool manager with typed worker boundary`
2. `refactor [7.6/n]: route streaming server through GpuPool`
3. `test [7.6/n]: GPU pool coverage (in-process + subprocess)`
4. `fix(streaming): restore missing asyncio import in server` (rebase fixup)
5. `feat(streaming): extract worker.py and add two-segment warmup`

Tests: 17/17 gpu_pool tests + 89/89 streaming tests green.

Ships:
- `GpuPool` ABC + `InProcessGpuPool` + `SubprocessGpuPool` +
  `PoolAssignment` / `PoolHealth` / `PoolAcquireTimeout`
- `worker.py` — per-GPU `worker_main` and two-segment warmup helper
- Subprocess startup uses typed `GeneratorConfig`, NOT flat kwargs
- Session-to-GPU binding with timeout + queue for contention
- Two-segment startup warmup per worker (segment 1 fresh + segment 2
  with returned `ContinuationState` so both compile branches are primed)
- `SessionStore` (from PR 7) wired for per-GPU continuation cache

Deferred to PR 7.10:

- **Audio re-encode (`LTX2AudioEncoder`, `AudioProcessor`)**: internal
  `_re_encode_audio` runs *inside* the per-step streaming loop
  (`_stream_av_fmp4_events` / `do_step_ltx2`). The whole-segment
  `pool.run()` path PR 7.6 ships doesn't need it. Re-encode is a
  per-step streaming concern that belongs with `generate_async`.
- **Deprecate `VideoGenerator.from_pretrained(**flat_kwargs)`**: belongs
  with PR 13 cleanup.

## PR 7.10 — the unlock PR

PR 7.10 adds `generate_async` and closes three open threads
simultaneously:

- Q-5 / D-5: audio re-encode for cross-segment continuity
- Q-9: Dynamo progress passthrough
- PR 7.5's mid-segment cancellation TODO (client disconnect →
  `asyncio.CancelledError` → GPU work stops)

Plus health-check helper:

```python
def default_health_check_request(self) -> GenerationRequest: ...
# Returns 256x256, 8 frames, 1 step. Lets Dynamo's
# FastVideoHealthCheckPayload.to_dict() produce a Dynamo
# health_check_payload kwarg without knowledge of FastVideo internals.
```

Stable public exports:

```python
from fastvideo import VideoGenerator
from fastvideo.api import (
    GenerationRequest, SamplingConfig, ContinuationState,
    VideoResult, VideoEvent,
    VideoProgressEvent, VideoPartialEvent, VideoFinalEvent,
)
```

Streaming server (PR 7.5) gets rewired to consume `generate_async`
directly — no wrapper duplication.

## Dynamo request/response mapping

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
  response_format             ->    (handled at adapter's output stage)

VideoFinalEvent               ->  NvVideosResponse
  video_bytes                 ->    data[0].b64_json  (response_format=b64_json)
  uploaded URL                ->    data[0].url       (response_format=url)
  metadata.inference_time_s   ->    inference_time_s
  continuation_state          ->    (reserved for future disaggregation)
```

## Open questions

1. **Router placement** — in-tree at `fastvideo/entrypoints/streaming/router/`
   (current implementation per PR 7.9) or separate package
   `fastvideo-router/` / `fastvideo/contrib/router/`. Effectively
   resolved in-tree by the PR 7.9 implementation.
2. **Session ID authority** — server-generated UUID; accept externally
   provided session ID only for resume flows.
3. **Disaggregation readiness contract test** — should PR 7.10 validate
   `ContinuationState` survives round-trip through Dynamo-style RPC
   (pickle or JSON), even though Dynamo isn't using it today?
   Recommended: yes; cheap regression guard.
4. **Dynamo progress/status passthrough** — should PR 7.10's handler
   contract emit intermediate `NvVideosResponse` chunks keyed off
   `VideoProgressEvent`, or stay aggregated-final-only? Recommended:
   stay aggregated-final for PR 7.10; revisit after Dynamo clarifies.
5. **`video_position_offset_sec` semantics** — see [decisions-log.md](decisions-log.md)
   open question; needs decision before PR 7.6 emits state.
6. **`SessionStore` / `BlobStore` lifecycle** — eviction, TTL, blob-drop
   on state replacement; defer to PR 7.5 design pass.
