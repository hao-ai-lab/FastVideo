# Cross-Repo Surfaces — Dreamverse + Dynamo

How Dreamverse consumes FastVideo today, what's already shared, what's
ad hoc, and what migrations land alongside each PR. Plus the Dynamo
backend contract.

For the streaming-server side see [streaming-server.md](streaming-server.md).
For the API design see [design.md](design.md). For PR sequence see
[pr-roadmap.md](pr-roadmap.md).

**Last updated:** 2026-05-03.

## The three surfaces

Dreamverse depends on FastVideo across three surfaces (in order of
stability):

1. **Pipeline construction** (stable)
2. **Realtime runtime** (in flight: PRs 7.5/7.6)
3. **Continuation state** (PR 7 typed; PR 7.6 wires server-held)

## Surface 1: Pipeline construction (stable)

`Dreamverse/server/video_generation.py:VideoGenerationWorker` calls
`VideoGenerator.from_pretrained(...)`.

After PR 6 the typed `GeneratorConfig` path exists; **as of `d80c2a8`
(May 2)** Dreamverse migrated to the typed path:

| Dreamverse usage | FastVideo public surface (post-PR 6) |
|---|---|
| `VideoGenerator.from_pretrained(model_path, ltx2_refine_enabled=…, …)` | `VideoGenerator.from_pretrained(config=GeneratorConfig(...))` |
| Flat `torch_compile_kwargs={…}` dict | `engine.compile.{backend,fullgraph,mode,dynamic,extras}` |
| `ltx2_vae_tiling=True` | `pipeline.vae_tiling=True` |
| `ltx2_refine_*` family | `pipeline.preset_overrides.refine.*` + `pipeline.components.upsampler_weights` |
| `enable_torch_compile_text_encoder` | `engine.compile.text_encoder_enabled` |

Refine knobs moved from `ltx2_refine_*` flat kwargs into
`preset_overrides["refine"]`. **The in-memory `pipeline_config` pin**
(`dit_config.quant_config = NVFP4Config()`) keeps using the legacy
`experimental["pipeline_config"]` carrier because typed
`transformer_quant: "NVFP4"` doesn't yet support setting
`layer_profile` (see [open-threads.md](open-threads.md) follow-up #4 +
[quantization.md](quantization.md)).

Legacy flat-kwarg path stays supported via `compat.py`; migration is
opt-in. PR 13's deprecation warnings are the eventual nudge.

## Surface 2: Realtime runtime (in flight: PRs 7.5–7.6)

`Dreamverse/server/runtime/factory.py` selects a runtime backend at
process start:

```python
def create_runtime_pool() -> RuntimePool:
    if os.getenv("FASTVIDEO_REALTIME_BASE_URL"):
        return FastVideoRealtimePool(base_url=..., ws_url=..., default_model_id=...)
    return GPUPool(get_available_gpus())  # in-process, wraps
                                            # fastvideo.entrypoints.realtime.local_runtime
```

Both backends speak the same `RuntimePool` / `RuntimeSlot` Protocol
(`server/runtime/interfaces.py`):

- `acquire(client_id, websocket=None) -> (gpu_id, RuntimeSlot)`
- `release(client_id)`
- `RuntimeSlot.{join_user, user_step, leave_user, register_stream_queue, ...}`

Today both impls reach into FastVideo-internal's
`fastvideo.entrypoints.realtime.local_runtime` (which exposes
`RealtimeRuntimeConfig`, `GPUPool`, `GPUSlot`). The remote backend talks
HTTP+WS to a separately-deployed runtime of the same shape.

**Contract that PR 7.5/7.6 must preserve:**

- `RealtimeRuntimeConfig` accepts `model_registry`, `default_model_id`,
  `default_height/width/num_frames/fps/num_inference_steps/guidance_scale/seed/negative_prompt`,
  `default_ltx2_image_crf`, `startup_warmup_{enabled,prompt,timeout_seconds}`.
- `GPUPool(gpu_ids: list[int], config: RealtimeRuntimeConfig)` constructor.
- `pool.initialize() / shutdown() / acquire() / release() / get_status()`.
- HTTP endpoints on the remote variant: `GET /healthz`, `GET /readyz`,
  `GET /status`, `WS /ws`. (Already match what
  `Dreamverse/server/routes/health.py` consumes.)

**These three health routes still need to migrate into FastVideo's
`build_app` to make `BE_FLAVOR=fastvideo` FE-compatible** — see
[streaming-server.md](streaming-server.md) "build_app route contract" +
[open-threads.md](open-threads.md) follow-up #1.

When PR 7.6 lands the upstream of `fastvideo/entrypoints/realtime/`,
Dreamverse should not need any code change unless the import path
renames. Decided: keep `streaming/` (post-PR-5.5 public name); ship
`realtime/__init__.py` as a re-export with `DeprecationWarning` for one
release cycle.

### Note on `default_ltx2_image_crf`

Dreamverse's `RealtimeRuntimeConfig` includes `default_ltx2_image_crf`.
The April 26 Dreamverse review (D-8) showed this getting passed to
`SamplingParam(...)` and **silently dropped** by the public schema. Post
`d80c2a8` (May 2 typed-config refactor), the migration target is
`request.stage_overrides.refine.image_crf` (per
[design.md](design.md) compatibility mapping table).

**Whether `d80c2a8` actually wired this through, or it's still latent,
is unverified.** See [open-threads.md](open-threads.md) item D-8.

## Surface 3: Continuation state (PR 7)

`Dreamverse/server/video_generation.py:89 ContinuationState` is
Dreamverse's hand-rolled per-session state holder. PR 7 introduced the
typed equivalent at
[`fastvideo/pipelines/basic/ltx2/continuation.py`](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/continuation.py).

### Field mapping

| Dreamverse | PR 7 `LTX2ContinuationState` | Notes |
|---|---|---|
| `video_images: list[PIL.Image]` | `video_frames: list[np.ndarray]` (uint8 H×W×3) | numpy is leaner; Dreamverse already round-trips PIL→numpy→PIL just to add noise |
| `audio_latents: torch.Tensor` `[B, C, T, mel]` | `audio_latents: torch.Tensor` (safetensors-serialized; bf16-safe) | unchanged shape; safetensors preserves bf16 |
| `LTX2_VIDEO_CONDITIONING_FRAME_IDX` (env) | `video_conditioning_frame_idx: int` | env constant → per-state field |
| `LTX2_VIDEO_CONDITIONING_STRENGTH` (env) | `video_conditioning_strength: float` | env constant → per-state field |
| `AUDIO_CONDITIONING_NUM_FRAMES` (env) | `audio_conditioning_num_frames: int` | env constant → per-state field |
| `AUDIO_CONDITIONING_STRENGTH` (env) | `audio_conditioning_strength: float` | env constant → per-state field |
| `audio_lps` (passed into `apply_audio`) | `audio_sample_rate: int \| None` | analogous; rename worth confirming with audio team |
| Computed `prefix_sec` per segment | `video_position_offset_sec: float` | **see open question below** |
| `segment_idx` (param to `apply_*`) | `segment_index: int` | per-state field |
| `VIDEO_CONTEXT_NOISE`, `AUDIO_CONTEXT_NOISE`, `ENABLE_AUDIO_COND` | not on state | runtime policy / regularization knobs, not portable session data |
| `apply_video / apply_audio / save_video / save_audio_latents / clear` | not on PR-7 state class | state is a pure data carrier; runtime owns lifecycle policy |

PR 7 is a strict superset of Dreamverse's data model **plus** lifts
several env globals into per-session typed fields.

### Lifecycle mapping

| Dreamverse pattern | `SessionStore` API |
|---|---|
| `self.continuation = ContinuationState()` per session | `state = session_store.snapshot(sid) or LTX2ContinuationState()` |
| `apply_video(req_kwargs, segment_idx)` + `apply_audio(req_kwargs, segment_idx, audio_lps)` | `state = session_store.snapshot(sid)`; runtime builds request from `state.video_frames` / `state.audio_latents` |
| `save_video(frames)` + `save_audio_latents(latents)` | runtime constructs new `LTX2ContinuationState`, `session_store.store(sid, ...)` |
| `clear()` at end of session | `session_store.drop(sid)` |

`SessionStore` and `BlobStore` ABCs ship with thread-safe in-memory
defaults (`InMemorySessionStore`, `InMemoryBlobStore`). Dreamverse can
adopt them as-is for the local runtime; remote runtimes can plug in
redis-backed implementations later.

### Wire format (HTTP/WS round-trip)

Dreamverse's `FastVideoRealtimePool` already speaks the realtime
runtime's HTTP+WS protocol. When PR 7.5/7.6 land state emission on the
server side, the on-the-wire payload is the public envelope:

```json
{
  "kind": "ltx2.v1",
  "payload": {
    "schema_version": 1,
    "segment_index": 3,
    "video_conditioning_frame_idx": 9,
    "video_conditioning_strength": 0.75,
    "audio_sample_rate": 24000,
    "audio_conditioning_num_frames": 5,
    "audio_conditioning_strength": 0.5,
    "video_position_offset_sec": 0.2,
    "video": {"frames_b64": ["..."]},
    "audio": {"safetensors_b64": "..."},
    "metadata": {}
  }
}
```

JSON-serializable end-to-end; safetensors blob preserves audio dtype
(incl. bf16). For payloads above the inline threshold a `BlobStore`
indirection replaces the b64-encoded body with `{"blob_id": "..."}`;
the blob itself stays inside the runtime that produced it.

## Migration plan per PR

| PR | Dreamverse action |
|---|---|
| PR 6 (landed) | Typed `GeneratorConfig` available; flat-kwarg path still works via compat. Optional migration. |
| PR 7 (landed) | Typed `LTX2ContinuationState` available. ~50-line Dreamverse PR: replace `server/video_generation.py:89` import; move `apply_*`/`save_*`/`clear` off the state class onto `VideoGenerationWorker`; read knobs from typed state instead of env globals; swap `list[PIL.Image]` → `list[np.ndarray]`. |
| PR 7.5 (open) | Streaming server skeleton — Dreamverse's `runtime/factory.py` either keeps building `GPUPool` from `RealtimeRuntimeConfig` (current path), or migrates to `ServeConfig.streaming` shape and invokes `fastvideo serve --config realtime.yaml`. Dreamverse's `RuntimePool`/`RuntimeSlot` Protocol can stay in place. |
| PR 7.6 (branch ready) | GPU pool upstream — `local_runtime.py` import becomes a public import with same symbols (`RealtimeRuntimeConfig`, `GPUPool`, `get_available_gpus`). Per-GPU continuation state inside the worker becomes a `SessionStore` reference (Dreamverse doesn't see this). `request.state` / `result.state` round-trip starts working end-to-end on the local runtime. |
| PR 7.10 (planned) | `generate_async` is canonical. Dreamverse's per-segment `user_step` flow can migrate from sync `generate_video(..., **kwargs)` to consuming the typed event stream. Optional; sync wrapper stays. |

## Dynamo backend contract

**FastVideo does not host any Dynamo code.** The backend package
(`args.py`, `main.py`, `backend.py`, `register.py`, `health_check.py`,
adapter, Dockerfile) lives entirely in the Dynamo repo at
`components/src/dynamo/fastvideo/`, modeled on
`components/src/dynamo/sglang/`.

FastVideo's only obligation is to expose a stable, typed Python API
that Dynamo's backend package imports.

### Contract surface

| Surface | Exposed as |
|---|---|
| Construction | `VideoGenerator.from_pretrained(model_path, **typed_kwargs)` (typed_kwargs = a stable subset from `GeneratorConfig`; no flat LTX2 legacy) |
| Sync execution | `generator.generate_video(request: GenerationRequest) -> VideoResult` |
| Async execution | `generator.generate_async(request: GenerationRequest) -> AsyncGenerator[VideoEvent, None]` (PR 7.10) |
| Typed request | `fastvideo.api.GenerationRequest`, `SamplingConfig`, `InputConfig` |
| Typed result | `VideoResult` with `video_bytes` or tensor frames + optional `ContinuationState` |
| Continuation | `ContinuationState(kind, payload)` — schema-versioned payloads |
| Health-check input | `VideoGenerator.default_health_check_request() -> GenerationRequest` (256x256 / 8 frames / 1 step) |
| Config dump | `GeneratorConfig.to_dict()` / `ServeConfig.to_dict()` |

### Request/response mapping (Dynamo ↔ FastVideo)

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

All fields exist on FastVideo's typed schema after PR 6 expansion (typed
LTX2 kwargs) + PR 7.10 (`generate_async` + health check).

### Reference: PR ai-dynamo/dynamo#7544

Closed draft establishing the Dynamo backend shape. Two frictions
identified:

1. Flat legacy LTX2 kwargs — solved by PR 6.
2. Sync-only generation — solved by PR 7.10's `generate_async`.

Next iteration of this PR (or its successor) will reopen against PR 8's
docs reference and land cleanly.

## Open questions across surfaces

| # | Question | Source | Status |
|---|---|---|---|
| Q-1 | Multi-model GPU pool | dreamverse_review D-1 | Deferred (production single-model) |
| Q-2 | LTX-2 prompt orchestration promotion to public | dreamverse_review D-2 | Open; consumer-side until 2nd consumer |
| Q-3 | Race-based provider fallback | dreamverse_review D-3 | Open; sequential is current public |
| Q-4 | Router upstream skip on Dreamverse | dreamverse_review D-4 | Resolved (PR 7.9 lands publicly, Dreamverse doesn't consume) |
| Q-5 / D-5 | `generate_async` cutover (audio re-encode) | dreamverse_review | **Blocked on PR 7.10** |
| D-6 | Don't upstream `realtime/local_runtime.py` | dreamverse_review | Resolved (Dreamverse switches to `streaming.gpu_pool.SubprocessGpuPool`) |
| D-7 / Q-6 | FP4Config public colocation | dreamverse_review | **Resolved May 2** — public NVFP4 landed with lazy flashinfer |
| D-8 | `ltx2_image_crf` silently dropped | dreamverse_review | **Unverified post-`d80c2a8`** — see [open-threads.md](open-threads.md) |
| D-9 | `aarch64-conda-linux-gnu-cc` triton compile failure | dreamverse_review | Operational; `ENABLE_TORCH_COMPILE=0` workaround |
| D-10 | Warmup OOM on shared GPU | dreamverse_review | Operational; idle-GPU pre-warm probe |
| D-11 | ffmpeg `Broken pipe` on disconnect | dreamverse_review | Cosmetic logging cleanup |
| — | `video_position_offset_sec` semantics (persistent vs per-segment) | dreamverse_integration | **Open — needs decision before PR 7.6 emits state** |
| — | `SessionStore` / `BlobStore` lifecycle (TTL/eviction/blob-drop) | dreamverse_integration | Open — defer to PR 7.5 design pass |

See [decisions-log.md](decisions-log.md) for full rationale per
decision.

## Don't / Cautions

- **Don't pop the Dreamverse stash on this branch.** It's 3867 lines of
  orphan modular refactor with broken absolute imports.
- **Don't change `RealtimeRuntimeConfig` shape without coordinating
  with Dreamverse `runtime/factory.py`.**
- **Don't promise public compatibility for private Dreamverse-only
  field aliases.** Those belong in the private adapter layer per design
  spec.
