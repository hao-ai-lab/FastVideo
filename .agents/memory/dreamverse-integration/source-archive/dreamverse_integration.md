# Dreamverse ↔ FastVideo Integration

## Status

Working integration record. Captures how Dreamverse consumes the
FastVideo public API today, what's already shared, what's still ad
hoc, and what migrations land alongside each PR in the API refactor
sequence.

Pinned versions (last reconciled this session):

| Repo | Branch | Commit | Note |
|---|---|---|---|
| FastVideo (public) | `origin/main` | `70ee5d23` | PR 6 merged |
| FastVideo (public) | `will/api_7` | `3de5f833` | PR 7 in flight (typed continuation state) |
| FastVideo-internal | `will/rebase-nbv` | `1adc513e` | pre-PR-1 on the API refactor; has live realtime runtime |
| Dreamverse | `master` | `dc500330` | uses local + remote FastVideo runtimes via `server/runtime/` |

## Related Documents

- [PR plan.md](../../PR%20plan.md) — PR-by-PR sequence for the API refactor
- [apirefactor.md](../../apirefactor.md) — design spec
- [streaming-server-upstream-plan.md](streaming-server-upstream-plan.md) — upstream plan for `ui/ltx2-streaming/server/`
- `../../../Dreamverse/server/video_generation.py` — Dreamverse's worker + local `ContinuationState`
- `../../../Dreamverse/server/runtime/{factory,backend,gpu_pool,interfaces}.py` — runtime abstraction
- `../../../FastVideo-internal/fastvideo/entrypoints/realtime/{api_server,local_runtime}.py` — internal's realtime runtime (PR 7.5/7.6 upstream source)

## Surface Area

Dreamverse depends on FastVideo across three surfaces. Listed in order
of how stable each is.

### 1. Pipeline construction (stable)

`Dreamverse/server/video_generation.py:VideoGenerationWorker` calls
`VideoGenerator.from_pretrained(...)` with flat LTX-2 kwargs today.
After PR 6 the typed `GeneratorConfig` path exists; Dreamverse can
migrate at its own pace.

| Dreamverse usage | FastVideo public surface (post-PR 6) |
|---|---|
| `VideoGenerator.from_pretrained(model_path, ltx2_refine_enabled=…, …)` | `VideoGenerator.from_pretrained(config=GeneratorConfig(...))` |
| Flat `torch_compile_kwargs={…}` dict | `engine.compile.{backend,fullgraph,mode,dynamic,extras}` |
| `ltx2_vae_tiling=True` | `pipeline.vae_tiling=True` |
| `ltx2_refine_*` family | `pipeline.preset_overrides.refine.*` + `pipeline.components.upsampler_weights` |
| `enable_torch_compile_text_encoder` | `engine.compile.text_encoder_enabled` |

The legacy flat-kwarg path stays supported via `compat.py`; migration
is opt-in. PR 13's deprecation warnings are the eventual nudge.

### 2. Realtime runtime (in flight: PRs 7.5–7.6)

`Dreamverse/server/runtime/factory.py` selects a runtime backend at
process start:

```python
def create_runtime_pool() -> RuntimePool:
    if os.getenv("FASTVIDEO_REALTIME_BASE_URL"):
        return FastVideoRealtimePool(base_url=..., ws_url=..., default_model_id=...)
    return GPUPool(get_available_gpus())  # in-process, wraps fastvideo.entrypoints.realtime.local_runtime
```

Both backends speak the same `RuntimePool` / `RuntimeSlot` Protocol
(`server/runtime/interfaces.py`):

- `acquire(client_id, websocket=None) -> (gpu_id, RuntimeSlot)`
- `release(client_id)`
- `RuntimeSlot.{join_user, user_step, leave_user, register_stream_queue, …}`

Today both impls reach into FastVideo-internal's
`fastvideo.entrypoints.realtime.local_runtime` (which exposes
`RealtimeRuntimeConfig`, `GPUPool`, `GPUSlot`). The remote backend
talks HTTP+WS to a separately-deployed runtime of the same shape.

**Contract that PR 7.5/7.6 must preserve:**

- `RealtimeRuntimeConfig` accepts `model_registry`, `default_model_id`,
  `default_height/width/num_frames/fps/num_inference_steps/guidance_scale/seed/negative_prompt`,
  `default_ltx2_image_crf`, `startup_warmup_{enabled,prompt,timeout_seconds}`.
- `GPUPool(gpu_ids: list[int], config: RealtimeRuntimeConfig)` constructor.
- `pool.initialize() / shutdown() / acquire() / release() / get_status()`.
- HTTP endpoints on the remote variant: `GET /healthz`, `GET /readyz`,
  `GET /status`, `WS /ws`. (These already match what
  `Dreamverse/server/routes/health.py` consumes.)

When PR 7.6 lands the upstream of `fastvideo/entrypoints/realtime/`,
Dreamverse should not need any code change unless we rename the import
path. **Open: do we rename `realtime/` → `streaming/` to match the
public package introduced in PR 5.5?** A deprecation alias module
keeps both working during transition.

### 3. Continuation state (PR 7)

`Dreamverse/server/video_generation.py:89 ContinuationState` is
Dreamverse's hand-rolled per-session state holder. PR 7 introduces
the typed equivalent at `fastvideo/pipelines/basic/ltx2/continuation.py`.

#### Field mapping

| Dreamverse | PR 7 `LTX2ContinuationState` | Notes |
|---|---|---|
| `video_images: list[PIL.Image]` | `video_frames: list[np.ndarray]` (uint8 H×W×3) | numpy is leaner; Dreamverse already round-trips PIL→numpy→PIL just to add noise |
| `audio_latents: torch.Tensor` `[B, C, T, mel]` | `audio_latents: torch.Tensor` (safetensors-serialized; bf16-safe) | unchanged shape; safetensors preserves dtype incl. `bfloat16` |
| `LTX2_VIDEO_CONDITIONING_FRAME_IDX` (env) | `video_conditioning_frame_idx: int` | env constant → per-state field |
| `LTX2_VIDEO_CONDITIONING_STRENGTH` (env) | `video_conditioning_strength: float` | env constant → per-state field |
| `AUDIO_CONDITIONING_NUM_FRAMES` (env) | `audio_conditioning_num_frames: int` | env constant → per-state field |
| `AUDIO_CONDITIONING_STRENGTH` (env) | `audio_conditioning_strength: float` | env constant → per-state field |
| `audio_lps` (passed into `apply_audio`) | `audio_sample_rate: int \| None` | analogous; rename worth confirming with audio team |
| Computed `prefix_sec` per segment | `video_position_offset_sec: float` | **see open question below** |
| `segment_idx` (param to apply_*) | `segment_index: int` | per-state field |
| `VIDEO_CONTEXT_NOISE`, `AUDIO_CONTEXT_NOISE`, `ENABLE_AUDIO_COND` | not on state | runtime policy / regularization knobs, not portable session data |
| `apply_video / apply_audio / save_video / save_audio_latents / clear` | not on PR-7 state class | state is a pure data carrier; runtime owns lifecycle policy |

PR-7 is a strict superset of Dreamverse's data model **plus** lifts
several env globals into per-session typed fields.

#### Lifecycle mapping

| Dreamverse pattern | `SessionStore` API |
|---|---|
| `self.continuation = ContinuationState()` per session | `state = session_store.snapshot(sid) or LTX2ContinuationState()` |
| `apply_video(req_kwargs, segment_idx)` + `apply_audio(req_kwargs, segment_idx, audio_lps)` | `state = session_store.snapshot(sid)`; runtime builds request from `state.video_frames` / `state.audio_latents` etc. |
| `save_video(frames)` + `save_audio_latents(latents)` | runtime constructs new `LTX2ContinuationState`, then `session_store.store(sid, new_state.to_continuation_state())` |
| `clear()` at end of session | `session_store.drop(sid)` |

`SessionStore` and `BlobStore` ABCs ship with thread-safe in-memory
defaults (`InMemorySessionStore`, `InMemoryBlobStore`). Dreamverse can
adopt them as-is for the local runtime; remote runtimes can plug in
redis-backed implementations later.

#### Wire format (HTTP/WS round-trip)

Dreamverse's `FastVideoRealtimePool` already speaks the realtime
runtime's HTTP+WS protocol. When PR 7.5/7.6 land state emission on
the server side, the on-the-wire payload is the public envelope:

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

## Migration Plan

Per PR landed, Dreamverse adoption is opt-in.

### After PR 7 merges

Single-file change in Dreamverse, ~50-line PR:

1. Replace `server/video_generation.py:89 ContinuationState` import
   with `from fastvideo.pipelines.basic.ltx2.continuation import LTX2ContinuationState`.
2. Move `apply_video`, `apply_audio`, `save_video`, `save_audio_latents`,
   `clear` off the state class onto `VideoGenerationWorker` (these are
   runtime policy that uses the state, not part of the state itself).
3. Update `apply_audio` to read knobs from `state.audio_conditioning_num_frames`
   and `state.audio_conditioning_strength` instead of the env globals
   `AUDIO_CONDITIONING_NUM_FRAMES` / `AUDIO_CONDITIONING_STRENGTH`. The
   env globals can stay as defaults that populate the state when a new
   session starts.
4. Same treatment for video knobs: `state.video_conditioning_frame_idx`,
   `state.video_conditioning_strength`.
5. Frame storage swaps `list[PIL.Image]` for `list[np.ndarray]` —
   simpler `save_video` (no PIL conversion) and simpler `clear` (no
   `.close()` loop).

### After PR 7.5 lands streaming server skeleton

Dreamverse's runtime/factory.py either:

- Continues to construct `GPUPool` from `RealtimeRuntimeConfig` (the
  current path), now backed by the upstreamed `fastvideo/entrypoints/realtime/`.
- Or migrates to the upstream's `ServeConfig.streaming` shape and
  invokes `fastvideo serve --config realtime.yaml` as the launch path.

Either way, `Dreamverse/server/runtime/interfaces.py` `RuntimePool` /
`RuntimeSlot` Protocol can stay in place — it was modeled after the
realtime runtime's surface. No interface change needed.

### After PR 7.6 lands the GPU pool upstream

- The `local_runtime.py` import in
  `Dreamverse/server/runtime/gpu_pool.py:24` becomes a public import
  with the same symbols (`RealtimeRuntimeConfig`, `GPUPool`,
  `get_available_gpus`).
- Per-GPU continuation state inside the worker (`ltx2_continuation_images`,
  `ltx2_continuation_audio_latents`) gets replaced by a `SessionStore`
  reference. Dreamverse doesn't see this change — it's runtime-internal.
- `request.state` / `result.state` round-trip starts working end-to-end
  on the local runtime. Dreamverse's worker can begin reading
  `result.state` and feeding `request.state` between segments.

### After PR 7.10 lands the Dynamo backend contract

- `VideoGenerator.generate_async(...) -> AsyncGenerator[VideoEvent, None]`
  is the canonical API.
- Dreamverse's per-segment `user_step` flow can migrate from the legacy
  sync `generate_video(..., **kwargs)` path to consuming the typed
  event stream. Optional; the sync wrapper stays.

## Open Questions

### `video_position_offset_sec` semantics

Dreamverse computes `prefix_sec = float(audio_extra) / 24.0` per
segment in `apply_audio`. Not persisted on `ContinuationState`.

PR-7 has `video_position_offset_sec` as a **state field**. Two valid
interpretations:

(a) **Persistent across segments** — accumulating time offset for
long sessions; useful for time-coherent audio chaining.
(b) **Per-segment hint that rides on the carrier** — runtime
overwrites every time; field is harmless redundancy.

Field's docstring leans toward (b). Decide before PR 7.6 starts
emitting/consuming it. If we land on (a), document the accumulation
rule explicitly.

### `BlobStore` / `SessionStore` lifecycle ownership

PR 7's in-memory implementations have no eviction, no TTL, no
automatic blob cleanup on state replacement. Documented as a
per-deployment policy decision.

When PR 7.5/7.6 land the live consumer, who owns:

- bounded session capacity (LRU? TTL? hard max?)
- blob `drop()` chained when a state is replaced
- session expiry on websocket disconnect

Probably the streaming server's session manager, but worth stating
explicitly in PR 7.5's design.

### `realtime/` vs `streaming/` package naming

Currently:

- Public PR 5.5 introduced `fastvideo/entrypoints/streaming/` (skeleton + typed config).
- Internal has `fastvideo/entrypoints/realtime/` (live runtime).
- Dreamverse imports from `fastvideo.entrypoints.realtime` (per the internal name).

PR 7.5 either picks one or ships a deprecation alias module.
Recommendation in `streaming-server-upstream-plan.md`: keep
`streaming/` (it's the post-PR-5.5 public name), provide
`realtime/__init__.py` as a re-export with a `DeprecationWarning` for
one release cycle so internal/Dreamverse can land import updates.

## Test Coverage on the FastVideo Side

PR 7 ships:

- `fastvideo/tests/api/test_ltx2_continuation.py` — typed
  state round-trip (inline + blob), bf16 preservation, JSON
  serializability, kind/version validation, schema_version guard.
- `fastvideo/tests/entrypoints/streaming/test_session_store.py` —
  store/snapshot/hydrate/drop behavior on `InMemorySessionStore`;
  put/get/drop on `InMemoryBlobStore`; thread-safety of both.

PR 7.5+ should add a contract test that exercises the round-trip via
the same wire format Dreamverse's `FastVideoRealtimePool` consumes.

## Changelog

| Date | Change |
|------|--------|
| 2026-04-23 | Initial draft. Captures PR 6 / PR 7 mapping; open questions on `video_position_offset_sec`, lifecycle ownership, and `realtime/` vs `streaming/` naming. |
