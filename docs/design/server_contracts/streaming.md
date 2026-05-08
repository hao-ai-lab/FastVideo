# Streaming WebSocket Server Contract

The streaming server (`fastvideo/entrypoints/streaming/server.py`) speaks
a JSON-over-WebSocket protocol with binary fMP4 chunks for media. This
document is the authoritative spec for the message catalogue and the
session state machine. Any change to either must update this document
in the same PR that touches `protocol.py` or `session.py`.

## Endpoint

| Path | Protocol | Purpose |
|---|---|---|
| `WS /v1/stream` | WebSocket (JSON + binary) | Per-session realtime streaming |
| `GET /health` | HTTP | Liveness probe (`status`, `stream_mode`, active `sessions`) |

The server is launched by `fastvideo serve --config <serve.yaml>` when
the config carries a `streaming:` block. Without that block the same CLI
launches the OpenAI stateless HTTP server instead.

## Connection lifecycle

Every WebSocket connection holds exactly one `Session`. Sessions move
through the states in `SessionState` (`fastvideo/entrypoints/streaming/session.py`).

```
                    ┌──────────────┐
                    │ INITIALIZING │  ← WebSocket accepted, before init frame
                    └──────┬───────┘
                           │ session_init_v2 received
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
         QUEUED      GPU_BINDING       REJECTED
            │              │              ↑
            │ slot ready   │              │ max-sessions hit
            ▼              ▼              │ or invalid init
                       ┌────────┐         │
                       │ ACTIVE │ ────────┘
                       └────┬───┘
              segment loop  │
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
            COMPLETE      ERROR      TIMEOUT
        (clean leave)  (any failure)  (idle / segment_cap reached)
```

Terminal states (`COMPLETE`, `ERROR`, `TIMEOUT`, `REJECTED`) are sinks —
no transitions out. The transition matrix is enforced in
`session.py::_VALID_TRANSITIONS`; bad transitions raise.

`SessionManager` enforces the per-process budgets pulled from
`StreamingConfig`:

- `session_timeout_seconds` — idle reaper drops sessions that haven't
  advanced; non-terminal sessions transition to `TIMEOUT`.
- `generation_segment_cap` — a session that hits the cap transitions to
  `COMPLETE` after the last segment ships.

## Message catalogue

Every JSON frame carries `{"type": <str>, ...}`. Pydantic models in
`protocol.py` are the source of truth; this table is the human-readable
view.

### Client → server

| `type` | Required fields | Purpose |
|---|---|---|
| `session_init_v2` | — | Opening frame. Carries preset, curated prompts, optional initial image, feature toggles, optional `continuation_state` to resume from a snapshot. |
| `segment_prompt_source` | `prompt` | Request the next segment using the supplied prompt; optional sampling overrides (`seed`, `num_inference_steps`, `guidance_scale`, `negative_prompt`). |
| `seed_prompts_updated` | `seed_prompts` | Replace the session's seed-prompt list; takes effect on the next segment. |
| `enhancement_updated` | `enabled` | Toggle prompt enhancement for subsequent segments. |
| `auto_extension_updated` | `enabled` | Toggle automatic per-segment prompt extension. |
| `loop_generation_updated` | `enabled` | Toggle loop-generation mode. |
| `generation_paused_updated` | `paused` | Pause/resume segment generation; queued requests defer. |
| `snapshot_state` | — | Request the current `ContinuationState` for export; server replies with `continuation_state_snapshot`. |

The opening frame must be `session_init_v2`. Any other first frame is
rejected with an `error` (code `invalid_message`) and the WebSocket is
closed.

### Server → client

| `type` | Carries | When emitted |
|---|---|---|
| `queue_status` | `position`, `queue_depth` | After `session_init_v2` accepted, before GPU binding. |
| `gpu_assigned` | GPU id, model id | Once a generator slot is bound. |
| `ltx2_stream_start` | session-level metadata | Once the session enters `ACTIVE`. |
| `ltx2_segment_start` | `segment_idx`, `prompt`, prompt source | When a `segment_prompt_source` request begins generation. |
| `step_complete` | `segment_idx`, denoise timings | After the segment's denoising loop finishes (before media emission). |
| `media_init` | `segment_idx`, mime, stream id | First frame of fMP4 output for the segment. |
| binary frame | fMP4 fragment bytes | Subsequent media chunks; the protocol enforces that `media_init` precedes any binary frames. |
| `media_segment_complete` | `segment_idx`, chunk count, byte count | Last media chunk for the segment. |
| `ltx2_segment_complete` | `segment_idx`, segment summary | Segment fully shipped; ready for the next `segment_prompt_source`. |
| `ltx2_stream_complete` | session summary | Session reached `generation_segment_cap` or client requested clean shutdown. |
| `session_timeout` | reason | Session hit `session_timeout_seconds`; immediately followed by close. |
| `continuation_state_snapshot` | `kind`, `payload` | Reply to `snapshot_state`. The payload is the same shape produced by `LTX2ContinuationState.to_continuation_state(...)`. |
| `error` | `code`, `message` | Any validation/runtime error. Non-fatal errors keep the connection open; fatal errors precede a `close`. |

## Continuation state

The session optionally accepts a `continuation_state` dict inside the
opening `session_init_v2` frame. When present, the server hydrates it
into a `ContinuationState(kind, payload)` envelope and feeds it as the
`request.state` on the first segment's `GenerationRequest` — letting a
client resume after a disconnect, migrate sessions across processes,
or replay a prior session.

After every segment, if the runtime returns a fresh state, the server
persists it to the `SessionStore` so a `snapshot_state` request can
export it. The store and serialization contracts live with the model
family (e.g. `fastvideo/pipelines/basic/ltx2/continuation.py` for LTX-2).

## Example flow

```
client                                                    server
──────                                                    ──────
WS /v1/stream  ─────── connect ─────────────────────────►
               ◄────── (accept)

{"type": "session_init_v2",
 "preset": "ltx2_two_stage",
 "curated_prompts": ["a fox in snow", "the fox jumps"],
 "initial_image": {...},
 "stream_mode": "av_fmp4"} ─────────────────────────────►

                                                          (validate, queue, bind)
               ◄──── {"type": "queue_status",
                       "position": 0, "queue_depth": 0}
               ◄──── {"type": "gpu_assigned",
                       "gpu_id": 0, "model_id": "..."}
               ◄──── {"type": "ltx2_stream_start", ...}

{"type": "segment_prompt_source",
 "prompt": "a fox in snow",
 "source": "curated"} ───────────────────────────────────►
                                                          (run pipeline)
               ◄──── {"type": "ltx2_segment_start",
                       "segment_idx": 1, ...}
               ◄──── {"type": "step_complete",
                       "segment_idx": 1, "timings": {...}}
               ◄──── {"type": "media_init",
                       "segment_idx": 1,
                       "mime": "video/mp4", ...}
               ◄──── <binary fMP4 init segment>
               ◄──── <binary fMP4 fragment>
               ◄──── <binary fMP4 fragment>
               ◄──── {"type": "media_segment_complete",
                       "segment_idx": 1, "chunks": 12}
               ◄──── {"type": "ltx2_segment_complete",
                       "segment_idx": 1, ...}

{"type": "segment_prompt_source",
 "prompt": "the fox jumps"} ─────────────────────────────►
                                                          (segment 2 …)

{"type": "snapshot_state"} ──────────────────────────────►
               ◄──── {"type": "continuation_state_snapshot",
                       "kind": "ltx2.v1",
                       "payload": {"schema_version": 1, ...}}

(close) ──────────────────────────────────────────────────►
                                                          (session → COMPLETE)
```

## Backward / forward compatibility

- Adding a new client message: append a Pydantic model to `protocol.py`
  with a unique `type`; add the discriminator entry to `ClientMessage`;
  add a row to the table above. Old clients that don't send the new
  message remain compatible.
- Adding a new server message: emit only when a new feature flag is
  enabled (or always emit, since clients ignore unknown types).
- Changing an existing message: bump the `type` (e.g. `session_init_v2`
  → `session_init_v3`) and accept both for one release cycle. Never
  silently change field semantics under the same `type`.
