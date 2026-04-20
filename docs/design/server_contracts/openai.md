# OpenAI-compatible HTTP Contract

The stateless FastVideo HTTP server lives at
[`fastvideo/entrypoints/openai/`](https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo/entrypoints/openai).
Launch: `fastvideo serve --config serve.yaml`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/videos/generations` | Synchronous video generation |
| `GET` | `/v1/videos` | List prior jobs held in the in-memory store |
| `GET` | `/v1/videos/{id}` | Job status / result |
| `GET` | `/v1/videos/{id}/content` | Download the MP4 once ready |
| `POST` | `/v1/images/generations` | Synchronous image generation |
| `GET` | `/v1/models` | Enumerate registered models |
| `GET` | `/health` | Liveness probe |

## `VideoGenerationsRequest` shape

Mirrors the OpenAI `POST /v1/videos/generations` shape:

```json
{
  "prompt": "a fox running through snow",
  "size": "1024x1536",
  "seconds": 5,
  "fps": 24,
  "num_frames": 121,
  "seed": 42,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "negative_prompt": "blurry, low quality",
  "input_reference": "/path/to/init.png"
}
```

SGLang-compatible extensions carried today:
`num_inference_steps`, `guidance_scale`, `guidance_scale_2`,
`true_cfg_scale`, `negative_prompt`, `enable_teacache`, `output_path`.

## Merge precedence

The server builds a `GenerationRequest` each call using three layers,
highest first:

1. **Request body (client-explicit)** — only fields carried in
   `request.model_fields_set` (Pydantic v2). Unset fields do not count,
   even if the Pydantic model has a schema default for them.
2. **`ServeConfig.default_request` (operator-explicit)** — projected via
   [`explicit_request_updates()`](../../../fastvideo/api/compat.py);
   only fields the operator actually wrote into the YAML count as
   defaults. Every other field inherits the schema default rather than
   being pinned.
3. **Hardcoded fallback** — e.g. `fps = 24`.

The gate matters: both surfaces carry schema defaults. Without
`model_fields_set` / explicit-path tracking, schema defaults would
masquerade as intent and silently shadow the other side.

See [`video_api.py::_build_generation_kwargs`](../../../fastvideo/entrypoints/openai/video_api.py)
for the canonical implementation; the per-request assembly lives there,
not in pipeline code.

## Continuation state

The stateless surface accepts an opaque `ContinuationState` round-trip.
Clients that want continuation pass the prior `state` blob back on the
next request, and receive a new one on the response when
`request.output.return_state = true`.

Shape:

```json
{
  "state": {
    "kind": "ltx2.v1",
    "payload": { "schema_version": 1, "segment_index": 3, ... }
  }
}
```

Payload is always JSON-serializable. Large tensors may live in an
opaque blob-store reference the client simply round-trips; see
[`LTX2ContinuationState`](../../../fastvideo/pipelines/basic/ltx2/continuation.py).

Continuation is not yet wired all the way through to
`generator.generate_video(...)` — PR 7.6 (GPU pool upstream) is the
pipeline-level consumer. PR 7 locked the envelope so this surface is
stable ahead of that plumbing.

## Error codes

| HTTP | Condition |
| --- | --- |
| `400 Bad Request` | Parse/validation failure (unknown field, type mismatch, incompatible preset/state) |
| `404 Not Found` | `GET /v1/videos/{id}` for an unknown job |
| `409 Conflict` | Job id already exists |
| `500 Internal Server Error` | Pipeline raised; body mirrors upstream OpenAI error envelope |
| `503 Service Unavailable` | No generator loaded, or shutdown in progress |

Errors include a JSON body with
`{"error": {"type": "...", "message": "..."}}` matching the OpenAI
Python SDK's expectation.

## What does not cross this boundary

* Flat legacy kwargs (`ltx2_refine_enabled`, `torch_compile_kwargs`,
  etc.) — these are init-time, configured via `ServeConfig.generator`,
  never per-request.
* Private Dreamverse-only fields — those live in a private adapter on
  the Dreamverse side; the public FastVideo surface never promises
  backward compatibility for them.
* Raw tensor payloads (`ltx2_audio_clean_latent` et al.) — these are
  derived by the pipeline from `ContinuationState`, never shipped as
  request fields.
