# OpenAI-compatible HTTP contract

FastVideo exposes one model-agnostic REST engine for image and video models.
Launch it from a typed serve config:

```bash
fastvideo serve --config examples/serving/openai_fasth3.yaml
```

All generation routes share one serialized engine. FastVideo pipelines mutate
per-request sampling state, and some adapters merge weights at load time, so a
single loaded pipeline is never entered concurrently by image and video
requests. HTTP handling and job polling remain asynchronous.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/v1/models` | List the served model and optional startup adapter |
| `GET` | `/v1/models/{model}` | Retrieve one served model card |
| `POST` | `/v1/videos` | Submit an asynchronous video job |
| `POST` | `/v1/videos/sync` | Generate and return an MP4 response directly |
| `GET` | `/v1/videos` | List in-memory jobs with `after`, `limit`, and `order` |
| `GET` | `/v1/videos/{id}` | Retrieve job status and metadata |
| `GET` | `/v1/videos/{id}/content` | Download a completed MP4 |
| `DELETE` | `/v1/videos/{id}` | Delete a job and its completed artifact |
| `POST` | `/v1/images` | Generate an image |
| `POST` | `/v1/images/edits` | Generate an image from image references |
| `GET` | `/v1/images/{id}/content` | Download a generated image |
| `GET` | `/health` | Liveness probe |

`POST /v1/videos/generations` remains an alias for older FastVideo clients.

## Video requests

The canonical shape follows vLLM-Omni and accepts SGLang's common flat
extensions. Fields that FastVideo cannot represent for the loaded model fail
at admission with HTTP 400 instead of creating a job that later fails.

```json
{
  "model": "fasth3",
  "prompt": "A fox runs through fresh snow.",
  "seconds": "5",
  "size": "1344x768",
  "video_params": {
    "fps": 24,
    "num_frames": 124
  },
  "seed": 42,
  "num_inference_steps": 5,
  "guidance_scale": 1.0,
  "image_reference": [
    {"image_url": "https://example.com/first-frame.png"}
  ],
  "extra_params": {
    "vsa_mode": "exempt"
  }
}
```

Resolution precedence matches vLLM-Omni:

1. `size`
2. top-level `width` and `height`
3. `video_params.width` and `video_params.height`

Top-level `fps` and `num_frames` similarly take precedence over the nested
block. If `num_frames` is absent, `seconds * fps` is used. FastVideo also keeps
the legacy `input_reference`, `reference_url`, `video_path`, and `video_url`
spellings.

Reference objects support URL or local-path strings through `image_url`,
`video_url`, and `audio_url`. `file_id` references are schema-compatible but
return HTTP 400 because FastVideo does not provide an OpenAI Files store.
Multipart `input_reference` uploads are saved under the configured output
directory.

## Jobs and synchronous responses

An asynchronous submission returns a `video` object in `queued` state. Its
status advances through `in_progress` to `completed` or `failed`. Completed
jobs expose `file_name`, the FastVideo compatibility extension `file_path`,
timings, and peak-memory metadata when the pipeline reports them.

`POST /v1/videos/sync` returns `video/mp4` bytes. It includes
`X-Request-Id`, `X-Model`, `X-Inference-Time-S`, `X-Stage-Durations`, and
`X-Peak-Memory-MB` headers.

FastVideo's synchronous CUDA execution cannot be interrupted after launch.
Deleting an in-progress resource removes it from the API immediately; the
engine remains serialized until the call exits and then removes any artifact.

## Model and LoRA selection

`server.served_model_name` controls the public model id. If omitted, the
checkpoint path is used. Requests that name another model fail with HTTP 400.

LoRAs are configured under
`generator.pipeline.components.{lora_path,lora_nickname,lora_strength}`. The
startup adapter appears in `/v1/models`, and requests can use either its model
nickname or a vLLM-Omni selector:

```json
{
  "prompt": "A fox runs through fresh snow.",
  "model": "fasth3-dense-datafree",
  "lora": {
    "name": "fasth3-dense-datafree",
    "path": "/models/adapter_model.safetensors",
    "scale": 1.0
  }
}
```

The selector must match the adapter already loaded at startup. FastH3 adapter
files can contain dense replacement tensors and VSA gates in addition to
low-rank factors, so swapping them inside concurrent requests would corrupt
shared pipeline state. A mismatch is rejected with HTTP 400.

## MiniMax-H3 and FastH3

FastH3 uses the same general routes and adapter. `task` is accepted for
SGLang-compatible H3 clients:

- `t2va` uses text only.
- `fl2va` takes one or two image references.
- `ref2va` takes ordered image, video, and audio references and requires a
  server started with `MiniMaxH3Ref2VAModularPipeline`.

The released FastH3 pipeline generates one packed video/audio result per
request, uses 24 fps, requires guidance scale 1, and accepts frame counts on
its causal-VAE grid. The serving examples pin its five-point distilled sigma
schedule (four DiT forwards).

## Defaults and errors

Incoming explicit fields override operator-explicit `default_request` fields,
which override model preset defaults. Pydantic defaults do not masquerade as
client intent; the transport uses `model_fields_set`, while typed config parsing
tracks the exact paths written by the operator.

Errors use the OpenAI envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": null,
    "code": 400
  }
}
```

Parse, model-selection, startup-LoRA, and unsupported-parameter failures are
HTTP 400; missing resources are HTTP 404; generation failures are stored on
asynchronous jobs and returned as HTTP 500 when that job is retrieved.
