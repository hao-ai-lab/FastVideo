# Dynamo Native Backend Integration

FastVideo exposes a stable Python API that the
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) project consumes
as a pure-Python import, same tier as `vllm`, `sglang`, `trtllm`.

**FastVideo hosts no Dynamo code.** The backend subpackage
(`components/src/dynamo/fastvideo/`) lives in the Dynamo repo. This doc
is the reference integrators copy when standing up that package — it
mirrors the structure used by `dynamo/components/src/dynamo/sglang/`
and is known to satisfy the (closed) draft
[ai-dynamo/dynamo#7544](https://github.com/ai-dynamo/dynamo/pull/7544)
pattern.

## What FastVideo provides

The public surface Dynamo imports is intentionally small:

```python
from fastvideo import VideoGenerator
from fastvideo.api import (
    ContinuationState,
    GenerationRequest,
    InputConfig,
    OutputConfig,
    SamplingConfig,
    # Post-PR 7.10:
    VideoEvent, VideoProgressEvent, VideoPartialEvent, VideoFinalEvent,
    VideoResult,
)
```

| Surface | Availability | Notes |
| --- | --- | --- |
| `VideoGenerator.from_pretrained(model_path, **typed_kwargs)` | Today | `typed_kwargs` is a stable subset from `GeneratorConfig` — no flat legacy LTX-2 kwargs (guaranteed after PR 6) |
| `VideoGenerator.generate(request: GenerationRequest) -> GenerationResult` | Today | Aggregated; Dynamo wraps in `asyncio.to_thread` under `asyncio.Lock` |
| `VideoGenerator.generate_async(request) -> AsyncGenerator[VideoEvent, None]` | **PR 7.10** | Canonical execution substrate; sync wrapper reroutes through this |
| `VideoGenerator.default_health_check_request() -> GenerationRequest` | **PR 7.10** | 256x256 / 8 frames / 1 step; lets Dynamo build its health payload without knowing any FastVideo internals |
| `fastvideo.api.GenerationRequest` / `SamplingConfig` / `InputConfig` | Today | Stable public dataclasses |
| `fastvideo.api.ContinuationState` | Today (PR 7) | JSON-safe envelope; kind-versioned payloads |
| `fastvideo.api.VideoResult` | Today | `frames`, `video_path`, `state`, `metadata` |
| `config_to_dict(cfg)` | Today | Used by Dynamo's `dump_config(path, config)` |

## Backend package layout

Modeled on `components/src/dynamo/sglang/`:

```
components/src/dynamo/fastvideo/
├── __init__.py
├── __main__.py               # Entry: python -m dynamo.fastvideo
├── main.py                   # worker() dispatch — mirrors sglang/main.py
├── args.py                   # FastVideoArgGroup — CLI → GeneratorConfig
├── backend_args.py           # Dynamo runtime flags (namespace, fs_url, ...)
├── init_video_generation.py  # init_video_generation(runtime, config)
├── register.py               # register_video_generation_model() for Dynamo
├── backend.py                # VideoGenerationWorkerHandler
├── health_check.py           # FastVideoHealthCheckPayload
├── protocol.py               # NvCreateVideoRequest ↔ GenerationRequest adapter
├── request_handlers/
│   └── video_generation/
│       └── video_generation_handler.py  # async generate(req, ctx)
├── README.md
└── CLAUDE.md                 # per-backend guidance
```

None of these files live in FastVideo.

## Request/response mapping

Dynamo's `NvCreateVideoRequest` / `VideoNvExt` / `NvVideosResponse` map
one-to-one onto FastVideo's typed schema:

```
NvCreateVideoRequest          ->  fastvideo.api.GenerationRequest
  prompt                      ->    request.prompt
  size="WxH"                  ->    request.sampling.width, height
  seconds                     ->    seconds * nvext.fps -> request.sampling.num_frames
  input_reference             ->    request.inputs.image_path / video_path
  nvext.fps                   ->    request.sampling.fps
  nvext.num_frames            ->    request.sampling.num_frames  (overrides seconds*fps)
  nvext.num_inference_steps   ->    request.sampling.num_inference_steps
  nvext.guidance_scale        ->    request.sampling.guidance_scale
  nvext.seed                  ->    request.sampling.seed
  nvext.negative_prompt       ->    request.sampling.negative_prompt
  nvext.continuation_state    ->    request.state  (opaque ContinuationState)
  response_format             ->    (handled by adapter at output)

VideoFinalEvent               ->  NvVideosResponse
  video_bytes                 ->    data[0].b64_json   (if response_format=b64_json)
  uploaded URL                ->    data[0].url        (if response_format=url)
  metadata.inference_time_s   ->    inference_time_s
  continuation_state          ->    nvext.continuation_state  (reserved for disagg)
```

## Example: aggregated handler (sync wrap)

Satisfies the PR #7544 shape; works today against
`VideoGenerator.generate`, upgrades cleanly to `generate_async` after
PR 7.10.

```python
# components/src/dynamo/fastvideo/request_handlers/video_generation/
# video_generation_handler.py
from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, AsyncGenerator

from fastvideo import VideoGenerator
from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig


class VideoGenerationWorkerHandler:
    def __init__(self, generator: VideoGenerator, config, fs=None):
        self.generator = generator
        self.config = config
        self.fs = fs
        self._lock = asyncio.Lock()  # aggregated = one-in-flight

    async def generate(
        self,
        request: dict[str, Any],
        context,
    ) -> AsyncGenerator[dict[str, Any], None]:
        req = _to_fastvideo_request(request)
        t0 = time.perf_counter()
        async with self._lock:
            result = await asyncio.to_thread(self.generator.generate, req)
        elapsed = time.perf_counter() - t0

        video_bytes = _materialize(result, self.fs, request.get("response_format"))
        yield {
            "data": [video_bytes],
            "inference_time_s": elapsed,
            "model": request.get("model"),
        }


def _to_fastvideo_request(request: dict[str, Any]) -> GenerationRequest:
    nvext = request.get("nvext") or {}
    fps = nvext.get("fps", 24)
    num_frames = nvext.get("num_frames") or (request.get("seconds") or 4) * fps
    width, height = _parse_size(request.get("size"))

    return GenerationRequest(
        prompt=request["prompt"],
        negative_prompt=nvext.get("negative_prompt"),
        inputs=InputConfig(
            image_path=request.get("input_reference"),
        ),
        sampling=SamplingConfig(
            width=width, height=height,
            num_frames=num_frames, fps=fps,
            num_inference_steps=nvext.get("num_inference_steps", 50),
            guidance_scale=nvext.get("guidance_scale", 1.0),
            seed=nvext.get("seed", 1024),
        ),
        output=OutputConfig(save_video=False, return_frames=False),
        state=nvext.get("continuation_state"),  # public ContinuationState
    )
```

`_parse_size` and `_materialize` are small adapter helpers owned by the
Dynamo backend package; they never appear in FastVideo.

## Example: streaming handler (post-PR 7.10)

```python
async def generate(self, request, context):
    req = _to_fastvideo_request(request)
    async for event in self.generator.generate_async(req):
        if event.__class__.__name__ == "VideoProgressEvent":
            yield {"status": "generating", "progress": event.step / event.total_steps}
        elif event.__class__.__name__ == "VideoFinalEvent":
            yield {
                "data": [{"b64_json": base64.b64encode(event.video_bytes).decode()}],
                "inference_time_s": event.metadata.get("inference_time_s"),
                "nvext": {"continuation_state": _serialize_state(event.continuation_state)},
            }
```

Aggregated and streaming differ only in which events the handler
forwards; both share one `generate_async` substrate.

## Example: health check

```python
# components/src/dynamo/fastvideo/health_check.py
from dynamo.health_check import HealthCheckPayload
from fastvideo import VideoGenerator


class FastVideoHealthCheckPayload(HealthCheckPayload):
    def __init__(self, generator: VideoGenerator) -> None:
        # Post-PR 7.10: generator.default_health_check_request() returns a
        # typed GenerationRequest; dump it into the same dict shape that
        # FastVideo's adapter accepts.
        req = generator.default_health_check_request()
        self.default_payload = {
            "prompt": req.prompt or "test",
            "size": f"{req.sampling.width}x{req.sampling.height}",
            "response_format": "b64_json",
            "nvext": {
                "fps": req.sampling.fps,
                "num_frames": req.sampling.num_frames,
                "num_inference_steps": req.sampling.num_inference_steps,
                "guidance_scale": req.sampling.guidance_scale,
            },
        }
        super().__init__()
```

Fallback (pre-PR 7.10) — hardcoded 256×256 / 8 frames / 1 step, matching
[`VideoGenerationHealthCheckPayload`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/sglang/health_check.py#L198-L226).

## Init function sketch

```python
# components/src/dynamo/fastvideo/init_video_generation.py
async def init_video_generation(runtime, config, shutdown_endpoints):
    from fastvideo import VideoGenerator
    from fastvideo.api import config_to_dict

    server_args, dynamo_args = config.server_args, config.dynamo_args
    generator = VideoGenerator.from_pretrained(**config.fastvideo_kwargs())

    dump_config(dynamo_args.dump_config_to, config)

    endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )
    shutdown_endpoints[:] = [endpoint]

    handler = VideoGenerationWorkerHandler(
        generator, config, fs=get_fs(dynamo_args.media_output_fs_url)
    )
    payload = FastVideoHealthCheckPayload(generator).to_dict()

    await asyncio.gather(
        endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            health_check_payload=payload,
        ),
        register_video_generation_model(
            generator, endpoint, server_args,
        ),
    )
```

## Args adapter

`FastVideoArgGroup` (Dynamo-side) converts CLI flags into a typed
`GeneratorConfig` — **never** into legacy flat kwargs. Because PR 6
added typed homes for every kwarg the internal `gpu_pool.py` used,
this adapter can build the config purely from the public typed schema:

```python
def build_generator_config(args) -> "GeneratorConfig":
    from fastvideo.api import (
        CompileConfig, ComponentConfig, EngineConfig, GeneratorConfig,
        OffloadConfig, ParallelismConfig, PipelineSelection,
    )
    return GeneratorConfig(
        model_path=args.model_path,
        engine=EngineConfig(
            num_gpus=args.num_gpus,
            parallelism=ParallelismConfig(tp_size=args.tp_size, sp_size=args.sp_size),
            offload=OffloadConfig(dit=args.dit_offload, text_encoder=args.te_offload),
            compile=CompileConfig(enabled=args.compile, mode=args.compile_mode),
        ),
        pipeline=PipelineSelection(
            workload_type=args.workload or "t2v",
            preset=args.preset,  # e.g. "ltx2_two_stage"
            components=ComponentConfig(
                upsampler_weights=args.refine_upsampler,
                lora_path=args.refine_lora,
            ),
        ),
    )
```

## Registration

Dynamo's Rust side skips HuggingFace `config.json` downloads for
`ModelType::Videos`, same fast path used by image diffusion. The
Python-side registration:

```python
# components/src/dynamo/fastvideo/register.py
from dynamo.llm import ModelDeploymentCard, ModelType, register_model


async def register_video_generation_model(generator, endpoint, server_args):
    mdc = ModelDeploymentCard.with_name_only(server_args.model_name or server_args.model_path)
    await register_model(endpoint, mdc, ModelType.Videos, readiness_gate=asyncio.Event())
```

## Contract guarantees

These guardrails let the Dynamo backend be written once and not
re-chase FastVideo drift:

1. `GenerationRequest` field paths are stable across PR 6 onward. Any
   breaking rename triggers a major bump and appears in
   [`inference_schema_parity_inventory.yaml`](../inference_schema_parity_inventory.yaml).
2. `ContinuationState.payload` is JSON-serializable or references
   opaque blob ids. Dynamo can round-trip it through RPC without
   special-casing torch tensors.
3. `VideoGenerator.from_pretrained` accepts a typed `GeneratorConfig`;
   legacy flat kwargs are compatibility-only and deprecate in PR 13.
4. `generate_async` (PR 7.10+) emits events in order
   `Progress* → Partial* → Final`; the final event always has exactly
   one occurrence per request.
5. `default_health_check_request()` (PR 7.10+) returns a request that
   passes `parse_config` and produces a non-zero-latency but bounded
   workload (256×256 / 8 frames / 1 step).

FastVideo's contract tests (`fastvideo/tests/contract/`) assert these
with mocked Dynamo-style handlers that import only the public surface.
If a change to FastVideo breaks the adapter pattern, those tests fail
at FastVideo's CI — before the Dynamo-side integration even knows.

## What the Dynamo adapter MUST NOT import

* Anything under `fastvideo.pipelines.*` directly (pipelines are
  internal; presets identify them by name on
  `PipelineSelection.preset`).
* `fastvideo.fastvideo_args.FastVideoArgs` (legacy compat type).
* `fastvideo.api.compat.*` private helpers
  (`_validate_continuation_state` etc.) — the public boundary is
  `VideoGenerator` + `fastvideo.api`.
* Any flat legacy LTX-2 kwarg (`ltx2_refine_upsampler_path`,
  `torch_compile_kwargs`, etc.) — all have typed homes in
  `GeneratorConfig`.

## Future: disaggregated prefill/decode

PR 7's continuation state was designed to survive RPC transport, so a
future Dynamo split where prefill yields state and decode hydrates it
is expressible without changing the contract. The streaming server
(PR 7.6) already uses the `SessionStore` pattern; Dynamo's disagg
could wire a distributed `SessionStore` backend by the same interface.

## See also

* [OpenAI HTTP contract](openai.md)
* [Streaming WebSocket protocol](streaming.md)
* Draft PR reference: [ai-dynamo/dynamo#7544](https://github.com/ai-dynamo/dynamo/pull/7544)
* Dynamo SGLang backend (template this doc is modeled on):
  [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/sglang)
