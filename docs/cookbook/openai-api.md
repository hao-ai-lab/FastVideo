# Run H3 with a server and playground

Start FastH3 on CUDA, one DGX Spark, or Apple Silicon MLX, then iterate on prompts in a browser,
with cURL, or from your app. The server and clients can run on the same machine.

The server supports the OpenAI-compatible video-job API. The Python and
JavaScript client examples use that interface, but requests go to FastVideo.
You do not need an OpenAI account or cloud key.

The [H3 recipe selector](minimax-h3.md) provides the same workflow with runtime
selection. This guide covers FastH3 Preview text-to-video/audio. Other H3
recipes keep their direct Python commands.

CUDA requests reuse one loaded `VideoGenerator`. The Python SDK can do the same
when you reuse the generator across `generate()` calls. MLX keeps one
`MiniMaxH3MLXPipeline` and a prompt-embedding cache, but still loads and releases
components between phases to fit unified memory. MLX serving does not keep all
weights resident or remove phase-loading time. Use a server when you want
separate clients to share the process without restarting scripts.

## Install and start the server

### CUDA

Use a FastVideo clone and an activated Python environment. Complete the
[CUDA installation requirements](../getting_started/installation/gpu.md)
before running these commands:

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
fastvideo serve --config examples/serving/openai_fasth3.yaml --server.host 127.0.0.1
```

The configuration loads `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` and
advertises it as `fasth3`. It configures four CUDA GPUs but does not record
a GPU model or VRAM requirement. This is a source-backed server profile, not
the measured GB200 Python performance profile. Compilation is disabled.

Keep the server running. In another terminal, check readiness:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
```

After model loading completes, the response is `{"status":"ok"}`.

### NVIDIA DGX Spark

Complete the [DGX Spark installation](../getting_started/installation/spark.md)
before running these commands. GB10 has no FA4 / sm_100a VSA kernel:

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e .
FASTVIDEO_VSA_SM100A=0 FASTVIDEO_FA4=0 FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3 \
  fastvideo serve --config examples/serving/openai_fasth3_spark.yaml --server.host 127.0.0.1
```

The configuration loads `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`
on one GB10 and advertises it as `fasth3`. Lazy module load still reloads
Qwen3-VL and the DiT between phases of each request. Legal `num_frames` values
are `17n+5`, capped at 345 (15 s); a 345-frame request on one Spark can OOM.
There is no cookbook server for two Sparks; use the generate YAML after
[pairing two Sparks](../getting_started/installation/spark_pair.md).

Keep the server running. In another terminal, check readiness:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
```

After model loading completes, the response is `{"status":"ok"}`.

### Apple Silicon MLX

Complete the [Apple Silicon installation](../getting_started/installation/mps.md#run-fasth3-preview),
including `ffmpeg`. From your FastVideo clone, install the MLX extra:

```bash
uv pip install -e ".[mlx]"
```

Download and convert the weights once. Skip this step if you already have the
snapshot and converted DiT:

```bash
hf download FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2 --local-dir ./FastH3-Preview-v0.2
python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py --model-root ./FastH3-Preview-v0.2/transformer --out ./FastH3-MLX --formats "int6"
```

Edit `generator.model_root` and `generator.mlx_checkpoint` in
`examples/serving/mlx_fasth3.yaml` if your weights are elsewhere. Paths are
relative to the directory where you launch the server. Start it with:

```bash
python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3.yaml
```

This uses the native MLX pipeline, not PyTorch MPS. The default output is
832 × 480, 124 frames at 24 fps, with four DiT forwards and the full H3 VAE.
The HTTP field `num_inference_steps=5` describes five sigma points; the adapter
passes `num_steps=4` to MLX. Temporal/spatial fast modes, VSA, reference inputs,
LoRA selection, and alternate decoders are not exposed by this server adapter.
Unsupported request options return HTTP 400 before a job starts.

The server binds to `127.0.0.1:8000` and advertises `fasth3`, so the playground
and clients below work unchanged. Do not run CUDA, Spark, and MLX servers on the
same port. MLX readiness means the pipeline is initialized; components load during
generation. The first request can take longer than a repeated cached prompt.

The MLX server has no recorded device or unified-memory requirement. The
direct Python recipe's M4 Max measurements are not a server benchmark or a
minimum-memory claim.

## Open the playground

Open [the local H3 playground](http://127.0.0.1:8000/playground/) after startup.
Write a prompt, optionally set a seed, then select **Generate video**. The page
checks the job status and shows the completed video with a download link. Edit
the prompt and generate again without restarting the server.

The playground sends requests to `/v1/videos` on the same server that serves
the page. **Use this prompt with cURL** shows the equivalent submission. Recent
jobs include requests from other clients, so a script and the playground can
use the same model process. Opening the page or copying a command does not
start generation.

The page URL includes the active job ID. Reloading that URL resumes status
checks without resubmitting the prompt. A failed connection or a 30-minute
polling timeout does not cancel execution. Select **Check status** to reconnect.
If submission itself is interrupted, check Recent jobs before submitting again.

This first playground supports H3 text-to-video/audio. Reference-media inputs
remain available through the API, not through the playground. It does not start
or manage a GPU server for you.

## Generate with cURL or an SDK

These examples use the server's resolution, frame count, and sampling defaults.
Do not copy Sora-specific durations or resolutions onto H3. Both server configs
use 124 frames, 24 fps, and the five-point distilled sigma schedule with four
DiT forwards. CUDA and one Spark use 1344 × 768; MLX uses 832 × 480.

Each client submits a job, checks for completion or failure, and downloads an
MP4 named after the job ID. Polling stops after 30 minutes; a timeout does not
cancel GPU execution. Keep the printed job ID to retrieve its status later.
Transport retries are disabled to avoid accidental duplicate submissions.

### OpenAI Python

Install the tested client, then run the checked-in example:

```bash
python -m pip install openai==3.6.0
python examples/serving/clients/video.py
```

```python
--8<-- "examples/serving/clients/video.py"
```

### OpenAI JavaScript

Use Node.js 22 or later on your computer or in your webapp's backend. Do not put a
private server key in browser code.

```bash
npm ci --prefix examples/serving/clients
node examples/serving/clients/video.mjs
```

```javascript
--8<-- "examples/serving/clients/video.mjs"
```

### cURL

Install `curl` and `jq`, then run:

```bash
bash examples/serving/clients/video.sh
```

```bash
--8<-- "examples/serving/clients/video.sh"
```

## Connect your app

Set `FASTVIDEO_BASE_URL` to the FastVideo endpoint, including `/v1`, and
`FASTVIDEO_MODEL` to its advertised model alias. The examples default to
`http://127.0.0.1:8000/v1` and `fasth3`.

For a remote GPU machine, keep the server bound to loopback and forward the
port. Replace `user@gpu-host` with your SSH destination:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-host
```

Then open `http://127.0.0.1:8000/playground/` on your computer. The same forwarded
address works for the cURL and SDK clients. If local port 8000 is occupied, use
`-L 8001:127.0.0.1:8000`, open port 8001, and set `FASTVIDEO_BASE_URL` to
`http://127.0.0.1:8001/v1` for the example clients.

Your webapp backend can submit a job and return its ID to the browser. Poll
from the backend, then proxy the completed download or store it in your own
artifact store. Do not hold a browser request open for the entire generation.

The client key `local` is a placeholder required by the SDK, not server
authentication. FastVideo's HTTP server has no built-in API-key check. Before
public deployment, put it behind an authenticated TLS proxy with restricted
origins, request limits, and access controls. If your proxy uses bearer tokens,
set `FASTVIDEO_API_KEY` on your backend. Never reuse an OpenAI cloud key here.

## Compatibility and limits

The client examples target the OpenAI video-job API, not Chat Completions.
The compatibility tests cover real HTTP requests with the pinned SDKs and a
fake generator. They establish client and protocol behavior, not GPU generation
quality, latency, or memory use. The server configuration still needs a recorded
H3 hardware run before it can be marked Verified in the cookbook.

| Operation | Endpoint |
| --- | --- |
| List served models | `GET /v1/models` |
| Create a video job | `POST /v1/videos` |
| Retrieve status, including a failed job | `GET /v1/videos/{id}` |
| List jobs | `GET /v1/videos` |
| Download the MP4 | `GET /v1/videos/{id}/content` |
| Delete a job and its artifact | `DELETE /v1/videos/{id}` |

Failed jobs return HTTP 200 with `status: "failed"` and an `error` object so
SDK polling can stop. Invalid requests return HTTP 400; missing jobs return
HTTP 404. The download endpoint supports only the `video` variant, not
thumbnails or spritesheets. Remix, extensions, characters, and an OpenAI Files
store are not implemented. `/v1/videos/sync` is a FastVideo extension and is
not used by these client examples.

OpenAI marks its hosted Sora API as deprecated. FastVideo runs its own models;
the [OpenAI video API reference](https://developers.openai.com/api/reference/python/resources/videos/methods/create)
describes the client interface, not FastVideo model availability. The examples
pin tested SDK versions because future SDKs may change or remove video helpers.

The server serializes generation through one loaded pipeline. Job metadata is
in memory and is lost on restart. Async artifacts remain on disk until deleted
through the API; establish a retention policy for production use. Deleting an
in-progress job does not interrupt an already running CUDA call.

For model-specific request fields and reference media, see the
[HTTP contract](../design/server_contracts/openai.md).
