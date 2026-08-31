# Run H3 with a server and playground

Load FastH3 once, then iterate on prompts in a browser, with cURL, or from your
app. The server and clients can run on the same machine. Each job reuses the
loaded model until you stop the server.

The server supports the OpenAI-compatible video-job API. The Python and
JavaScript client examples use that interface, but requests go to FastVideo.
You do not need an OpenAI account or cloud key.

The [H3 recipe selector](minimax-h3.md) provides the same workflow with runtime
selection. MLX recipes use Python directly; they do not have a server adapter.

The Python SDK can also keep a model loaded. Create one `VideoGenerator` and
reuse it for multiple `generate()` calls in the same process. Re-running a
standalone script creates a new process and loads the model again. Use a server
when you want separate clients to share that process.

## Install and start the server

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
Do not copy Sora-specific durations or resolutions onto H3. The server config
uses 1344 × 768, 124 frames, 24 fps, and the five-point distilled sigma schedule
with four DiT forwards.

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
