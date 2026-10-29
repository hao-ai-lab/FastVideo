# PAN 2 recipes

<RecipeFamily family="pan2" />

Generate a video from text, or from a first frame, with `IFM/PAN2`. One pipeline
serves both modes: leave out the first frame for text-to-video and supply one
for image-to-video. The defaults are 704×1248, 313 frames (13 s at 24 fps),
50 steps, guidance 2.5 and PAN 2's default negative prompt. See [PAN 2 Text/Image to Video](../inference/pan2.md)
for the model, sampler and parallelism details.

## Run on a GPU

Follow the [CUDA installation guide](../getting_started/installation/gpu.md), then run
from the repository root. The full `IFM/PAN2` checkpoint is not public yet; pass a
local directory or private repo you can access with `PAN2_MODEL_PATH`. PAN 2 runs
with its prompt and video guardrails by default, so install them first (see
[Guardrails](../inference/pan2.md#guardrails)):

```bash
pip install git+https://github.com/MBZUAI-IFM/PAN2-guardrail
python examples/inference/basic/basic_pan2.py
```

Set `IMAGE_PATH` in the script to a first frame for image-to-video. With more
GPUs, `PAN2_NUM_GPUS=4` shards the video tokens with sequence parallelism; the
count must divide PAN 2's 32 attention heads. Each rank still holds the full
29B DiT, so extra GPUs speed up generation but do not reduce per-device weight
memory. See [CPU offloading](../inference/offloading.md) for devices that cannot
hold the DiT.

## Serve with the OpenAI-compatible API

`examples/serving/openai_pan2.yaml` serves PAN 2 through the shared
`/v1/videos` routes. It mirrors the Python example: one GPU, the text encoder
offloaded between requests, and the pipeline's own sampling defaults.

```bash
fastvideo serve --config examples/serving/openai_pan2.yaml --server.host 127.0.0.1
```

Text-to-video uses the same request as any other served video model, with the
alias `pan2`:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model": "pan2", "prompt": "A red fox trots through fresh snow in a pine forest at dawn."}'
```

For image-to-video, add a first frame in `image_reference` (or the legacy
`input_reference`). The server then runs image-to-video; no second deployment or
workload is needed.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model": "pan2", "prompt": "The fox turns toward the camera.",
       "image_reference": {"image_url": "https://example.com/first-frame.png"}}'
```

Local image paths must sit under `server.media_roots`, and private or loopback
URLs need `server.allow_private_media_urls`. Poll `/v1/videos/{id}` and download
`/v1/videos/{id}/content` as in the [serving guide](openai-api.md). Generation
time for the default 13-second clip is not measured, so raise the client's
polling deadline for long jobs instead of treating a slow job as failed.

The cookbook's generated serving clients send a text prompt only. The image
request above is the supported way to supply a first frame over HTTP, and no
recorded generation backs it yet.

## Evidence

Both recipes are **Source-backed**: the model ID, example and serving YAML are
checked against each other, and the serving defaults are tested against the
registered `pan2_t2v` preset. No GPU generation, memory measurement or HTTP
client run is recorded, so hardware fit and throughput are unknown. Numerical
parity with the reference implementation is checked on CPU with a public
random-weight checkpoint, which verifies the port and produces noise rather than
useful video.
