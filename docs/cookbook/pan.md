# PAN 2 recipes

<RecipeFamily family="pan2" />

Generate a video from text, or from a first frame, with `IFM/PAN2`. One pipeline
serves both modes: leave out the first frame for text-to-video and supply one
for image-to-video. The defaults are 704×1248, 313 frames (13 s at 24 fps),
50 steps, guidance 2.5 and PAN 2's default negative prompt. See [PAN 2 Text/Image to Video](../inference/pan2.md)
for the model, sampler and parallelism details.

`IFM/FastPAN2` is PAN 2's 8-call distilled student: the same pipeline and
inputs, 8 transformer calls without guidance or a negative prompt, and VSA2
sparse attention on B200, GB200 and B300 GPUs. See [FastPAN2](#fastpan2) below.

## Run on a GPU

Follow the [CUDA installation guide](../getting_started/installation/gpu.md), then run
from the repository root. Set `PAN2_MODEL_PATH` to run from a local copy of the
checkpoint. PAN 2 runs with its prompt and video guardrails by default, so install
them first (see [Guardrails](../inference/pan2.md#guardrails)):

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail
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

## FastPAN2

FastPAN2 needs a B200, GB200 or B300 GPU (sm_100a / sm_103a): its VSA2 sparse
attention kernels build from `fastvideo-kernel/vsa2_train` on first use, so run
from a source checkout. The example renders a 5 s text-to-video clip at
704×1248 on one GPU; set `IMAGE_PATH` in the script for image-to-video and
`FASTPAN2_NUM_GPUS=8` for sequence parallelism. Install the guardrails first, as
for PAN 2.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail
python examples/inference/basic/basic_fastpan2.py
```

The checkpoint fixes the sampler, so leave `num_inference_steps`,
`guidance_scale` and the negative prompt unset. Frames follow
`4 * ceil(seconds * 24 / 4) + 1` (121, 241 and 481 for 5, 10 and 20 s). On 8
B300 GPUs a 5, 10 and 20 s clip takes 2.2, 3.8 and 7.7 s, against 17.9, 41.2 and
113.0 s for PAN 2 with 50 guided steps; on one B300 it takes 9.4, 19.5 and 42.4 s.
See [FastPAN2](../inference/pan2.md#fastpan2) for how it samples.

`examples/serving/openai_fastpan2.yaml` serves it on the same `/v1/videos`
routes, with the alias `fastpan2` and a 5 s default clip:

```bash
fastvideo serve --config examples/serving/openai_fastpan2.yaml --server.host 127.0.0.1
curl --fail-with-body http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model": "fastpan2", "prompt": "A red fox trots through fresh snow in a pine forest at dawn."}'
```

A first frame in `image_reference` switches the request to image-to-video, as
for PAN 2.

## Evidence

The PAN 2 recipes are **Source-backed**: the model ID, example and serving YAML are
checked against each other, and the serving defaults are tested against the
registered `pan2_t2v` preset. No GPU generation, memory measurement or HTTP
client run is recorded, so hardware fit and throughput are unknown. Numerical
parity with the reference implementation is checked by the PAN 2 parity tests.

The FastPAN2 recipes are **Source-backed** as well. Their example and the
8-GPU sampler were run on B300 GPUs from the Hub checkpoint (the timings above,
and renders that match the reference implementation's); the HTTP serving path
has no recorded run.
