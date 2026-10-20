# OpenAI-compatible serving examples

The REST serving engine is model-agnostic. Any model supported by
`VideoGenerator` can use the same `/v1/models`, `/v1/videos`, and `/v1/images`
surface. The two CUDA FastH3 configs here are source-backed examples, not
recorded serving benchmarks. Both configure four CUDA GPUs without a GPU model
or VRAM claim. The Wan configs are copied from the working generate config or
example for each checkpoint.

Use the [H3 server cookbook](https://haoailab.com/FastVideo/cookbook/openai-api/)
for the guided install, server, and client workflow.

Launch the full FastH3 checkpoint:

```bash
fastvideo serve --config examples/serving/openai_fasth3.yaml --server.host 127.0.0.1
```

Launch the dense FastH3 LoRA on the base MiniMax-H3 checkpoint:

```bash
adapter_path="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors)"

fastvideo serve --config examples/serving/openai_fasth3_lora.yaml \
  --server.host 127.0.0.1 \
  --generator.pipeline.components.lora_path "$adapter_path"
```

FastH3 adapters are hybrid startup patches: alongside low-rank factors they
may contain dense deltas and a VSA compression-gate replacement. They must be
selected when the server starts. A request may carry the vLLM-Omni `lora`
selector, but its name, path, and scale must match that startup adapter. A VSA
adapter also needs `attention_backend: VIDEO_SPARSE_ATTN_H3`, `VSA_sparsity`,
and `VSA_tile_size` like the full-checkpoint config.

After startup, open `http://127.0.0.1:8000/playground/` to generate and review
H3 videos in your browser. CUDA prompts reuse the loaded model. The playground
and API clients share the same server; both can run locally.

For native Apple Silicon MLX, use
`python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3.yaml`
(Preview) or `examples/serving/mlx_fasth3_8step.yaml` (8-Step V2) after preparing
the weights and editing their paths in that configuration. The same playground
and OpenAI Python client work with MLX. Its pipeline still loads and releases
components between phases to limit unified-memory use. See the cookbook for
setup and the supported text-to-video/audio request fields.

CUDA FastH3 8-Step V2 serving uses `examples/serving/openai_fasth3_8step.yaml`.

## Wan configs

Four Wan CUDA configs mirror the working generate config or example for each
checkpoint:

| Config | Model | GPUs | Served alias |
| --- | --- | --- | --- |
| `openai_fastwan21_1_3b.yaml` | FastWan2.1 T2V 1.3B (DMD, VSA) | 1 | `fastwan21-1.3b` |
| `openai_wan21_i2v_14b.yaml` | Wan2.1 I2V 14B 480P | 2 | `wan21-i2v-14b` |
| `openai_wan22_t2v_a14b.yaml` | Wan2.2 T2V A14B | 2 | `wan22-t2v-a14b` |
| `openai_wan22_ti2v_5b.yaml` | Wan2.2 TI2V 5B | 1 | `wan22-ti2v-5b` |

The 1.3B config needs the VSA attention backend:

```bash
FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN \
  fastvideo serve --config examples/serving/openai_fastwan21_1_3b.yaml
```

The I2V 14B and TI2V 5B servers accept image-conditioned requests. Supply the
source image through the OpenAI-compatible reference fields, for example
`{"input_reference": "/path/to/first-frame.png"}` or
`{"image_reference": [{"image_url": "https://example.com/first-frame.png"}]}`.
The I2V 14B server requires one on every request; the TI2V 5B server treats
its absence as text-to-video.

## Other families

These configs are source-backed declarations; none is a hardware-fit claim.
`docs/cookbook/serving-coverage.md` records the full baseline list and the
uncovered registry families.

| Config | Model | Task | GPUs |
| --- | --- | --- | --- |
| `openai_wan22_i2v_a14b.yaml` | Wan2.2 I2V A14B | image-to-video | 1 |
| `openai_hunyuan15_t2v_480p.yaml` | HunyuanVideo 1.5 T2V 480p | text-to-video | 1 |
| `openai_hunyuan15_i2v_480p.yaml` | HunyuanVideo 1.5 I2V 480p | image-to-video | 1 |
| `openai_kandinsky6_ti2va.yaml` | Kandinsky 6 Pro TI2VA | text/image-to-video with audio | 1 |
| `openai_kandinsky6_ti2va_distilled.yaml` | Kandinsky 6 Pro-distill TI2VA | text/image-to-video with audio | 1 |
| `openai_ltx2_distilled.yaml` | LTX2 Distilled | text-to-video with audio | 4 |
| `openai_flux1_dev.yaml` | FLUX.1-dev | text-to-image | 1 |
| `openai_flux2_dev.yaml` | FLUX.2-dev | text-to-image | 1 |
| `openai_flux2_klein.yaml` | FLUX.2-klein-4B | text-to-image | 1 |
| `openai_sd35.yaml` | SD3.5 Medium | text-to-image | 1 |
| `openai_stable_audio.yaml` | Stable Audio Open 1.0 | text-to-audio | 1 |
| `openai_mmaudio.yaml` | MMAudio large-44k-v2 | video-to-audio | 1 |

Image servers use `/v1/images/generations` and `/v1/images/edits`. Send the
model's advertised alias as `model` and set `FASTVIDEO_MODEL` for the sample
clients. SD3.5 Medium is gated; accept its license and authenticate before
starting the server.

## Audio jobs

Audio uses the same asynchronous job routes as video so clients keep one
polling and deletion contract. The job's `media_type` is `audio/wav`, and
`/v1/videos/{id}/content` returns WAV bytes. Set `seconds` or
`audio_start_in_s`/`audio_end_in_s`; do not send `size`, `width`, `height`, or
`video_params`.

MMAudio is video-to-audio and requires `video_reference` (a local path or
URL). URLs may point at a local or cluster media service. The server downloads
at most 2 GiB per reference by default and removes that copy when the job
finishes; set `FASTVIDEO_MEDIA_REFERENCE_MAX_BYTES` for longer clips. Stable
Audio is text-to-audio. Stable Audio needs the optional inference dependencies
listed in `examples/inference/basic/basic_stable_audio.py`.

```bash
python -m pip install httpx
FASTVIDEO_MODEL=stable-audio FASTVIDEO_SECONDS=6 \
  python examples/serving/clients/audio.py
FASTVIDEO_MODEL=mmaudio-large-44k-v2 \
FASTVIDEO_VIDEO_REFERENCE=/path/to/clip.mp4 \
  python examples/serving/clients/audio.py
```

Node and cURL clients are in `examples/serving/clients/audio.mjs` and
`audio.sh`.

Or submit, poll, and download video with the OpenAI Python client:

```bash
python -m pip install openai==3.6.0
python examples/serving/clients/video.py
```

The `clients/` directory also contains cURL and OpenAI JavaScript examples.
They default to model `fasth3`; set `FASTVIDEO_MODEL` to the advertised alias
when using a LoRA or another family config. Set `FASTVIDEO_BASE_URL` for another endpoint.
These clients call your FastVideo server, not OpenAI's cloud. The SDK key
`local` is a placeholder, not authentication. Keep the server on loopback or
use an authenticated proxy for remote access.

For a blocking call, `POST /v1/videos/sync` returns the generated media body
directly. Check its `Content-Type` before saving the response.
