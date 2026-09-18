# Infinite Livestream

Infinite Livestream is a chat-driven FastH3 broadcast. Viewers type prompts into a web
page, the app rewrites them with an LLM, generates clips with FastVideo, and
plays them back as one continuous HLS stream on that same page. When nobody is
typing it feeds itself from a preset of idle prompts, so the channel never goes
dark.

It lives in this monorepo under `apps/infinite_livestream/`.

```
chat -> Director -> PromptUpsampler (OpenAI-compatible LLM)
           |
           v enqueue / move / pop
        Engine -> FastH3Backend -> FastVideo
           |  frames + audio
           v
        Pacer -> HlsSink -> the page's <video>
```

Everything runs in a single process, and the page, the playlist and the chat
endpoint are served from one HTTP origin, so publishing the stream means
pointing a tunnel or reverse proxy at one port.

## Requirements

- Linux with NVIDIA GPUs. The [default configuration](infinite_livestream/configs/infinite_livestream.yaml)
  targets four GB200 GPUs: Blackwell `sm_100a` sparse attention, a replicated
  transformer, and GPU-resident text encoder and VAEs. A different GPU setup
  needs corresponding changes to `runtime` and `inference`; the GPU count must
  divide the model's attention head count.
- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- A CUDA 13 toolkit with `nvcc` and a compatible C++ compiler for the kernel
  source build below.
- A complete FastH3 checkpoint; see [Download weights](#download-weights).
- An API key for prompt rewriting. The default configuration uses OpenAI;
  rewriting runs for idle filler too, so the stream needs the key even when
  nobody is chatting.
- FFmpeg with the `libx264` and `aac` encoders on `PATH`; see [FFmpeg](#ffmpeg).

## Install

Use a source checkout containing `apps/infinite_livestream/`. Run these commands
from the FastVideo repository root. The app is packaged in FastVideo's
`infinite-livestream` extra; `fasth3` adds its generator dependencies.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

git submodule update --init --recursive \
  fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

# Point this at your CUDA 13 toolkit.
export CUDA_HOME=/usr/local/cuda
export CUDACXX="$CUDA_HOME/bin/nvcc"
TORCH_CUDA_ARCH_LIST=10.0a UV_TORCH_BACKEND=cu130 \
  uv pip install -e ".[fasth3,infinite-livestream]"
```

This installs the pinned FA4 package and builds this checkout's
`fastvideo-kernel` with the Blackwell VSA extension. The CUDA and architecture
settings above match the default GB200 configuration. See the
[kernel build guide](../../fastvideo-kernel/README.md#installation) for build
prerequisites and troubleshooting.

## Download weights

Use [FastH3 v1 VSA-DataFree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree),
FastVideo's recommended four-step checkpoint. Its VSA-H3 attention settings
match this app's defaults. Download the whole snapshot, including the text
encoder and both VAEs, into a local directory accessible to the GPU host:

```bash
export LIVESTREAM_WEIGHTS_PATH=/absolute/path/to/FastH3-v1-VSA-DataFree
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --local-dir "$LIVESTREAM_WEIGHTS_PATH"
```

The app expects `modular_model_index.json` and the `transformer`, `text_encoder`,
`tokenizer`, `processor`, `vae`, `audio_vae`, `scheduler` and `audio_scheduler`
directories. Downloading only the transformer is insufficient; the text encoder
and VAEs need their weight files as well as their configs.

## FFmpeg

Install FFmpeg with `libx264` and `aac` encoding support. For example, on
Debian/Ubuntu:

```bash
sudo apt install ffmpeg
```

For the optimized native build, follow Dreamverse's
[FFmpeg instructions](../dreamverse/README.md#optional-building-ffmpeg-for-better-performance).
After running that installer, source its environment file in the shell that
will launch the livestream:

```bash
source apps/dreamverse/scripts/ffmpeg-env.sh
```

## Quick start

With the virtual environment active and `LIVESTREAM_WEIGHTS_PATH` set above:

```bash
export OPENAI_API_KEY=...
infinite-livestream-server
```

Open **<http://localhost:8081>** on the server, or use the server's hostname when
connecting remotely. The default bind address is `0.0.0.0`; `--port` overrides
the port.

The page and the HLS stream start while the model loads, so a viewer arriving
during startup sees the page and a black stream. Weight loading and compile
warm-up take several minutes. Check readiness with:

```bash
curl http://localhost:8081/healthz
```

`{"connected": true}` means model loading and warm-up have finished.

## Configuration

Copy the [default YAML](infinite_livestream/configs/infinite_livestream.yaml)
from the repository root, edit it, and pass the copy with `--config`:

```bash
cp apps/infinite_livestream/infinite_livestream/configs/infinite_livestream.yaml my-config.yaml
infinite-livestream-server --config my-config.yaml
```

| Block | Contents |
|---|---|
| `inference` | What the checkpoint is asked for: clip length, canvas, sparse-attention kernels, compile policy. |
| `runtime` | How it is hosted: GPU count, sharding, offload. |
| `upsampler` | Prompt rewriting: model, endpoint, how many clips one prompt may become. |
| `moderation` | Whether viewer prompts are checked, and against which endpoint. |
| `director` | Idle filler depth, per-viewer cooldown, chat command, filler directory. |
| `output` | Where the playlist is written (defaults under `$XDG_STATE_HOME`), and the video bitrate. |
| `web` | Bind address and port. |

For another OpenAI-compatible provider, set `upsampler.base_url` and
`upsampler.model`. Moderation is enabled by default and uses that endpoint too
unless `moderation.base_url` is set. If the provider does not offer `/moderations`,
configure a separate moderation endpoint and export its `MODERATION_API_KEY`.

API keys and the machine's weights path stay in the environment:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required. Prompt rewriting runs for the idle filler too, so the stream does not start without it. |
| `LIVESTREAM_WEIGHTS_PATH` | Required. FastH3 model directory. `--weights` overrides it. |
| `MODERATION_API_KEY` | Optional. Falls back to `OPENAI_API_KEY`. |

## Clip geometry

Clip geometry is fixed by the checkpoint: 24 fps, frame counts of the form
`17n + 5`, lengths between 5 and 15 seconds, and a 768 pixel short edge.
`inference.clip_seconds: 14.375` is the longest clip it can produce, at 345
frames.

Keeping one clip length means one compiled shape. Setting
`inference.warmup_lengths: all` warms every legal length instead, which makes
startup slower but avoids a one-off compile stall on a viewer's first
odd-length clip.

## API Endpoints

| Route | Description |
|---|---|
| `GET /` | The watch page. |
| `GET /assets/<file>` | Logo and favicon. |
| `GET /hls/<file>` | Playlist and segments, written by `infinite_livestream/sink.py`. |
| `GET /healthz` | `{"connected": bool}`, true once the model is loaded. |
| `WS /state` | One JSON snapshot on connect, then one per change. |
| `POST /chat` | `{"author": str, "text": str}`. Returns 429 with `retry_after` when that viewer is still on cooldown. |

The cooldown is answered by `POST /chat` rather than reported later, so the
sender's page can disable its send box and count down. The chat feed is shared
by every viewer, so refusals are kept out of it.

## Idle fillers

When nobody is typing, the stream keeps itself fed from a list of prompts.
`director.fillers` names the directory holding `fillers.json`, and defaults to
the one that ships in `infinite_livestream/presets/`.

```json
{
  "style": "the look and tone for idle filler clips",
  "idle_prompts": ["a lighthouse keeper teaching a seagull to play chess"]
}
```

`style` defines the house style for idle filler clips. Viewer prompts may use
their own style by default (`upsampler.viewer_free_style: true`). Set
`upsampler.viewer_free_style: false` to apply the house style to viewer prompts
as well.
`idle_prompts` feeds the filler; an empty list turns the filler off, as does
`director.idle_queue_target: 0`.

To change the stream's identity, copy the directory, edit `fillers.json` and
point `director.fillers` at it. The file is read once, at startup.

## Now-playing titles

Each viewer sits at a different point in the stream, so the server cannot say
what is on screen. It publishes which clip occupies which instant, and the page
locates itself against that.

Where a browser reports the date of the frame it is showing, the page uses it
directly. Most browsers report nothing, so the page falls back to the live edge
the server publishes minus how far behind its own buffer edge it is playing.
Append `?debug=1` to the page URL to see which source answered.

## Tests

```bash
pytest apps/infinite_livestream/infinite_livestream/tests -m "not gpu"
```

One test is marked `gpu`. It checks that `infinite_livestream/clip_plan.py`'s copy of
MiniMax-H3's packing constants still matches FastVideo's, and importing the
upstream module needs a live CUDA driver. Run it when the pinned FastVideo
version moves.

## Adding another model

`FastH3Backend.submit(frames, prompt, seed, height, width)` is the seam.
Everything above it, meaning the engine, director, queues, pacer, sink and web
app, is model-agnostic. Everything below it is MiniMax-H3 specific:
`clip_plan.py` is its geometry and `backend.py` selects its kernels.

A second checkpoint needs its own geometry module and its own backend behind
that seam. LTX-2, for example, packs `8n + 1` frames at different resolutions.

## Troubleshooting

**`ffmpeg not found on PATH`.** Follow [FFmpeg](#ffmpeg). If you used the native
build, source `apps/dreamverse/scripts/ffmpeg-env.sh` before starting the app.

**The weights are incomplete.** Startup lists the missing components before
any GPU work begins. The model directory needs `transformer`, `text_encoder`,
`tokenizer`, `processor`, `vae`, `audio_vae`, `scheduler`, `audio_scheduler`
and `modular_model_index.json`. See [Download weights](#download-weights) for
the complete checkpoint.

**`FastH3's sm100a route needs fastvideo-kernel built with the Blackwell VSA
extension`.** Startup checks for the fast sparse-attention kernel before
loading any weights. Follow [Install](#install) to build it, or set
`inference.vsa_kernel: triton` in your config to use the slower fallback.

**`FastH3's FA4 route needs the pinned flash-attn-4 package`.** Include the
`fasth3` extra as shown in [Install](#install), or set `inference.fa4: false` in
your config.

**A clip takes much longer than the others.** Each distinct clip length is a
separate compiled shape, and the first clip at a new length pays a one-off
compile cost. `inference.warmup_lengths: all` pays all of them at startup instead.
