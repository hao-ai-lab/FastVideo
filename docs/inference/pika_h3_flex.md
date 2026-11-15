# Pika H3 Flex

Pika H3 Flex is a post-trained MiniMax-H3 model that keeps H3's packed
video/audio transformer, Qwen3-VL conditioner, and video/audio VAEs. Pika ships
two rank-128 PEFT adapters instead of a second base checkpoint:

| Adapter | Trunk | Tasks |
|---|---|---|
| `fl2va/adapter.safetensors` | `transformer` | T2VA, I2VA, L2VA, FL2VA |
| `ref2va/adapter.safetensors` | `transformer_ref` | Ref2VA |

FastVideo detects `Pika-Labs/Pika-H3-Flex`, downloads only the selected task's
adapter, and applies it as an ordinary LoRA. The released adapter is selected
automatically; pass `lora_path` only when overriding it with a compatible
Pika adapter.

## Sampling contract

Pika's sampler is fixed, not a tunable step count:

* 6 transformer forwards over clean-time rungs
  `1000, 750, 500, 250, 100, 40, 0`;
* video shift `12.0`, audio shift `3.0`;
* no CFG, `guidance_scale=1.0`, empty negative prompt;
* each step re-noises the predicted clean sample with fresh Gaussian noise from
  a generator seeded `seed + 1`.

`num_inference_steps` must therefore be `6`; other values are rejected. This is
not a PDD checkpoint and it has no trained VSA gates, so PDD fused blocks and
VSA sparsity are not used.

## Tasks

The commands below assume a FlashAttention-4 host; see
[Attention backend](#attention-backend) before running them elsewhere.

```bash
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1

# text to video + audio
python examples/inference/basic/basic_pika_h3_flex.py --task t2va \
    --prompt "a paper boat drifting down a rainy street"

# 480p portrait, ten seconds
python examples/inference/basic/basic_pika_h3_flex.py --task t2va \
    --resolution 480p --aspect 9:16 --duration 10 \
    --prompt "a lighthouse at dusk, waves breaking below"

# first frame, last frame, or both
python examples/inference/basic/basic_pika_h3_flex.py --task i2va \
    --image first.png --prompt "the camera slowly pushes in"
python examples/inference/basic/basic_pika_h3_flex.py --task fl2va \
    --image first.png --last-image last.png --prompt "the flower opens"

# ordered references, including reference video and audio
python examples/inference/basic/basic_pika_h3_flex.py --task ref2va \
    --reference image:person.png --reference video:dance.mp4 \
    --reference audio:voice.wav --prompt "the person dances to the music"
```

Ref2VA accepts up to 12 references: 9 images, 3 videos, and 3 audio. These
counts are enforced locally, as is the rule that audio-only requests are
invalid. A video reference needs at least 22 decoded frames at 24 fps. At least
one image or video is required, and the Ref2VA pipeline must be selected with
`PikaH3FlexRef2VAModularPipeline` (the example does this).

Pika's hosted API also documents a 2–15 s limit per video/audio reference and a
15 s total per modality. **The local path does not currently enforce those
duration caps**; it trims each reference to the requested output duration. Treat
the per-reference and per-modality limits as recommendations until the checks
are added.

## Canvas and frames

Pika uses an exact 32-pixel bucket lattice. A request that matches a bucket is
kept exactly; otherwise FastVideo picks the tier from the requested short edge
(`<= 480` → p480, else p768) and snaps to the nearest aspect bucket in that tier.

| Tier | Short edge | Max pixels | Reference-image short edge |
|---|---|---|---|
| p480 | 480 | 832×480 | 1024 |
| p768 | 768 | 1344×768 | 2048 |

Frame counts follow Pika's `17n + 5` grid at 24 fps (5–15 s, 120–362 aligned
frames). The default 16:9 p768 canvas is 1344×768 and 124 frames.

| Duration | 5 s | 10 s | 15 s |
|---|---|---|---|
| `num_frames` | 124 | 243 | 362 |

The example script picks the bucket with `--resolution {480p,768p}` and
`--aspect` (`21:9`, `16:9`, `3:2`, `4:3`, `5:4`, `1:1`, `4:5`, `3:4`, `2:3`,
`9:16`, `9:21`), and the frame count with `--duration` seconds. `--height`,
`--width`, and `--num-frames` override them.

Through the API, a request that omits `height` and `width` uses the 16:9
bucket of the configured tier:

```python
PipelineSelection(experimental={"pika_resolution_tier": "p480"})  # 832x480 when the size is omitted
```

An explicit `height`/`width` always wins and selects its own tier. "Omitted"
means absent from a request parsed from a mapping or config file, or from a
`generate_video(...)` call without `height`/`width`; a `GenerationRequest`
constructed in Python counts `SamplingConfig`'s 1280×720 default as set, which
snaps to the 768p 16:9 bucket. Pass the size explicitly in that case.

Pika center-crops both keyframes to the bucket. MiniMax-H3 stretches its first
frame instead; do not reuse that fit for Pika.

## Attention backend

Run Pika with FlashAttention-4 on Hopper or Blackwell:

```bash
FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1 \
    python examples/inference/basic/basic_pika_h3_flex.py --task t2va --prompt "..."
```

FA4 is opt-in. Installing `flash_attn.cute` is not enough:
`fastvideo/attention/utils/flash_attn_default.py` only selects it when
`FASTVIDEO_FA4=1`, and neither the example's defaults nor `--lossless-accel`
set that variable. With `FASTVIDEO_FA4=1` and an unusable FA4 install the run
fails at import instead of silently falling back.

Without FA4, the `FLASH_ATTN` backend uses FA3 or FA2 if one is installed. On a
host with neither (common on Blackwell), backend selection falls back to
`TORCH_SDPA`. SDPA at 768p is much slower and can run out of memory, so do not
use it for 768p benchmarks or production runs.

## Performance and memory

All measurements: one GB200, 768p 16:9 (1344×768), 5 s (124 frames), six
forwards, warm requests, FA4, every component resident (`--offload resident`).

| Configuration (T2VA) | Generation time |
|---|---|
| Official `pika-h3-pro` CLI | 59.7 s |
| FastVideo, BF16 | 26.2 s |
| FastVideo, NVFP4 `quality-36L` decoder | 18.0 s |
| FastVideo, NVFP4 `fast-8L` decoder | 17.4 s |

The 36L decoder measured 40.5 dB PSNR against the BF16 decoder on the same
latents. The 8L decoder was timed but not scored here.

Task coverage, seed 42, NVFP4 text encoder, BF16 decoder, also with
`FASTVIDEO_MINIMAX_H3_FUSIONS=all`:

| Task | Wall | Denoise | Decode | Peak memory | Check |
|---|---|---|---|---|---|
| T2VA | 25.49 s | 16.70 s | 8.04 s | 104.6 GiB | on prompt; 21.7 dB PSNR vs the BF16-encoder output |
| I2VA | 27.37 s | 18.02 s | 8.23 s | 104.9 GiB | first frame vs input 26.5 dB, SSIM 0.715 |
| FL2VA | 28.95 s | 19.49 s | 8.12 s | 105.2 GiB | first frame 39.6 dB / 0.971, last frame 29.1 dB / 0.819 |
| Ref2VA, two image references | 51.17 s | 38.42 s | 7.74 s | 108.5 GiB | 64,241 packed tokens; both references used |

The NVFP4 text encoder's first request in a fresh process paid a one-time
first-use cost of about 183 s; later requests did not.

Peak resident memory for the BF16 FastVideo T2VA run was about 129 GiB. The
official CLI's sampling peak was 71.7 GiB; the two figures measure different
things (process-resident memory versus the sampling phase).

The example's default `--offload lazy` loads components on demand and offloads
the encoders and VAE. It fits smaller devices but reloads the DiT on every
request: the same T2VA request took 48.06 s (28.31 s denoise) on GB200 in that
mode. Use `--offload resident` when the device holds every component.

## Training-free acceleration

The 6 forwards already use FastVideo's normal inference path. Available
lossless switches:

* `FASTVIDEO_MINIMAX_H3_EXACT_KERNELS=all` — exact fused H3 kernels;
* `FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=16` — cache reference conditioning
  (Ref2VA);
* `FASTVIDEO_H3_VAE_TILE_PARALLEL=1` — parallel VAE tiles;
* `FASTVIDEO_ULYSSES_A2A=auto` — Ulysses all-to-all for multi-GPU sequence
  parallelism;
* `OffloadConfig(lazy_module_load=True)` and `dit_layerwise=True` for smaller
  GPUs;
* `CompileConfig(enabled=True)` for `torch.compile`;
* FlashAttention-4 through `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
  FASTVIDEO_FA4=1` (see [Attention backend](#attention-backend)).

`--lossless-accel` in the example enables the first four before the generator
starts. It does not enable FA4; set the two attention variables at launch. Kernel work beyond the existing exact kernels is intentionally not part
of this integration.

## NVFP4 on Blackwell

Pika can reuse FastVideo's existing H3 NVFP4 VAE decoder and NVFP4 Qwen3-VL
text encoder. Both are weight formats, not new sampler paths; the six-forward
Pika schedule is unchanged.

```bash
# 36-block quality decoder: needs the full 36-block H3 VAE
python examples/inference/basic/basic_pika_h3_flex.py --task t2va \
    --prompt "a paper boat drifting down a rainy street" \
    --nvfp4-decoder ./FastH3-NVFP4-VAE-Decoder/quality-36L/decoder-nvfp4.safetensors

# 8-block fast decoder: depth-cuts any H3 VAE with at least 8 blocks
python examples/inference/basic/basic_pika_h3_flex.py --task ref2va \
    --reference image:person.png --reference video:dance.mp4 \
    --prompt "the person dances to the music" \
    --nvfp4-decoder ./FastH3-NVFP4-VAE-Decoder/fast-8L/decoder-nvfp4.safetensors \
    --nvfp4-text-encoder ./fasth3-nvfp4-encoder/text_encoder
```

* `quality-36L` keeps the full decoder depth and is the closest match to Pika's
  dense decoder. `fast-8L` is much smaller and faster but is a distilled
  student; expect a visible quality drop. Pika's six-step re-noise latents are
  not the student's original training distribution, so validate 8L on your
  prompts before using it for delivery.
* The NVFP4 decoder replaces only `vae.decoder`; keyframe and reference
  encoding still use the FP32 VAE encoder.
* The NVFP4 text encoder quantizes the Qwen3-VL language stack while keeping
  the vision tower in bf16, so Ref2VA reference images/videos still work.
* Do not select `taeh3` with an NVFP4 decoder; `h3-vae` is required.
* Both options require Blackwell FP4 hardware. The fused decoder kernels and
  CUDA graphs are already enabled by default; see
  `FASTVIDEO_H3_VAE_NVFP4_FUSED` and `FASTVIDEO_H3_VAE_NVFP4_CUDA_GRAPH`.
* There is no NVFP4 **VAE encoder** path. The H3 video VAE encoder is a causal
  3D CNN and stays FP32; quantizing it would need a separate convolution
  kernel/quantization effort.

If a process first loads the 8L decoder it depth-cuts the VAE to eight blocks;
loading 36L afterwards in the same process is not supported. Choose one decoder
tier per process.

## Hardware

The selected H3 trunk plus Qwen3-VL does not fit comfortably on a 24 GB card
without offload. A single 80 GB GPU can run with text-encoder offload; use
layerwise DiT offload or multiple GPUs for smaller devices. Ref2VA encodes
reference images at up to a 2048 short edge, so its conditioning memory is
higher than T2VA's.
