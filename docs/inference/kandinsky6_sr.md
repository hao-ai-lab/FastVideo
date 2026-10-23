# Kandinsky 6 Video Super-Resolution

`Kandinsky6SRPipeline` upscales a low-resolution video by x2, x2.25 or x4 (video-to-video, no text prompt). The clip is
encoded once with the SR VAE (KVAE); its latent is cut into overlapping tiles, each tile is enlarged by the *latent
upscaler*, refined by a text-free SR DiT in a few denoising steps and decoded, and the tiles are blended back together.
The source audio is kept. The clip can come from anywhere, for example from the [Kandinsky 6 T2IVA](kandinsky6.md)
pipeline.

## Models

| Variant | Hub repo | Scheduler | Default steps per tile |
|---|---|---|---|
| Flow matching | `kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers` | `FlowMatchEulerDiscreteScheduler` (shift 5.0) | 4 |
| Distilled | `kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` | `PiflowScheduler` (shift 3.5, `n_grid` 10) | 2 |

Both repos have the same layout and share `vae/` and `latent_upscaler/`:

```text
model_index.json      _class_name = Kandinsky6SRPipeline
transformer/          Kandinsky6SRTransformer3DModel (sr_params: trained base resolution, RoPE scale, noise level)
vae/                  Kandinsky6SRVAE
latent_upscaler/      Kandinsky6SRLatentUpscalerBank (x2 and x4 models)
scheduler/            FlowMatchEulerDiscreteScheduler or PiflowScheduler
```

The scheduler component drives the denoising loop, and each repo resolves to its own preset (4 or 2 steps). The
distilled transformer's head holds `n_grid` predictions per latent channel, which `PiflowScheduler` integrates.

## Usage

```bash
python examples/inference/basic/basic_kandinsky6_sr.py --video-path input.mp4 --scale 2.25
python examples/inference/basic/basic_kandinsky6_sr.py --video-path input.mp4 \
    --model-path kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers
```

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained("kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", num_gpus=1)
result = generator.generate({
    "inputs": {"video_path": "input.mp4"},
    "output": {"output_path": "outputs_video/sr", "return_frames": False},
    "extensions": {"sr_resolution_scale": 2.25, "sr_target_resolution": "fullhd"},
})
```

The output geometry and frame rate follow the input clip; the request's `height` / `width` / `num_frames` are not used.
Clips are resampled to 24 fps by a fixed stride and only the first 121 frames (5 s, `1 + 8k`-aligned) are processed.
The source audio (mono, 44.1 kHz) is trimmed to the processed span and muxed into the output.

## Request parameters

The `sr_*` options belong only to Kandinsky6 SR. Pass them in `request.extensions` (as above), or under
`request.stage_overrides.sr`. The SR pipeline reads them from `ForwardBatch.extra`; they are not fields of the
shared `SamplingParam` or `ForwardBatch`. Other model families reject these options. Existing
`generate_video(..., sr_resolution_scale=...)` calls remain supported for Kandinsky6 SR; replace direct
`SamplingParam(sr_...=...)` construction with request extensions or these keyword arguments.
For the config-based CLI, use dotted overrides such as `--request.extensions.sr_resolution_scale 4`, rather than
shared `--sr-*` flags. The model-specific example above keeps its `--scale` and `--tiles-batch-size` flags.

| Field | Default | Meaning |
|---|---|---|
| `num_inference_steps` | 4 / 2 | Denoising steps (DiT calls) per tile. The upstream Diffusers pipeline counts grid points instead (its 5 is 4 steps here). The distilled model was trained for 2; other values run with a warning. |
| `sr_resolution_scale` | `2.25` | Total upscale: `2`, `4` or `2.25` (x1.125 pixel pre-upscale, then x2). |
| `sr_tiles_batch_size` | `1` | Tiles denoised per DiT call (raise it only if memory allows). |
| `sr_tile_min_overlap` | `0.20` | Minimum overlap between neighbouring tiles, as a fraction of the tile. |
| `sr_target_resolution` | `None` | Downscale the result to `hd`, `fullhd`, `2k` or `WxH`. |
| `sr_target_resize_mode` | `fit` | `fit` keeps the aspect ratio, `exact` uses the bucket dimensions. |
| `seed` | `42` | Tile group *k* (of `sr_tiles_batch_size` tiles) is seeded with `seed + index of its first tile`. |

Two Python-only inputs take raw tensors and are passed as keyword arguments of `generate_video()`:

- `sr_lr_latent`: an unscaled KVAE latent `[T, C, H, W]` of the source, instead of `video_path` (skips decoding and
  encoding the video; scale 2 or 4 only, since 2.25 needs the pixel pre-upscale).
- `sr_audio` / `sr_audio_sample_rate`: a mono waveform in `[-1, 1]` to mux instead of the source's own audio.

## Limitations

- NABLA block-sparse attention, which both repos request for 512-pixel tiles, is not wired; the DiT runs dense attention
  and logs a warning.
- No sequence or tensor parallelism: the DiT runs on one GPU. Tiles are processed one group after another.
- One clip per request.
