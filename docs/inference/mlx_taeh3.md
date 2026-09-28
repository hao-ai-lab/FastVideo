# Decode H3 previews with TAEH3 on MLX

TAEH3 is an optional tiny video decoder for MiniMax H3. It replaces only
video reconstruction. The denoiser, sampler, resolution, frame count, and
audio decoder stay unchanged. The full H3 VAE remains the default.

TAEH3 produces a different reconstruction. Fine fur, fabric, vegetation,
and surface textures can look softer. Use it for previews or when you accept
that tradeoff. It is not a lossless acceleration of the full VAE.

## Generate a video

Use your existing MLX FastH3 environment and converted checkpoint:

```bash
python examples/inference/basic/mlx_fasth3.py \
  --model-root ~/models/FastH3-Preview-v0.2 \
  --mlx-checkpoint ~/models/FastH3-MLX/int6 \
  --prompt 'A red panda beside a mountain lake at sunrise.' \
  --height 480 --width 832 --num-frames 124 --steps 4 --seed 2027 \
  --video-decode-backend taeh3 --vae-dtype fp32 \
  --output-path video_samples/taeh3_preview.mp4
```

The first run downloads a 22.7 MB safetensors checkpoint from an immutable
upstream revision and verifies its SHA-256 digest. No remote Python code runs.
The cache is `~/.cache/fastvideo/taehv/taeh3.safetensors`.
Use `--taeh3-checkpoint /path/to/taeh3.safetensors` for offline use or a custom
trained checkpoint. Custom files must match the decoder architecture; they
are not required to match the upstream digest.

| Option | Default | Behavior |
| --- | --- | --- |
| `--video-decode-backend` | `h3-vae` | Select `taeh3` for approximate decoding. |
| `--taeh3-checkpoint` | Unset | Use the pinned upstream checkpoint from cache. |
| `--taeh3-chunk-size` | `5` | Latent frames per execution chunk. Smaller chunks reduce feature memory. |
| `--vae-dtype` | `fp32` | Decoder computation dtype. FP16 and BF16 are separate numerical tradeoffs. |

`--tiled-video-decode` controls the full VAE only. TAEH3 uses the whole spatial
canvas and bounded temporal chunks. Its memory blocks carry state across
chunks. The pipeline reports `video_decode_backend`, decode timing, and MLX
peak memory alongside the existing generation metrics.

The mode composes with `--fast-spatial` and temporal `--fast`. Those options
change the denoising workload and have additional quality costs. A TAEH3-only
measurement does not establish the quality or speed of a combined mode.

## Latent contract

The native decoder reads normalized diffusion latents in NCTHW layout through
`decode_latents_taeh3_mlx`. Do not apply the full VAE's mean, standard deviation,
or pixel denormalization. Its 24 latent channels reconstruct RGB at 16 times
the latent spatial dimensions.

H3 uses latent lengths `5*k-3`, such as 2, 7, and 37. TAEH3 removes three raw
frames from each group of 20 decoder outputs. Thus 37 latent frames produce
124 RGB frames. The port validates that contract before returning output.

## Provenance and validation

Architecture and weights come from Ollin Boer Bohan's MIT-licensed
[TAEHV H3 implementation](https://github.com/madebyollin/taehv/commit/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c).
The [Aryan fork](https://github.com/aryan5v/taehv/tree/aryan/first-class-taeh3)
adds an explicit `TAEH3` API and checkpoint-loading tests. The fork is not an
official MiniMax release and does not contain newly trained H3 weights.

Run the numerical tests against a local TAEHV checkout containing the released
weights:

```bash
TAEH3_REFERENCE_DIR=/path/to/taehv \
  python -m pytest fastvideo/tests/mlx/test_mlx_taeh3.py -q
```

Tests compare MLX FP32 with upstream sequential FP32 and parallel FP64 at
`atol=1e-5, rtol=1e-5`. The initial parallel CPU FP32 comparison failed that
strict gate, reaching about `4e-5` maximum error on a 37-latent small fixture.
CPU convolution rounding changes with its batch size. The FP64 reference
and sequential FP32 checks distinguish this from a temporal chunking error.
The original failed comparison is not reported as a pass.

Passing these tests means the MLX port agrees with the tiny decoder within
the specified tolerance. It does not mean TAEH3 matches the full H3 VAE.

## Compare decoders without another denoising run

Save the normalized packed `video_rows` returned by `pipeline.denoise` as
`np.savez("latents.npz", video=video_rows)`. Then run:

```bash
python examples/inference/basic/mlx_h3_decode_benchmark.py \
  --latents latents.npz \
  --model-root ~/models/FastH3-Preview-v0.2 \
  --mlx-checkpoint ~/models/FastH3-MLX/int6 \
  --height 480 --width 832 --num-frames 124 \
  --output-dir outputs/taeh3_comparison
```

Use a fresh output directory. The benchmark writes the first decoded frame
arrays and a JSON report with the input digest, MLX version, device, per-trial
latency, MLX peak active memory, lifetime process peak RSS, and swap snapshots.
Decoder loading is included; first-time checkpoint downloading is excluded.
Only run one MLX workload at a time. `--repeats` reverses the decoder order on
alternate trials. Do not treat two memory counters as additive or infer zero
page-outs from unchanged swap snapshots.

## Measured decoder results

On an Apple M4 Max with 36 GB unified memory, MLX 0.32.2, FP32 decoding,
and five-latent execution chunks:

| Workload | Full H3 VAE | TAEH3 |
| --- | --- | --- |
| Saved 37-frame latents to 124 RGB frames, 832x480 | 107.90 s | 1.44 s |
| MLX peak active memory for that decode | 11.03 GiB | 3.62 GiB |

These are one matched pair using the same seed-2027 production latent file,
including decoder loading. Both swap snapshots stayed unchanged. The decoded
images differ: PSNR against the full VAE was 29.86 dB, and inspected frames
showed softer fine detail. The approximately 75x ratio applies only to this
decoder comparison, not the entire generation pipeline.

Eight additional TAEH3-only decodes measured 0.96 s on first use and a 0.98 s median across seven warm trials, ranging from 0.96 to 0.99 s. Decoder construction and weight loading were included in each trial. The full VAE was not repeated eight times.

### Native resolution with temporal fast

A separate uncached run with `--fast --video-decode-backend taeh3`, without
spatial fast, completed in **205.47 s wall time**. It kept the native 832x480
canvas, denoised 73 source frames, and used RIFE to produce 124 output frames
with full-duration audio. Seed 2027, four steps, dense attention, INT6.

Prompt encoding took 16.66 s, denoising 181.14 s, TAEH3 decoding 0.64 s, RIFE
5.23 s, audio decoding 0.73 s, and muxing 0.40 s. Peak denoise MLX allocation
was 17.87 GiB. System swap use rose from 1231.19 to 2753.75 MiB.
The output has 124 H.264 frames at 832x480 and stereo AAC at 32 kHz.

This is one combined-mode measurement. It preserves spatial resolution but
still combines frame interpolation with approximate decoding. Frame samples
retain more fine detail than the spatial fast experiment below. Motion and
speech need human review; do not infer native dense generation parity.

### Spatial fast experiment, not the preferred quality path

A separate uncached generation with `--fast-spatial --video-decode-backend
taeh3` produced a 124-frame, 832x480 MP4 with stereo audio in **95.75 s wall
time**. Seed 2028, four denoising calls, INT6 weights, dense attention, and no
temporal fast mode were used. The internal canvas was 416x256, then cropped
and upscaled to the requested output size.

That run spent 16.66 s encoding the prompt, 77.05 s denoising, 0.34 s decoding
video, 0.30 s upscaling, 0.73 s decoding audio, and 0.40 s muxing. Peak denoise
MLX allocation was 16.85 GiB. System swap use rose from 1041.94 to 1247.19 MiB;
this measurement does not attribute that increase to a particular process.
It is one end-to-end result, not a repeated benchmark or native-resolution
quality comparison. The reduced canvas visibly loses detail, especially in
the opening frames. Speech intelligibility and motion quality need human
review before treating this combination as a final-output preset.
