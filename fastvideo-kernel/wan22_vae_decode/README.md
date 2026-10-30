# Wan2.2 VAE decoder (sm_100a, sm_103a)

The decoder of the Wan2.2 VAE (the VAE of Wan2.2-TI2V-5B: 48 latent channels, 16 x 16 spatial and 4x temporal
compression) at 704 x 1248, on one, two or four B200 / GB200 GPUs of a node. It decodes latents [1, 48, T, 44, 78]
into the video [1, 3, 4T - 3, 704, 1248] for any T >= 1, as diffusers' `AutoencoderKLWan.decode(z).sample` does;
`reference.py` defines the result exactly, a plain-PyTorch decoder bit-exact with diffusers in fp32 and in bf16.

## Use

```python
import sys
sys.path.insert(0, "fastvideo-kernel/wan22_vae_decode")
from wan22_vae_decode import WanVAEDecoder

decoder = WanVAEDecoder(vae.state_dict(), device="cuda")   # diffusers' AutoencoderKLWan, or its state dict
video = decoder.decode(latents)   # [1, 48, T, 44, 78], as the VAE's decode takes them -> [1, 3, 4T-3, 704, 1248] bf16
```

- **Over several GPUs**, every rank builds a decoder with the same process group (`WanVAEDecoder(sd, device, pg)`,
  default: torch.distributed's default group) and calls `decode` with the full latents; rank 0 gets the video and the
  other ranks `None`. On one GPU without torch.distributed, the decoder initializes a single-rank NCCL group.
- The first call builds the extension from `solution/` (about a minute) into `$WAN22_VAE_DECODE_BUILD_DIR` (default
  `~/.cache/fastvideo/wan22_vae_decode`). Each process builds in a directory of its own, so concurrent processes do
  not collide, and puts the finished build in the `cache/` beside it, from which later processes copy it.
- `WanVAEDecoder(..., frames_per_pass=4)` sets the latent frames each GPU carries through one pass (see below).
- `WanVAEDecoder` prepares the weights once (layout, phase and Winograd transforms of the weights). The raw entry
  points are `prepare(weights, rank, world, pg)`, `run(z, weights, rank, world, pg, out)` (4 frames per pass) and
  `run_with(..., out, frames_per_pass)` in `solution/binding.cpp`, destination-passing, on the current stream;
  `weights` are the 110 tensors of `weight_names()` in bf16.

## Performance and memory

At PAN2's default 704 x 1248, 24 fps, on GB200 GPUs, against PAN2's decode today (diffusers' `AutoencoderKLWan` in bf16
under autocast, one GPU, one latent frame at a time). Memory is the peak allocated and, after it, the peak reserved by
PyTorch's caching allocator (what nvidia-smi shows), per GPU; the default is 4 latent frames per GPU per pass:

| Decoder | 8 s (193 frames) | 12 s (289 frames) | 20 s (481 frames) |
|---|---|---|---|
| Release, 1 GPU | 5.88 s, 12.9 / 17.1 GiB | 8.84 s, 13.9 / 19.4 GiB | 14.76 s, 15.8 / 91.5 GiB |
| **This decoder, 1 GPU** | **1.17 s** (x5.0), 20.7 / 36.3 GiB | **1.74 s** (x5.1), 21.2 / 36.8 GiB | **2.89 s** (x5.1), 22.2 / 37.8 GiB |
| **This decoder, 2 GPUs** | **619 ms**, 18.7 / 33.0 GiB | **929 ms**, 19.2 / 33.5 GiB | **1.54 s**, 20.2 / 34.5 GiB |
| **This decoder, 4 GPUs** | **328 ms**, 17.8 / 41.4 GiB | **475 ms**, 18.3 / 34.1 GiB | **787 ms** (x18.8), 19.2 / 42.9 GiB |

`frames_per_pass` trades memory for speed and leaves the video unchanged. On one GPU:

| Latent frames per pass | 8 s | 12 s | 20 s |
|---|---|---|---|
| 2 | 1.25 s, 14.4 / 24.4 GiB | 1.88 s, 14.9 / 24.9 GiB | 3.13 s, 15.9 / 25.8 GiB |
| 4 (default) | 1.17 s, 20.7 / 36.3 GiB | 1.74 s, 21.2 / 36.8 GiB | 2.89 s, 22.2 / 37.8 GiB |
| 8 | 1.10 s, 33.7 / 62.1 GiB | 1.64 s, 34.2 / 62.6 GiB | 2.72 s, 35.2 / 63.6 GiB |
| 16 | 1.06 s, 59.8 / 121.0 GiB | 1.58 s, 60.2 / 121.7 GiB | 2.67 s, 61.2 / 122.5 GiB |

- With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the default reserves 30.6-32.0 GiB on one GPU and 2 frames
  per pass 19.6-21.1 GiB, at the same speed.
- On the optimization run's nine workloads (`tools/bench.py`: 121, 241 and 481 frames at 1, 2 and 4 GPUs, SM clocks
  1.6-1.95 GHz), the default's geomean is 758 ms, against 1043 ms for the run's first solution (x1.38) and 698 ms at
  16 frames per pass.
- The CUTLASS Winograd component products take 60 % of the GPU time, the transforms and epilogues 30 %
  (`notes/optimization.md`).

## Correctness

Against the fp32 decode of real latents, with `d = video - reference` per clip: every element finite and in [-1, 1],
max |d| at most three times what the release's bf16 decode errs by, and the RMS of d likewise. On this tree, every
clip at worlds 1, 2 and 4 and the short clips `f1`, `f5`, `f17` at worlds 1 and 4 pass, using 0.30-0.66 of the max
allowance and 0.35-0.42 of the RMS one (PSNR 56-60 dB against fp32; the release's bf16 decode has 58-61 dB).

```bash
cd fastvideo-kernel/wan22_vae_decode
DATA=/path/to/data python tools/bench.py --solution solution            # every workload this job has the GPUs for
DATA=/path/to/data python tools/bench.py --workload f5x1 f17x1 --no-time  # correctness of the short clips
cd .. && WAN22_VAE_WEIGHTS=/path/to/vae python -m pytest tests/test_wan22_vae_decode.py   # the wrapper, one GPU
```

`tools/bench.py` reads `$DATA`: the latents (`latents_<source>.pt`), the weights in `weight_names()` order
(`decoder_weights_bf16.pt`, `weight_names.json`), the fp32 decodes (`ref_<clip>.pt`) and the release decode's errors
(`calibration.json`); `tools/spec.py` documents each. The tests need only the VAE's weights.

## Limits

- sm_100a and sm_103a (B200 / GB200, B300): `tcgen05` with tensor memory, TMA and 2-CTA clusters; the loader builds
  for the GPU it runs on. CUDA 13, torch 2.12 tested; the tests pass on B300 too, built for sm_103a.
- 704 x 1248 video (latents 44 x 78) and batch 1; any T >= 1.
- 1, 2 or 4 GPUs of one node were checked; the code accepts up to 8 ranks.
- bf16 weights and activations, fp32 accumulation and fp32 RMS-norm statistics.
- In FastVideo, `WanVAEConfig.decode_kernel` routes `AutoencoderKLWan.decode` here where it applies (on one GPU per
  rank, `WanVAEDecoder(..., local=True)`), and PAN2 sampling turns it on (`PAN2PipelineConfig.vae_decode_kernel`). On
  a B300 with PAN2 latents it decodes 7x faster than the fp32 decode, at PSNR 61-63 dB and SSIM 0.998 against it.

## Layout

- `solution/`: the kernels and their binding; `solution.json` holds the build flags. The CUTLASS and CuTe headers
  come from fastvideo-kernel's CUTLASS submodule (`include/cutlass`, or `$CUTLASS_DIR`); the decoder's structure follows
  diffusers' (`WAN_LICENSE.txt`).
- `wan22_vae_decode.py`: `WanVAEDecoder`, `decoder_weights` and `weight_names`.
- `reference.py`: the definition of the result.
- `tools/`: `bench.py` (check, audit and time), `loader.py` (the build), `spec.py` (workloads, weights, data),
  `audit.py`, `bounds.py`.
- `notes/optimization.md`: the design, what each merge bought, what did not pay, and where to go next.
