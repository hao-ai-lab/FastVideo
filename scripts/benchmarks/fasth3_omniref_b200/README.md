# FastH3 OmniRef: lossless speed-ups on 4x B200

`examples/inference/basic/basic_fasth3_omniref_pdd.py --lossless-accel` turns on
every bit-identical speed-up the MiniMax-H3 Ref2VA pipeline has. The output is
bitwise the same as without the flag: the same uint8 video frames and the same
float audio samples. This directory holds the case and the script that measure
it.

## The case

One 5 s clip (124 frames at 24 fps), 1344x768, one reference image, 4x B200,
four-way sequence parallel (`--num-gpus 4`).

| field | value |
| --- | --- |
| checkpoint | `FastVideo/FastH3-OmniRef-v5-DMPDD8-w03-cfg2-step3500` and the base MiniMax-H3 its `fastvideo_inference.json` pins |
| prompt | `Picture 1 walks slowly toward the camera across the living room, smiling, and says, "Every detail was chosen with care." Smooth tracking shot, warm afternoon light, light footsteps on the wooden floor.` |
| reference | `speaker.png`, 720x400 RGB, sha256 `d38ba667...a6c92c` (shared with the PR; not committed) |
| seed | 413410 |
| sampling | the contract's 8 fused PDD blocks, guidance 1.0, no CFG batch |

`case_speaker.json` holds the same values. The script checks the reference
image's sha256 before it runs.

## Method

```bash
python scripts/benchmarks/fasth3_omniref_b200/bench_lossless.py --mode off \
    --model-path FastVideo/FastH3-OmniRef-v5-DMPDD8-w03-cfg2-step3500 \
    --reference speaker.png --num-gpus 4 --out runs/off
python scripts/benchmarks/fasth3_omniref_b200/bench_lossless.py --mode on \
    --model-path FastVideo/FastH3-OmniRef-v5-DMPDD8-w03-cfg2-step3500 \
    --reference speaker.png --num-gpus 4 --out runs/on
python scripts/benchmarks/fasth3_omniref_b200/bench_lossless.py --compare runs/off runs/on
```

Run with `FASTVIDEO_STAGE_LOGGING=1` for per-stage times. On a machine whose
NVLink SHARP setup NCCL cannot use, also set `NCCL_NVLS_ENABLE=0` (both modes).

Each mode runs in its own process, because the speed-ups are environment flags
that workers read when they start. Both modes build the generator through the
example (the flag is the only difference). Each mode then does three runs:

1. an untimed warm-up with another prompt and a grey reference, so CUDA, NCCL
   and Triton are warm but the reference cache holds nothing from the case;
2. the case, timed (`cold`: the prompt and reference have not been seen);
3. the case again, timed (`repeat`: with the flag on, the reference-encode
   cache hits).

The timing is the wall time of `generate()`: encoders, denoising, VAE decode
and the transfer of the frames to the caller, without the mp4 write. The
script hashes the decoded uint8 frames and the raw float audio with SHA-256,
and `--compare` fails unless every hash matches between the two modes.

## What `--lossless-accel` turns on

| setting | what it does | why the bits do not change |
| --- | --- | --- |
| `offload.text_encoder=False`, `offload.vae=False` | keeps Qwen3-VL (63 GB) and the VAEs on the GPU instead of copying them from host memory for every clip | same weights, same kernels |
| `pin_cpu_memory=True` | pinned host buffer for the decoded clip | copy only |
| `vae_parallel_decode` / `vae_parallel_encode` | the existing chunk-parallel VAE paths | already proven equal to serial |
| `FASTVIDEO_ULYSSES_A2A=auto` | the existing fused NVLink Ulysses all-to-all | data movement only |
| `FASTVIDEO_MINIMAX_H3_EXACT_KERNELS=all` | Triton RoPE, AdaLN modulate / gated residual and SwiGLU kernels | rounds to bf16 exactly where the eager ops do |
| `FASTVIDEO_H3_VSA_HEADS_FIRST_TILE=1` | VSA tiles scattered straight into the kernels' heads-first layout | data movement only |
| `FASTVIDEO_H3_VAE_TILE_PARALLEL=1` | VAE spatial tiles split across the ranks (decode and keyframe encode) | same per-tile forwards, same stitch |
| `FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=16` | reuses repeat Qwen3-VL presentations and keyframe latents, keyed by content | pure functions of their inputs |

The exact kernels are not `FASTVIDEO_MINIMAX_H3_FUSIONS`. Those fusions keep
FP32 across a chain and round once, which is more accurate but changes about
95% of the output pixels. The exact kernels do each eager op's bf16 rounding,
in the same order. They use PTX `mul.rn`/`add.rn`/`div.rn`, because Triton
would otherwise contract a multiply-add into an FMA or fold the intermediate
roundings away. If both are on, the fusions take precedence.

## Results

B200 (180 GB), the case above, PyTorch 2.12.0+cu130, `NCCL_NVLS_ENABLE=0`.

4x B200. These were measured with the same techniques applied as run-time
patches over FastVideo `9edc8adf` plus the OmniRef PR, before they were ported
here:

| run | off | on | speed-up | bits |
| --- | --- | --- | --- | --- |
| cold | 42 s | 14.7 s | 2.9x | identical |
| repeat | 42 s | 13.1 s | 3.2x | identical |

Where the time went (seconds, cold, off -> on): conditioning 15-17 -> 1.1,
reference VAE encode 2.5 -> 0.43, denoising 13.8 -> 11.0, VAE decode 7.5 -> 1.4.

This branch, with this script:

| GPUs | run | off | on | speed-up | bits |
| --- | --- | --- | --- | --- | --- |
| 1 | cold | 67.1 s | 44.4 s | 1.51x | identical |
| 1 | repeat | 67.2 s | 42.1 s | 1.60x | identical |
| 2 | cold | 54.9 s | 26.6 s | 2.06x | identical |
| 2 | repeat | 54.6 s | 24.9 s | 2.19x | identical |

Two GPUs, cold, off -> on: conditioning 17.0 -> 1.1 s (0.06 s on repeat),
reference VAE encode 2.7 -> 0.8 s (0.06 s), denoising 26.7 -> 21.5 s, VAE
decode 7.8 -> 2.7 s. All four runs and both GPU counts give the same digests
(frames `a48362b9...`, audio `0f0d42f8...`). One GPU runs none of the multi-rank
paths (all-to-all, VAE splits), and its denoising dominates (46.6 -> 36.2 s),
so the gain is smaller there.
