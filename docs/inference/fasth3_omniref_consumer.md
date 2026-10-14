# FastH3 OmniRef on consumer devices

FastH3 OmniRef (Ref2VA) is a reference-conditioned PDD student: a prompt plus
ordered image/video/audio references become one video+audio clip in eight
fused transformer forwards. It has an extra conditioning prefix - the text, the
Qwen3-VL vision tokens of every reference, and any reference audio - on top of
the target video and audio rows.

This page covers running it on one consumer Blackwell device (RTX 5090, RTX PRO
6000, DGX Spark) at the full NVFP4 stack, and the closest equivalent on an RTX
4090, which has no FP4 hardware.

The checkpoint is a FastH3 OmniRef PDD export with a **packed NVFP4
`transformer_ref`**: NVFP4 linears for every block, with the blocks the
calibration sweep kept in bf16 (42-49) absent from the packed file and loaded
from the dense shards, plus static FFN activation scales. The export records its
own bf16 block list in the safetensors metadata, so nothing has to be declared
at load time.

## Composing the model directory

An OmniRef export carries only what distillation changed: `transformer_ref/`,
the two scheduler configs and `fastvideo_inference.json`. Everything else is
base MiniMax-H3. `examples/inference/basic/basic_fasth3_omniref_pdd.py` composes
them into one directory of symlinks and runs the stack:

```bash
python examples/inference/basic/basic_fasth3_omniref_pdd.py \
  --model-path ./FastH3-OmniRef-QAD120-NVFP4 \
  --image reference.png --audio voice.wav \
  --prompt "a person on a sofa, lit by a warm lamp, speaking to camera" \
  --profile spark
```

`--profile` is what makes the run device-appropriate. Without it the script
keeps its data-center defaults. The composed directory is what the example
recipes under `examples/inference/basic/basic_fasth3_omniref_*.yaml` point
`model_path` at.

## What each profile does

Every Blackwell profile is the same numerics, so a clip made on one device is
the same clip as on another. Only the memory policy differs.

| Profile | Device | Geometry | Offload |
|---|---|---|---|
| `spark` | DGX Spark, GB10, 128 GB unified | 480p | none: encoder, DiT and decoder all resident, as the released Spark recipe |
| `rtx-pro-6000` | RTX PRO 6000, sm_120, 96 GB | 768p | none: encoder, DiT and decoder all resident |
| `rtx-5090` | RTX 5090, sm_120, 32 GB | 480p | encoder not resident, denoise modules load on demand, AdaLN table |
| `rtx-4090` | RTX 4090, sm_89, 24 GB | 480p | as the 5090, with a hard allocator cap |

The DiT itself stays whole on every profile - no layerwise block streaming. The
32 and 24 GB cards only give up the text encoder, which is the one component
larger than the card, and the AdaLN projections (a third of the DiT's bytes on
disk), which the precomputed table replaces.

This is a lower memory stack than the earlier INT8 deployments: NVFP4 halves
every quantized weight against INT8, in the DiT, the encoder and the decoder,
and the 8-block `fast-8L` decoder is a further 4.5x smaller than the 36-block
one. The 32 GB card fit the INT8 stack before; NVFP4 makes it comfortable.

The NVFP4 stack the Blackwell profiles run is:

- **Linears:** NVFP4, from the packed export, with its bf16 blocks.
- **Attention:** VSA sparsity 0.9, 128-token tiles, reference keep 0.1, on the
  FP4 sparse kernel, with the dense prefix rows split off and run at 8-bit Q/K
  and bf16 P·V. The prefix split is what keeps the text, reference and audio
  rows - the conditioning the clip is built from - out of the FP4 rounding. It
  is single-rank for now: at `sp_size` above 1 every row still goes through the
  FP4 kernel.
- **Text encoder:** the NVFP4 Qwen3-VL encoder, unchanged from the 8-step V2
  NVFP4 consumer release, including the vision tokens every reference adds.
- **Decoder:** the pre-packed NVFP4 decoder (`decoder-nvfp4.safetensors`), which
  is read straight onto the GPU without materializing the full-precision
  weights of its quantized linears.

The RTX 4090 profile substitutes FP8 linears for NVFP4 and bf16 VSA attention
for the FP4 kernel, and decodes with the VAE folder's own dense light decoder:
the NVFP4 decoder's GEMMs need FP4 hardware too, so `--decoder` does not apply
there.

## Environment

The profiles set these; set them yourself if you drive the YAML recipes.

| Variable | Value | Why |
|---|---|---|
| `FASTVIDEO_H3_ADALN_TABLE` | path, every device | drops the 26 GB of `adaln_proj`; the 24/32 GB profiles need it to stay resident |
| `FASTVIDEO_ATTENTION_BACKEND` | `VIDEO_SPARSE_ATTN_H3` | the trained backend; the contract fixes the tile size |
| `FASTVIDEO_H3_VSA_FP4` | `1` | FP4 sparse attention (sm_120/sm_121) |
| `FASTVIDEO_H3_VSA_FP4_PREFIX` | `fp8` | the prefix split; `bf16` is the higher-precision variant |
| `FASTVIDEO_H3_VAE_NVFP4_DECODER` | `/path/decoder-nvfp4.safetensors` | the pre-packed NVFP4 decoder |
| `FASTVIDEO_MINIMAX_H3_FUSIONS` | `all` | bit-exact fused kernels |
| `FASTVIDEO_VSA_TRITON` | `1` | Triton VSA on the consumer parts |
| `FASTVIDEO_VSA_SM100A`, `FASTVIDEO_FA4` | `0` | no sm_100a kernel and no FlashAttention 4 here |
| `FASTVIDEO_NVFP4_MM_BACKEND` | `cudnn` on Spark | 2.7-3x faster than CUTLASS on the H3 FFN shapes on GB10, bit-identical |
| `FASTVIDEO_H3_VAE_TILE_BATCH` | `12` Spark, `4` 5090 | decoder tiles per call |
| `FASTVIDEO_H3_VAE_INT8_OVERLAY` | `0` on Spark | the INT8 ConvRot overlay is slower and less accurate on GB10 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` on the 4090 | keeps a 24 GB card from fragmenting |

`fastvideo-kernel` must be built for the device: `12.0a` on sm_120, `12.1a` on
GB10 (which needs CUDA 13). Building on the machine resolves the right one.

### Fallbacks, and what they cost

The FP4 attention route needs that kernel build. If it is missing or fails, the
attention falls back in this order, and never silently:

1. **FP4 kernel + prefix split** (the profiles above): fastest, the validated
   numerics.
2. **`FASTVIDEO_H3_VSA_FINE_ATTENTION=emulator-mix8`**: the Triton emulator of
   the same kernel, with the same prefix split. Needs no consumer-Blackwell
   kernel build, but is slower. Use it to confirm a bad clip is the kernel and
   not the model.
3. **bf16 VSA**: unset `FASTVIDEO_H3_VSA_FP4`. The reference quality, no FP4
   kernel needed. This is what an sm_89 device runs.

Dropping a component for a whole run is the last resort. If one NVFP4 component
fails a quality check, keep the rest and move the smallest piece - the way the
export already keeps blocks 42-49 in bf16.

## Memory

What each component costs, at 480p with one image reference. The AdaLN column is
why the 24 and 32 GB cards pass `FASTVIDEO_H3_ADALN_TABLE`; the same weights
without it are a third of the DiT and do not fit alongside the rest.

| Component | NVFP4, bf16 AdaLN | with the AdaLN table |
|---|---|---|
| DiT weights on disk | 44.5 GB | 18.5 GB |
| Qwen3-VL text encoder | 24.7 GiB (measured) | same |
| 36-block decoder, frozen | ~1.4 GiB | same |
| 8-block `fast-8L` decoder | 0.31 GiB (measured) | same |

So the resident set is ~45 GiB where the encoder can stay, and ~19 GiB where it
cannot. The 32 GB card is the only one of these that has to stream anything.

These are estimates until the devices are in hand, apart from the encoder and
decoder figures, which were measured. See the device-test checklist in the
branch notes.

## Apple Silicon

`fastvideo/mlx_runtime` has an MLX MiniMax-H3 runtime, but its pipeline is
T2V/FL2VA-class: Ref2VA is only referenced in comments. Running OmniRef on a Mac
needs an MLX Ref2VA path (reference conditioning into the packed layout, the VSA
per-reference regions, the eight-forward PDD ladder) and an MLX build of the
NVFP4 decoder. Scoped, not implemented.
