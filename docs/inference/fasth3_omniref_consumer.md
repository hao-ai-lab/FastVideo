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
| `rtx-4090` | RTX 4090, sm_89, 24 GB | 480p | as the 5090, with expandable segments to reduce fragmentation |

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
- **Attention:** VSA sparsity 0.9, 128-token tiles, reference keep 0.1. Video
  query tiles run on the FP4 sparse kernel - which quantizes Q, K, V and P, so
  those rows are FP4 end to end and use single-level P quantization. The dense
  prefix query tiles are split off and run densely at 8-bit Q/K with bf16 P·V,
  which keeps the text, reference and audio rows - the conditioning the clip is
  built from - out of the FP4 rounding. The split is single-rank for now: at
  `sp_size` above 1 every row goes through the FP4 kernel.
- **Text encoder:** the NVFP4 Qwen3-VL encoder, unchanged from the 8-step V2
  NVFP4 consumer release, including the vision tokens every reference adds.
- **Decoder:** the pre-packed NVFP4 decoder (`decoder-nvfp4.safetensors`), which
  is read straight onto the GPU without materializing the full-precision
  weights of its quantized linears.

The RTX 4090 profile requires a **dense or FP8 OmniRef export**, composed with
the base components. Use `--model-path <dense-or-fp8-export> --profile rtx-4090`, or
point the matching YAML recipe at its composed directory. A packed NVFP4 export
omits the quantized linears from its dense shards and cannot load as FP8.
The profile uses FP8 linears and the unquantized Triton emulator for the
checkpoint's fixed 128-token VSA tiles. It decodes with the VAE folder's dense decoder:
the NVFP4 decoder's GEMMs need FP4 hardware too, so `--decoder` does not apply
there.

To create the directory named by the RTX 4090 YAML recipe from a local dense export:

```bash
python examples/inference/basic/basic_fasth3_omniref_pdd.py \
  --model-path ./FastH3-OmniRef-Dense \
  --composed-dir ./FastH3-OmniRef-Dense-Composed \
  --profile rtx-4090 --adaln-table /path/to/adaln_table.pt \
  --image reference.png --prompt "a person speaking to camera"
```

Replace `./FastH3-OmniRef-Dense` with your dense or FP8 student export, including
its Ref2VA PDD inference contract.

### What this does and does not buy

Two things this first version does not do, so the numbers are not read as more
than they are:

- **The 8-bit prefix rows are simulated, not accelerated.** They are rounded to
  FP8 and then attended in bf16, so no FP8 tensor cores are involved. Since
  those rows are 55-65% of the attention, the speed-up is smaller than the
  24-33% the FP4 kernel measured on its own; the gain is quality, not time.
  Making it real needs an FP8-QK kernel variant.
- **The kernel does not reproduce the trained attention numerics exactly.** The
  student was trained with two-level P quantization and K smoothing; the
  released consumer kernel, as on `main`, uses single-level P and applies no K
  smoothing, and the prefix split does not change that for video rows. Closing
  that is a change to the kernel call, not to this path.

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

The FP4 attention route needs that kernel build. These are explicit alternatives;
a missing native kernel raises an error rather than selecting another mode:

1. **FP4 kernel + prefix split** (the profiles above): the native consumer route,
   with the numerical differences described above.
2. **`FASTVIDEO_H3_VSA_FINE_ATTENTION=emulator-mix8`**: Triton FP4 Q/K for video
   query tiles, per-token FP8 Q/K for prefix tiles, BF16 P.V and K smoothing.
   Needs no consumer-Blackwell extension but is slower. This Q/K-only mode has
   different numerics from the native kernel, which quantizes P.V as well.
3. **bf16 VSA**: set `FASTVIDEO_H3_VSA_FP4=0` and
   `FASTVIDEO_H3_VSA_FINE_ATTENTION=emulator-bf16`. This runs unquantized
   attention on the same 128-token tiles with Triton on consumer CUDA GPUs,
   including sm_89. Disabling FP4 alone leaves the generic tile-128 route,
   which requires the data-center sm_100a/sm_103a extension.

A nonempty fine-attention override takes priority over `FASTVIDEO_H3_VSA_FP4=1`,
including when a consumer profile sets that switch. Select it before constructing
the generator. The emulator replaces fine attention only; it preserves VSA
selection, partial-tile masking and the compression branch. These overrides use
the eager path; disable inference compilation when using them.

### Build FA4 for data-center Blackwell

The optional `fa4-mix8`, `fa4-mixmx` and `fa4-qk` overrides use a separate
FlashAttention dependency on sm_100/sm_103. The pinned revision and bundled patch,
including sparse FP4 support and per-token score scales for `fa4-mix8`, have
[installation instructions](https://github.com/aryan5v/FastVideo/blob/aryan/omniref-consumer-upstream/scripts/benchmarks/fasth3_omniref_b200/README.md#fa4-dependency-for-omniref).

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

Budget about 69 GiB for the DiT and encoder when both stay resident, before the
decoder and activations. The AdaLN table reduces the DiT's resident footprint.
Both the RTX 5090 and RTX 4090 profiles offload the encoder and load denoise
modules on demand; the 4090 also uses larger FP8 DiT weights instead of NVFP4.

These are estimates until the devices are in hand, apart from the encoder and
decoder figures, which were measured. See the device-test checklist in the
branch notes.

## Apple Silicon

`fastvideo/mlx_runtime` has an MLX MiniMax-H3 runtime, but its pipeline is
T2V/FL2VA-class: Ref2VA is only referenced in comments. Running OmniRef on a Mac
needs an MLX Ref2VA path (reference conditioning into the packed layout, the VSA
per-reference regions, the eight-forward PDD ladder) and an MLX build of the
NVFP4 decoder. Scoped, not implemented.
