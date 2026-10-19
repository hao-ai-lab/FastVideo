# FastH3 OmniRef inference

FastH3 OmniRef (Ref2VA) is a reference-conditioned PDD student: a prompt plus
ordered image, video and audio references become one video and audio clip in
eight fused transformer forwards. Its extra conditioning prefix - the text, the
Qwen3-VL vision tokens of every reference and any reference audio - sits on top
of the target video and audio rows. See
[distilled FastH3 checkpoints](fasth3-distilled.md#ref2va-pdd-students) for the
sampling contract and checkpoint layout.

This page covers running the NVFP4 stack on one consumer Blackwell device
(RTX 5090, RTX PRO 6000, DGX Spark) and the closest equivalent on an RTX 4090,
which has no FP4 hardware.

## Composing the model directory

An OmniRef export carries only what distillation changed: `transformer_ref/`,
the two scheduler configs and `fastvideo_inference.json`. Everything else is
base MiniMax-H3. `basic_fasth3_omniref_pdd.py` composes them into one directory
of symlinks and runs the stack:

```bash
python examples/inference/basic/basic_fasth3_omniref_pdd.py \
  --model-path <OmniRef export directory or HF repo id> \
  --image reference.png --audio voice.wav \
  --prompt "a person on a sofa, lit by a warm lamp, speaking to camera" \
  --profile spark
```

`--profile` selects the device's offload policy, geometry and environment.
Without it the script keeps its data-center defaults. Each profile also has a
recipe under `examples/inference/basic/basic_fasth3_omniref_*.yaml`; point its
`model_path` at the composed directory and run it with
`fastvideo generate --config <recipe>`.

The Blackwell profiles need a packed NVFP4 OmniRef export. The RTX 4090 needs a
dense or FP8 export: a packed NVFP4 export omits the quantized linears from its
dense shards and cannot load as FP8.

Pass `--adaln-table /path/to/adaln_table.pt` to drop the AdaLN projections (the
24/32 GB cards need it to keep the DiT resident), and
`--decoder /path/to/decoder-nvfp4.safetensors` to swap in the pre-packed NVFP4
decoder. Write an AdaLN table once with `FASTVIDEO_H3_ADALN_DUMP`; build a
decoder file with `scripts/distill/minimax_h3_nvfp4_decoder/export_deploy.py`.

## Device profiles

Every Blackwell profile runs the same numerics, so a clip is the same clip on
every device. Only the memory policy and the geometry differ.

| Profile | Device | Geometry | Offload |
|---|---|---|---|
| `spark` | DGX Spark, GB10, 128 GB unified | 480p | none: encoder, DiT and decoder resident |
| `rtx-pro-6000` | RTX PRO 6000, sm_120, 96 GB | 768p | none: encoder, DiT and decoder resident |
| `rtx-5090` | RTX 5090, sm_120, 32 GB | 480p | non-resident encoder, sequential denoise-module load, AdaLN table |
| `rtx-4090` | RTX 4090, sm_89, 24 GB | 480p | as the 5090, plus expandable segments |

The DiT stays whole on every profile. The 24 and 32 GB cards offload the text
encoder and replace the AdaLN projections (a third of the DiT's bytes on disk)
with a precomputed table.

The Blackwell stack is:

- **Linears:** NVFP4 from the packed export. Blocks the calibration kept in
  bf16 are absent from the packed file and load from the dense shards; the
  export records its own skip list in the safetensors metadata.
- **Attention:** VSA sparsity 0.9, 128-token tiles, reference keep 0.1. Video
  query tiles run on the FP4 sparse kernel. Dense prefix query tiles - the
  text, reference and audio rows - are split off and run at 8-bit Q/K with bf16
  P.V, keeping the conditioning out of the FP4 rounding. The split applies on
  the single-rank and sequence-parallel paths alike.
- **Text encoder:** the NVFP4 Qwen3-VL encoder, including reference vision
  tokens.
- **Decoder:** the pre-packed NVFP4 decoder (`decoder-nvfp4.safetensors`), read
  straight onto the GPU without materializing full-precision weights.

The RTX 4090 profile uses FP8 linears, the unquantized Triton emulator for the
checkpoint's fixed 128-token VSA tiles and the VAE folder's dense decoder. The
NVFP4 decoder's GEMMs need FP4 hardware, so `--decoder` does not apply there.

The FP8 prefix split is the release mode: it reproduces the attention numerics
the student was trained with, and every device profile selects it. It runs the
prefix rows densely in bf16 after 8-bit Q/K rounding, which costs about 24% of
denoise time on one Spark. The split applies at `sp_size` 1 and above; the
sequence-parallel path keeps the same prefix and video row split.

`--attention-prefix bf16` is a higher-precision diagnostic mode, not a faster
one. Keeping every row on the FP4 kernel (`FASTVIDEO_H3_VSA_FP4_PREFIX=` empty)
is a low-level escape hatch, not a release mode: on a matched 480p/5 s clip it
sits 4.3 dB farther from the bf16 attention reference on every frame (20.96 vs
25.26 dB PSNR; SSIM 0.785 vs 0.852; structural correlation 0.864 vs 0.945).
Use the 36L decoder for the best-quality clip and the 8L decoder on memory-bound
devices; both run the FP8 split.

## Environment

The profiles set these; set them yourself when driving the YAML recipes.

| Variable | Value | Why |
|---|---|---|
| `FASTVIDEO_H3_ADALN_TABLE` | path, every device | drops the 26 GB of `adaln_proj`; the 24/32 GB profiles need it |
| `FASTVIDEO_ATTENTION_BACKEND` | `VIDEO_SPARSE_ATTN_H3` | the trained backend; the contract fixes the tile size |
| `FASTVIDEO_H3_VSA_FP4` | `1` | FP4 sparse attention (sm_120/sm_121) |
| `FASTVIDEO_H3_VSA_FP4_PREFIX` | `fp8` | the prefix split; `bf16` is the higher-precision variant |
| `FASTVIDEO_H3_VAE_NVFP4_DECODER` | path to `decoder-nvfp4.safetensors` | the pre-packed NVFP4 decoder |
| `FASTVIDEO_MINIMAX_H3_FUSIONS` | `all` | bit-exact fused kernels |
| `FASTVIDEO_VSA_TRITON` | `1` | Triton VSA on the consumer parts |
| `FASTVIDEO_VSA_PTX_BLACKWELL`, `FASTVIDEO_FA4` | `0` | no data-center kernel and no FlashAttention 4 |
| `FASTVIDEO_NVFP4_MM_BACKEND` | `cudnn` on Spark | faster than CUTLASS on the H3 FFN shapes on GB10, bit-identical |
| `FASTVIDEO_H3_VAE_TILE_BATCH` | `12` Spark, `4` 5090/4090 | decoder tiles per call |
| `FASTVIDEO_H3_VAE_INT8_OVERLAY` | `0` on Spark | the INT8 ConvRot overlay is slower on GB10 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` on the 4090 | reduces allocator fragmentation |

`fastvideo-kernel` must be built for the device: `12.0a` on sm_120 and `12.1a`
on GB10 (CUDA 13). Building on the machine resolves the right one.

### Fallbacks

The FP4 route needs that kernel build. These are explicit alternatives; a
missing native kernel raises rather than silently selecting another mode.

1. **FP4 kernel + prefix split** (the profiles above).
2. **`FASTVIDEO_H3_VSA_FINE_ATTENTION=emulator-mix8`**: Triton FP4 Q/K for video
   query tiles, per-token FP8 Q/K for prefix tiles, bf16 P.V and K smoothing.
   Needs no consumer-Blackwell extension but is slower, and its numerics differ
   from the native kernel (which also quantizes P.V).
3. **bf16 VSA**: set `FASTVIDEO_H3_VSA_FP4=0` and
   `FASTVIDEO_H3_VSA_FINE_ATTENTION=emulator-bf16`. This runs unquantized
   attention on the same tiles with Triton on consumer CUDA GPUs, including
   sm_89.

A nonempty fine-attention override takes priority over
`FASTVIDEO_H3_VSA_FP4=1`. These overrides use the eager path; disable inference
compilation when using them.

Two limits of this first version are worth knowing. The 8-bit prefix rows are
simulated: they are rounded to FP8 and attended in bf16, so the gain is quality,
not time. And the released kernel does not reproduce the trained attention
numerics exactly: the student was trained with two-level P quantization and K
smoothing, while the shipped kernel uses single-level P without K smoothing.

## Memory

What each component costs at 480p with one image reference. The AdaLN column is
why the 24 and 32 GB cards pass `FASTVIDEO_H3_ADALN_TABLE`.

| Component | NVFP4, bf16 AdaLN | with the AdaLN table |
|---|---|---|
| DiT weights on disk | 44.5 GB | 18.5 GB |
| Qwen3-VL text encoder | 24.7 GiB | same |
| 36-block decoder, frozen | ~1.4 GiB | same |
| 8-block `fast-8L` decoder | 0.31 GiB | same |

Budget about 69 GiB for the DiT and encoder when both stay resident, before
the decoder and activations. The encoder and decoder figures are measured; the
rest are estimates. Both the RTX 5090 and RTX 4090 offload the encoder and load
denoise modules on demand; the 4090 also uses the larger FP8 DiT weights.

## Multiple Sparks

Two Sparks run the same stack with `--num-gpus 2`, Ray and sequence
parallelism over a direct link (`basic_fasth3_omniref_spark_pair.yaml`).
Sequence parallelism cannot use three ranks: MiniMax-H3 has 56 attention
heads, so `sp_size` must divide 56. For three Sparks, enable the encoder split
(`FASTVIDEO_H3_ENCODER_SPLIT=1`, `FASTVIDEO_H3_ENCODER_NODES=1`): one node
runs the Qwen3-VL conditioner and the other two run the DiT and VAEs with
`sp_size=2`. The prompt-embedding broadcast still spans all three ranks, so
all three nodes need a common fast link; the split does not fit a triangle of
separate point-to-point subnets without routes between them.

## Apple Silicon

`fastvideo/mlx_runtime` has an MLX MiniMax-H3 runtime, but its pipeline is
T2V/FL2VA-class. Running OmniRef on a Mac needs an MLX Ref2VA path and an MLX
build of the NVFP4 decoder; it is not implemented.
