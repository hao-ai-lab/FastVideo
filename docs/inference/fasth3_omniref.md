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

## At a glance

| | |
|---|---|
| Checkpoint | [FastVideo/FastH3-OmniRef-NVFP4-QAD](https://huggingface.co/FastVideo/FastH3-OmniRef-NVFP4-QAD) |
| Inputs | A prompt plus ordered references: up to 9 images, 3 videos and 3 audio clips, 12 in total |
| Output | Video with synchronized stereo audio at 24 fps |
| Clip length | 124, 243 or 362 frames (5, 10 or 15 seconds) |
| Sampling | 8 fused transformer forwards, guidance 1.0 |
| Local hardware | RTX 5090, RTX PRO 6000 and DGX Spark (NVFP4); RTX 4090 (FP8) |
| Cookbook | [MiniMax H3 cookbook](../cookbook/minimax-h3.md) |

## Composing the model directory

The NVFP4 checkpoint is
[FastH3 OmniRef QAD](https://huggingface.co/FastVideo/FastH3-OmniRef-NVFP4-QAD).
`basic_fasth3_omniref_pdd.py` accepts a complete checkpoint or a transformer-only
export. For an export containing `transformer_ref/`, the two scheduler configs
and `fastvideo_inference.json`, it composes the missing components from base
MiniMax-H3 into one directory of symlinks and runs the stack:

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
| `rtx-4090` | RTX 4090, sm_89, 24 GB | 480p | layerwise DiT stream (16 blocks resident), non-resident encoder, FFN chunk 16384, expandable segments |

The 96 GB and 128 GB profiles keep the DiT whole. The 32 GB 5090 also keeps it
resident but offloads the text encoder; the 24 GB 4090 streams the FP8 DiT
layerwise with 16 blocks resident. The 24 and 32 GB cards replace the AdaLN
projections (a third of the DiT's bytes on disk) with a precomputed table.

The `spark` and `rtx-5090` profiles enable the bit-identical lossless set by
default: exact fused kernels, heads-first VSA tiles, parallel VAE paths, and the
content-keyed reference memo. On a repeated reference this removes conditioning
and latent preparation from the warm path, worth 5-16% of warm wall time on
Spark and 3-5 s on the 5090; a first-time reference sees no regression.
`--lossless-accel` enables the same set on the other profiles.

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
Layerwise offload is mandatory: the 47 GB FP8 DiT cannot be resident on 24 GB.
The profile keeps 16 blocks resident, chunks the FFN at 16384 tokens and
disables pinned swap. Repeated requests in one process exceed the 24 GB GPU and
the 57.7 GiB host cgroup once the 16 GB text encoder reloads; run one process
per clip on this card.

The Blackwell YAML examples also select FP4 attention and the prefix precision
through `pipeline.experimental.h3_vsa_fp4`. Explicit YAML values take precedence
over the corresponding environment flags. A missing native extension raises an
error. To choose BF16 VSA for OmniRef's 128-token tiles, set
`h3_vsa_fp4.enabled: false` and
`FASTVIDEO_H3_VSA_FINE_ATTENTION=emulator-bf16`.

The Blackwell profiles select the FP8 prefix split to protect the conditioning
queries from FP4 rounding. The split runs prefix rows densely in bf16 after
8-bit Q/K rounding, which costs about 24% of
denoise time on one Spark. The split applies at `sp_size` 1 and above; the
sequence-parallel path keeps the same prefix and video row split.

`--attention-prefix bf16` is a higher-precision diagnostic mode, not a faster
one. Keeping every row on the FP4 kernel (`FASTVIDEO_H3_VSA_FP4_PREFIX=` empty)
is a low-level escape hatch, not a release mode: on a matched 480p/5 s clip it
sits 4.3 dB farther from the bf16 attention reference on every frame (20.96 vs
25.26 dB PSNR; SSIM 0.785 vs 0.852; structural correlation 0.864 vs 0.945).
Use the 36L decoder for the best-quality clip and the 8L decoder on memory-bound
devices; both run the FP8 split.

## Serving

`examples/serving/openai_fasth3_omniref_{rtx5090,rtx_pro6000,spark}.yaml` serve
the same stacks through the OpenAI-compatible API; their `generator` blocks match
the inference recipes above. Start one with the environment from its header:

```bash
fastvideo serve --config examples/serving/openai_fasth3_omniref_spark.yaml
```

Requests send ordered `image_reference`, `video_reference` and audio references.
The server reads local paths only under `server.media_roots`. These serving
configurations follow the maintained inference recipes and have not yet been
validated as servers on each device.

Select the NVFP4 decoder per configuration with
`generator.pipeline.experimental.h3_vae_nvfp4_decoder`, pointing at a file from
[FastVideo/FastH3-NVFP4-VAE-Decoder](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder);
`FASTVIDEO_H3_VAE_NVFP4_DECODER` remains the fallback. The composed directory
holds base MiniMax-H3's full 36-block VAE, so both the 36-block quality tier and
the 8-block fast tier apply.

## Continuing a video

A Ref2VA request can continue an earlier video. It takes the earlier video's last
5, 22 or 39 frames (and their sound, if any) as a continuation clip, and the
generated video starts by regenerating those frames at the same positions: the
clip's latent rows take the coordinates of the target's first latent frames and
audio latents, the reference context of
[ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum).
Drop the clip's frames, and the matching audio, from the new video, then append
the rest to the earlier one.

- Over HTTP, set `continuation_video` to the clip's URL or local path (under
  `server.media_roots`) beside the usual references. In Python, append
  `MiniMaxH3Reference(source=clip, media_type="continuation")` to `references`.
- The clip is resampled to 24 fps and must then have 5, 22 or 39 frames on the
  target canvas. A request takes at most one clip and still needs at least one
  reference; the clip does not count toward the reference limits and is not
  shown to the text encoder.
- `num_frames` counts the clip: 22 frames plus five new seconds is 142,
  aligned up to 158.
- Write the prompt from the clip's actual end state. When the prompt's opening
  contradicts the clip, the model tends to cut to a new shot right after it.
- Each continuation drifts a little toward higher saturation and sharpness, so
  fewer, longer segments (two 15-second ones rather than six 5-second ones) hold
  up better.

Measured on a FastH3 OmniRef PDD student at 1344×768, 22-frame clips re-encoded
from the earlier output: the jump at a join was about half that of conditioning
the next segment on the earlier last frame as an image reference, and the sound
continued instead of restarting from near silence.

## Environment

The profiles set these; set them yourself when driving the YAML recipes.

| Variable | Value | Why |
|---|---|---|
| `FASTVIDEO_H3_ADALN_TABLE` | path, every device | drops the 26 GB of `adaln_proj`; the 24/32 GB profiles need it |
| `FASTVIDEO_ATTENTION_BACKEND` | `VIDEO_SPARSE_ATTN_H3` | the trained backend; the contract fixes the tile size |
| `FASTVIDEO_H3_VSA_FP4` | `1` on Blackwell, `0` on the 4090 | FP4 sparse attention; sm_89 has no FP4 |
| `FASTVIDEO_H3_VSA_FP4_PREFIX` | `fp8` | the prefix split; `bf16` is the higher-precision variant |
| `FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P` | `1` on QAD profiles | the P quantizer QAD trained with |
| `FASTVIDEO_H3_VSA_FP4_K_SMOOTH` | `1` on QAD profiles | subtract the per-head key mean before FP4 quantization, as QAD trained |
| `FASTVIDEO_H3_GATE_CHUNK_ROWS` | `8192` on QAD profiles | bit-identical gate chunking; caps the 768p10 transient |
| `FASTVIDEO_H3_VSA_FINE_ATTENTION` | `emulator-bf16` on the 4090 | bf16 VSA on sm_89 |
| `FASTVIDEO_H3_VAE_NVFP4_DECODER` | path to `decoder-nvfp4.safetensors` | the pre-packed NVFP4 decoder |
| `FASTVIDEO_MINIMAX_H3_FUSIONS` | `all` | bit-exact fused kernels |
| `FASTVIDEO_VSA_TRITON` | `1` | Triton VSA on the consumer parts |
| `FASTVIDEO_VSA_PTX_BLACKWELL`, `FASTVIDEO_FA4` | `0` | no data-center kernel and no FlashAttention 4 |
| `FASTVIDEO_NVFP4_MM_ROW_SPLIT` | `table` on Spark, globally `off` | opt into the cuDNN row chunks measured on GB10 |
| `FASTVIDEO_NVFP4_MM_BACKEND` | `cudnn` on Spark | faster than CUTLASS on the H3 FFN shapes on GB10, bit-identical |
| `FASTVIDEO_H3_VAE_TILE_BATCH` | `12` Spark, `4` 5090/4090 | decoder tiles per call |
| `FASTVIDEO_H3_VAE_INT8_OVERLAY` | `0` on Spark | the INT8 ConvRot overlay is slower on GB10 |
| `FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS`, `FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS` | `1`, `16` on the 4090 | stream the packed FP8 weights; keep 16 blocks resident |
| `FASTVIDEO_H3_FFN_CHUNK_TOKENS`, `FASTVIDEO_H3_PINNED_SWAP` | `16384`, `0` on the 4090 | bound the FFN activation and avoid pinned host duplicates |
| `FASTVIDEO_ULYSSES_A2A`, `FASTVIDEO_MINIMAX_H3_EXACT_KERNELS`, `FASTVIDEO_H3_VSA_HEADS_FIRST_TILE`, `FASTVIDEO_H3_VAE_TILE_PARALLEL`, `FASTVIDEO_H3_REF2VA_MEMO_ENTRIES` | lossless set, on for Spark/5090 | bit-identical speed-ups; the memo helps repeated references |
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

The prefix rows are rounded to FP8 and attended in bf16; this protects their
precision rather than accelerating them. For the FP4 video-query kernel, the
Blackwell profiles enable two-level probability quantization
(`FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P=1`) and key smoothing
(`FASTVIDEO_H3_VSA_FP4_K_SMOOTH=1`), matching those settings in the OmniRef QAD
v2 training recipe. The kernel defaults to single-level probability
quantization without key smoothing when these switches are unset. Matching
these two settings does not establish exact parity for the entire attention
path, including the protected prefix and sequence-parallel exchanges.

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

Apple Silicon runs OmniRef natively through MLX, with a locally converted
INT8/INT6/INT4 DiT instead of NVFP4. See
[FastH3 OmniRef on Apple Silicon (MLX)](fasth3_omniref_mlx.md).
