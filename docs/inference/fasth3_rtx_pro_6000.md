# FastH3 NVFP4 on RTX PRO 6000 (sm_120)

This page covers serving the FastH3 NVFP4 checkpoints on one RTX PRO 6000
Blackwell (sm_120, 96 GB) with every component resident: the NVFP4 text
encoder, the NVFP4 denoiser and an INT8 light VAE. It also records what was
measured and tried along the way. Every switch is opt-in and defaults to the
existing behavior.

## Results

All results are for one RTX PRO 6000 (Modal), a 10.1 s clip at 1344x768
(243 frames, 73.6k packed tokens), the light INT8 VAE and warm runs.

| Checkpoint | Denoise | Video decode | End to end | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| [`FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) (4 forwards, VSA 0.9) | 30.1–31.0 s | 9.0 s | **41.2 / 42.4 s** | 79.6 GB |
| [`FastH3-8-Step-V2-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4) (8 forwards, VSA 0.8) | 75.6 s | 9.0 s | **86.5 s** | — |

These end-to-end runs predate the warp-skip kernel change below, which cuts
sparse attention by a further 1.5x, so they are upper bounds. Conditioning
takes about 0.12 s, frame post-processing plus MP4 writing about 0.9 s, and
the first clip at a new shape about 200 s (VAE compile).

At 480p (124 frames, 15.1k tokens) V2 8-step denoises in 12.5 s on a cold
run, down from 16.8 s warm before this work.

## Usage

### 1. Convert the checkpoint

The published NVFP4 checkpoints use ModelOpt's unified Hugging Face layout.
`convert_minimax_h3_modelopt_nvfp4_dit.py` repacks it into FastVideo's packed
export, `transformer/nvfp4_weights.safetensors`. The calibrated FFN weights
and scales are carried over bit for bit and only the scale bytes are
swizzled. Optionally it also quantizes the BF16 attention projections and the
VSA compression gates:

```bash
python scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py \
  --src /models/FastH3-8-Step-V2-NVFP4/transformer \
  --dst /models/fasth3-v2-fv/transformer \
  --quantize-attention --quantize-gate
```

Every exported linear is probed through the loader's `mm_fp4` path against a
BF16 matmul with the dequantized weight. Conversion refuses to write when any
error exceeds `--max-probe-error` (default 0.3; the converted V2 and 4-step
checkpoints probe at 0.14). The rest of the model folder (text encoder,
VAEs, schedulers, `fastvideo_inference.json`) is used as-is; the
NVFP4 text encoder comes from
`convert_minimax_h3_text_encoder_nvfp4.py`.

The flags select the `layer_profile` to load the result with:

| Flags | Linears in NVFP4 | `layer_profile` |
| --- | --- | --- |
| *(none)* | FFN `fc_in`/`fc_out` | `h3_dit_ffn` |
| `--quantize-attention` | + attention `to_{q,k,v,out}` | `h3_dit` |
| `--quantize-attention --quantize-gate` | + VSA `to_gate_compress` | `h3_dit_vsa` |

### 2. Generate

```python
import os
os.environ.update({
    "FASTVIDEO_H3_VSA_FP4": "1",              # sparse FP4 attention
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",    # Triton norm/modulate/RoPE/SwiGLU fusions
    "FASTVIDEO_NVFP4_MM_BACKEND": "cutlass",  # see "FP4 GEMM backend" below
    "FASTVIDEO_H3_VAE_TILE_BATCH": "28",      # one decoder call per 1344x768 tile grid
})
from fastvideo import VideoGenerator

generator = VideoGenerator.from_config({
    "model_path": "/models/fasth3-v2-fv",
    "engine": {
        "num_gpus": 1,
        "quantization": {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"},
        "compile": {"enabled": False, "vae_enabled": True},
    },
    "pipeline": {"experimental": {"attention_backend": "VIDEO_SPARSE_ATTN_H3",
                                  "VSA_sparsity": 0.8, "VSA_tile_size": 64}},
})
generator.generate({"prompt": "...", "sampling": {"height": 768, "width": 1344, "num_frames": 243,
                                                   "num_inference_steps": 9}})
```

Use `VSA_sparsity` 0.9 and `num_inference_steps` 5 for the 4-step checkpoint
(see its `fastvideo_inference.json`). Frame counts must be `17n + 5`: 243
frames is the closest to 10 s.

## What changed

### Block-sparse FP4 attention for VSA tiles (`fastvideo-kernel`)

SageAttention3's sm_120 FP4 kernel (`attn_qat_infer`) gains a block-sparse
forward, `fwd_sparse`, exposed as `sageattn_blackwell_sparse` (head-major
inputs) and `sageattn_blackwell_sparse_bshd` (sequence-major inputs, quantized
in place without a transpose). `vsa_tile_mask_to_fp4_blocks` turns a VSA tile
mask into the kernel's lists:

- **Block lists.** Query block `m` visits only the 128-token KV blocks in
  `q2k_idx[b, h, m, :q2k_num[b, h, m]]`.
- **Quadrant masks for 64-token tiles.** The kernel computes on 128x128
  blocks, but V2 and the 4-step preview use 64-token VSA tiles. Each listed
  block carries a 4-bit `q2k_quad` (one bit per 64x64 quadrant). The kernel
  masks unselected quadrants to `-inf`, so the result is exactly VSA's tile-64
  semantics.
- **Valid counts per 64-column half.** `kv_valid` gives the valid tokens in
  each 64-column half, so partially filled tiles can pad mid-block.
- **Warp-level skipping.** Each MMA warp owns 16 query rows and so sits
  inside one 64-row half. A warp skips a listed block that its half did not
  select, and the P·V chunk of a key half it did not select. Masked scores
  contribute exactly zero, and the warp sharing its tensor-core partition runs
  faster meanwhile. This recovers most of the work that pairing 64-token tiles
  into 128-token blocks adds.
- **First-visited block.** Lists run in descending block order because the
  kernel visits them last entry first. Block 0 (the first prefix tile, which
  VSA-H3's exempt mode gives every query) is therefore visited first, so every
  row starts from a finite running max. `validate=True` checks this. The model
  integration uses only exempt mode.

The dense and sparse entry points also stop allocating `delta_s`. With Q
smoothing off (the default), each call used to allocate and zero a
`[B, H, L/128, L]` fp32 tensor: 9.5 GB at 73k tokens. Its int32 batch stride
also overflowed the TMA descriptor ("Failed to initialize the TMA descriptor
1", then an illegal instruction), so FP4 attention could not run 10 s 1344x768
clips at all. A cached `[B, H, 1, L]` zero row read with `per_block_mean=False`
replaces it, and the outputs are bit-identical.

Correctness (`fastvideo-kernel/tests/test_attn_qat_infer_sparse.py`, RTX PRO
6000):

| Layout | Error vs token-masked fp32 | Dense FP4 floor |
| --- | ---: | ---: |
| 64-token tiles, odd count, partial tiles | 0.190 | 0.191 |
| 64-token tiles, even count | 0.191 | 0.193 |
| 256-token tiles, partial tile | 0.195 | 0.196 |

Errors are relative L2. The sparse kernel sits exactly at the FP4 noise
floor; random Gaussian inputs make that floor large. Full block lists
reproduce the dense kernel bit for bit.

### Model integration (`FASTVIDEO_H3_VSA_FP4=1`)

`fastvideo/models/dits/minimax_h3_vsa_fp4.py` replaces only the attention core
of `MiniMaxH3Attention`. VSA-H3's tile pooling, top-k mask, exempt prefix and
gated compression branch are unchanged. Per block, it:

1. Gathers the attention input into tile order once (one `hidden_size`-wide
   pass). Pad rows stay zero, so the q/k/v pad rows are exactly zero through
   the bias-free projections, RMSNorm and RoPE.
2. Quantizes that input once and shares it between `to_q`, `to_k` and `to_v`.
   NVFP4 activations use a unit global scale, so this is exact.
3. Applies QK-norm and RoPE with tile-ordered `cos`/`sin`, computed once per
   step.
4. Runs the sparse FP4 kernel on sequence-major tensors and gathers the output
   back to packed order before `to_out`.

This replaces the generic path's concat, four tile scatters and three
transposes. The route applies only to no-grad, non-compiled, single
sequence-parallel-rank calls in exempt mode; everything else keeps the
existing path.

Two smaller pieces ship alongside it:

- **Packed gate check.** With `--quantize-gate`, `to_gate_compress` loses its
  BF16 weight, so the gate-activity check reads the packed E2M1 bytes instead.
- **Compiled residual.** With `FASTVIDEO_MINIMAX_H3_FUSIONS` enabled, each
  block's final `hidden + gate[indices] * ffn_out` runs as one compiled op
  instead of materializing the gathered gate.

### FP4 GEMM backend (`FASTVIDEO_NVFP4_MM_BACKEND`)

FlashInfer's `mm_fp4(backend="auto")` picks a kernel about 2x slower than
`cutlass` or `cudnn` on sm_120 once activations reach tens of thousands of
rows. At 15k rows all three match.

| Linear | 73.6k rows: `auto` | 73.6k rows: `cutlass` | 73.6k rows: `cudnn` | 15.1k rows: `auto` |
| --- | ---: | ---: | ---: | ---: |
| `to_q` (5376→7168) | 7.96 ms | 3.96 ms | 4.24 ms | 0.88 ms |
| `to_out` (7168→5376) | 9.07 ms | 4.09 ms | 4.27 ms | 0.92 ms |
| `fc_in` (5376→28672) | 21.54 ms | 16.59 ms | 16.17 ms | 3.01 ms |
| `fc_out` (14336→5376) | 18.06 ms | 8.09 ms | 8.43 ms | 1.65 ms |

### Batched VAE tile decode (`FASTVIDEO_H3_VAE_TILE_BATCH`)

The H3 video VAE decodes 256-pixel spatial tiles one at a time. A 1344x768
clip is a 4x7 grid per temporal chunk, so a 10 s clip is roughly 400 small
decoder calls. The ViT decoder treats batch entries independently, so
`FASTVIDEO_H3_VAE_TILE_BATCH=N` decodes up to `N` equal-shaped tiles per call;
28 covers a full 1344x768 grid. The decoded tiles are the same as per-tile
decoding.

## Per-block measurements

One H3 transformer block (hidden 5376, 56 heads, FFN 14336), RTX PRO 6000:

| Component | 480p, 124 f (15.1k tokens) | 768p, 243 f (73.6k tokens) |
| --- | ---: | ---: |
| VSA Triton BF16 attention (kernel + pooling/mask) | 14.8 ms | 180.1 ms |
| Tile scatter of q/k/v/gate + gather (generic path) | 2.7 ms | 12.7 ms |
| Dense FP4 attention (SageAttention3) | 10.5 ms | 211.9 ms |
| Sparse FP4, quadrant masks | 7.9 ms | 123.4 ms |
| Sparse FP4, quadrant masks + warp skip | — | **83.0 ms** (VSA 0.8) / **49.3 ms** (VSA 0.9) |
| Dense BF16 SDPA | 18.1 ms | — |
| Modulation: eager / fused / compiled | 4.07 / 1.42 / 0.58 ms | 20.3 / 7.0 / 2.8 ms |
| SwiGLU: eager / fused | 1.53 / 0.88 ms | 7.37 / 4.24 ms |
| QK-norm + RoPE: eager / fused | 4.64 / 1.12 ms | 22.4 / 5.2 ms |

Before this work, a 480p block cost about 40 ms: 17.7 ms of attention, 12 ms
of linears and 10 ms of eager elementwise ops. Over 50 blocks that is 2.0 s
per step, which matches the measured 2.1 s.

Block density each kernel granularity computes at 768p (fraction of the
dense attention). "Selected" is what VSA needs; the other columns are what
each block shape computes:

| VSA sparsity | Selected (64x64) | 128x128 blocks | 64-row x 128 | 128 x 64-col |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | 0.222 | 0.434 | 0.317 | 0.313 |
| 0.9 | 0.125 | 0.254 | 0.180 | 0.177 |

## What was tried and not shipped

- **Dense FP4 attention for VSA-trained students.** `ATTN_QAT_INFER` does not
  build `to_gate_compress`. A VSA-distilled checkpoint such as V2 carries
  trained gates, so the strict loader refuses it ("Parameter
  ...to_gate_compress.weight not found"). Dense FP4 attention is also slower
  than sparse FP4 at 768p (212 vs 83 ms per block).
- **Multi-GPU (Ulysses) FP8 exchange.** On 8x RTX PRO 6000 (PCIe only), NCCL
  all-to-all moves about 21 GB/s per GPU, with NCCL P2P on or off. A BF16
  q/k/v/gate exchange at 73.6k tokens therefore costs 24 ms per block, and the
  attention output another 6.6 ms; with FP4 payloads q/k/v/gate drop to 7.2 ms.
  The branch `h3-sm120-experimental` keeps a sequence-parallel path that:
  - sends q/k/v as FP8 with one scale per token and head;
  - never sends the VSA gate, applying it on each rank after a small
    all-gather of the per-tile compression output;
  - returns the attention output as FP8.

  It is estimated at about 18–20 s per 10 s clip for V2 8-step on 8 GPUs. It
  has not executed yet (8-GPU capacity was unavailable), so it is not part of
  this change. The same branch holds the Modal drivers behind every number on
  this page.
- **64-row query blocks.** The kernel traits allow `kBlockM = 64`, which would
  remove the query-side pairing waste. Warp-level skipping recovers most of
  that waste without a second kernel instantiation, so it was not built.

## Known limitations

- **End-to-end quality.** The kernel matches a masked reference at the FP4
  noise floor. Generated videos have not yet been A/B-compared against the
  BF16 Triton VSA path on the H3 audio/video metrics.
- **Activation scales.** The packed export drops ModelOpt's calibrated static
  `input_scale` and quantizes activations with a unit global scale and dynamic
  per-16 block scales, as FastVideo's NVFP4 linears do elsewhere.
- **Decode cost.** Video decode (9 s at 10 s/768p with the light INT8 VAE) is
  the next largest cost after denoising.
- **Hardware.** Everything here targets sm_120. The GeForce RTX 5090 shares
  the architecture but has 32 GB, which needs a reduced-AdaLN checkpoint and a
  non-resident text encoder at this resolution.
