# DGX Spark (GB10): Performance & Tuning

You have FastVideo [installed on a DGX Spark](spark.md) — this page is what to
run next. It covers **which models are practical on the GB10, what actually
makes them faster, and what won't help (and why)**, so you don't burn a night
tuning knobs that can't move on this hardware.

!!! tip "TL;DR"
    - **Use distilled few-step models** (e.g. `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`).
      They run in ~40 s/video. Full-step models are 12–47 min on the GB10.
    - On few-step models, **VAE decode is the bottleneck**, not attention — it's
      bandwidth-bound on the Spark's unified memory.
    - **bf16 VAE decode** is the real, lossless lever (FastVideo already turns it
      on for Wan). **FlashAttention, linear quantization, and `torch.compile` of
      the VAE give little or nothing here** — see the table below.
    - Heavy runs can make the box unreachable — run generations with VAE tiling on
      and `nice -n 19`. See [Running safely](#running-safely-dont-lock-the-box).

## The hardware reality (this explains everything below)

The GB10 pairs a Blackwell GPU (`sm_121`) with **128 GB of unified LPDDR5X memory
(~270 GB/s) shared between CPU and GPU**. That bandwidth is roughly **10× below a
datacenter GPU's HBM**. Two consequences drive every tuning decision:

1. **Memory-bandwidth-bound stages hurt disproportionately.** VAE decode moves a
   lot of data and becomes the dominant cost on short (few-step) generations.
2. **Compute-bound stages scale with step count.** Full-step diffusion (50+
   steps) is denoise-bound and simply takes a long time here.

## Use distilled few-step models

The single biggest lever on the GB10 is **model choice**. A 3-step distilled
model is ~18× faster than the full-step version of the same architecture:

| Model | Steps | Time / video | Bottleneck |
|---|---|---|---|
| FastWan2.1-T2V-1.3B (distilled) | 3 | **~40 s** | VAE decode |
| Wan2.1-T2V-1.3B (full-step) | 50 | ~12 min | denoise |
| Cosmos-Predict2.5-2B (full-step) | 51 | ~47 min | denoise |
| LTX2.3-distilled (+audio) | 8 | ~6 min | mixed |

The bottleneck flips from decode to denoise at around **4 steps**. Below that,
you're paying mostly for VAE decode; above it, mostly for the denoising loop.

!!! note "Few-step timings are noisy — measure in-process"
    On a 3-step run, one-time per-process startup (Triton autotune, allocator
    warmup) dominates and never amortizes, so single-run totals wobble ~±30%.
    Compare levers **back-to-back in one process or as medians**, never as two
    separate single runs. The [reproduction script](#reproduce-these-numbers)
    does this for you.

## bf16 VAE decode — the real lever (already on for Wan)

Because few-step generation is decode-bound, VAE decode precision is where the
time is. Decoding in **bf16 instead of fp32 is essentially lossless** (MS-SSIM
~0.9999 vs fp32 on the identical latent) and ~1.14× faster — worth roughly
5–7% end-to-end on a decode-bound few-step model.

**FastVideo already defaults Wan's decode to bf16** (`vae_decode_precision="bf16"`,
with encode kept at fp32), so for the recommended Wan/FastWan models there's
nothing to set. If you run a model that still defaults to an fp32 decode, set the
decode-only override yourself:

```python
from fastvideo.configs.pipelines.base import PipelineConfig

pipeline_config = PipelineConfig.from_pretrained(model_id)
pipeline_config.vae_decode_precision = "bf16"   # decode-only; leaves encode precision alone
```

Decode is output-only, so lowering its precision is safe. (Encode seeds the
denoising trajectory for I2V/causal models, so that stays at the pipeline's
default — don't lower `vae_precision` blindly for those.)

## Memory: one unified 128 GB pool

The GB10 has **no separate VRAM** — CPU and GPU share one 128 GB LPDDR5X pool
(~118 GB usable). Two practical consequences:

- **`nvidia-smi` reports memory as `[N/A]`** on the GB10, and the system "used"
  figure conflates CPU + GPU + cache, so it's only a soft upper bound — treat the
  whole 128 GB as one shared budget. For a per-run figure, use FastVideo's own
  `peak_memory_mb` (reported on the generation result and by the performance
  benchmark), which is measured inside the worker that runs the model.
- **The 128 GB is a *working-set* ceiling, not storage** — the model cache lives
  on the NVMe (3.7 TB, ample). What has to fit in 128 GB is weights + activations
  + KV, and — critically — the **VAE decode buffers**, which is why tiling matters
  (an untiled high-res decode can spike the pool into swap and lock the box).

The recommended few-step models are comfortable here: their weights are small
(1.3–2 B) and few-step generation keeps activations modest — a Wan2.1-1.3B
few-step generation peaks at **~8.4 GB** (measured), a small fraction of the pool.
The pressure comes from **decode resolution/frames**, not the model — a
1080p×121-frame untiled decode is what pushes the pool toward its ceiling, which
is why VAE tiling stays
on by default.

## What helps vs. what doesn't on the GB10

The honest summary — most "obvious" GPU optimizations don't move the needle on
this hardware, for reasons specific to it:

| Lever | Effect on the GB10 | Use it? |
|---|---|---|
| Distilled few-step model | ~18× vs full-step | ✅ **the primary lever** |
| bf16 VAE decode | ~1.14×, lossless; ~5–7% e2e on few-step | ✅ default for Wan |
| VSA (video sparse attention) | works out of the box (Triton kernel auto-selects on `sm_121`) | ✅ automatic |
| Building FlashAttention | **no speedup** — Torch SDPA already hits an efficient flash kernel on `sm_121`, and FA2 ties it | ❌ not worth building |
| `torch.compile` of the VAE decode | recompile storm (per-frame varying shapes) → ~1.1× | ❌ dead end |
| Linear (fp8 / nvfp4) quantization on long-sequence models (e.g. Cosmos) | ~nothing — see below | ❌ wrong lever here |
| FP4 attention (`ATTN_QAT_INFER`) | works on `sm_121` (#1598); helps, but needs a QAT-trained checkpoint | ⚠️ opt-in — see below |
| FP4 linear on short-sequence models (LTX2) | up to −24% denoise at 1080p (#1594) | ⚠️ model/resolution-dependent |

### Why linear quantization is the wrong lever on long-sequence models

Quantizing the linear (GEMM) layers is a natural first instinct, but on a
long-sequence video model it buys almost nothing on the GB10. A video-DiT denoise
step is dominated by **O(N²) attention** at these sequence lengths (tens of
thousands of tokens); the linear layers are a small single-digit fraction of the
work. Quantizing them faster leaves the attention-bound total essentially
unchanged — measured at ~1% on Cosmos-2.5, i.e. noise, and full-step CFG models
also lose quality to per-step quantization error.

The same mechanism **does** help on **short-sequence** models: LTX2's aggressive
VAE compression gives it short attention sequences, so FP4 linear reaches −24%
there (#1594). The rule: **on the GB10, the lever that matters is attention
(sparse or FP4), not the linear layers** — unless the model has short sequences.

### FP4 on the GB10 (opt-in)

Block-scaled FP4 works on `sm_121` under CUDA 13:

- **FP4 attention** (`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER`, #1598) is
  numerically correct on the GB10 and ~6% faster denoise, but it only preserves
  quality on a **quantization-aware-distilled checkpoint** (e.g.
  `FastVideo/FastWan-QAD-1.3B`) — stock weights aren't trained to tolerate it.
- **FP4 linear** helps only where sequences are short (LTX2, above).

The [`qad_fp4_ab.py`](#reproduce-these-numbers) harness reproduces the FP4
attention A/B on the QAD checkpoint.

## Running safely (don't lock the box)

The GB10 is easy to make **unreachable** — a heavy build or an untiled high-res
decode starves the ~20 ARM cores and unified memory, `sshd` can't get cycles, and
you're locked out at *"Connection timed out during banner exchange"* until the box
is power-cycled. To avoid it:

- **Inference:** keep **VAE tiling on** (the default), use sane resolution/frames,
  and run under `nice -n 19`:
  ```bash
  nice -n 19 nohup python your_script.py > run.log 2>&1 &
  ```
- **Builds** (flash-attn, kernel): `nice -n 19`, `MAX_JOBS=2`, `nohup`. Never a
  bare foreground high-parallelism build.
- Leave `*_cpu_offload` at the example defaults — "CPU" offload is the *same*
  unified RAM on the GB10, so the win is tiling + sane resolution, not offloading.

## Gotchas specific to the GB10

A few things that surprise people on this box (beyond the memory notes above):

- **Don't force `TORCH_SDPA` on a VSA checkpoint** (FastWan, LTX2.3-distilled).
  The SDPA path builds a model without the gate weights the checkpoint carries and
  fails to load. Run the model natively — VSA auto-routes to its Triton kernel on
  `sm_121`.
- **Few-step timings are noisy run-to-run** (~±30%) — one-time startup dominates a
  3-step run. Compare in-process / as medians, never two separate single runs (the
  benchmark script does this).
- **`nvidia-smi` shows `[N/A]` for memory** — see [Memory](#memory-one-unified-128-gb-pool).
- **Cosmos-2.5** uses a Qwen2.5-VL text encoder; make sure you're on a FastVideo
  build recent enough to include its `transformers`-compatibility handling before
  running it.

## Reproduce these numbers

Two scripts under `examples/inference/optimizations/` reproduce the claims on
your own GB10:

```bash
# Headline: few-step generation timing (median) + the bf16-vs-fp32 decode A/B.
# FASTVIDEO_STAGE_LOGGING=1 also prints the denoise / decode / text split.
FASTVIDEO_STAGE_LOGGING=1 nice -n 19 \
    python examples/inference/optimizations/spark_benchmark.py

# FP4 attention quality/speed A/B on the QAD checkpoint (one arm per run).
QAD_LINEAR=0 FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER nice -n 19 \
    python examples/inference/optimizations/qad_fp4_ab.py
```

See also the [Optimizations](../../inference/optimizations.md) reference for the
full list of attention backends and quantization options.
