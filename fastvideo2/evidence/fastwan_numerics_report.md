# FastWan variants — bitwise alignment vs fastvideo-main

Authority for FastWan artifacts is **fastvideo main** (they were distilled in
that stack); the alignment target was bit-exactness against main's own serving
path, pinned to main's exposed knobs so the goldens measure the artifact, not
the accelerator stack: `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN` (QAD) /
`VIDEO_SPARSE_ATTN` (VSA), FP8 per-tensor dynamic quant (QAD), no torch.compile,
no FSDP, single GB200.

Goldens captured at main commit `c459a1897899ffcec3be7765534d81000b9bb9c1`;
the vendored forward (`wan21/model_fv.py`) was read at `e3f47dc2de2d…` — all
10 numerics-relevant source files verified byte-identical between the two.

## Final anchor results (all rows target 0.0 — bitwise)

| row | fastwan-qad-fp8-1.3b | fastwan-t2v-1.3b (VSA) |
|---|---|---|
| dit bf16 probes t∈{1000,757,522} | 0.0 / 0.0 / 0.0 | — |
| dit fp8 probes t∈{1000,757,522} | 0.0 / 0.0 / 0.0 | — |
| dit vsa probes t∈{1000,757,522} | — | 0.0 / 0.0 / 0.0 |
| text_encoder (e2e + probe prompts) | 0.0 / 0.0 | 0.0 / 0.0 |
| e2e step-1 / step-2 latent chain | 0.0 / 0.0 | 0.0 / 0.0 |
| e2e final latents (81f, 480×832, 3 steps) | 0.0 | 0.0 |

Ledger: `anchor.fastwan-qad-main` and `anchor.fastwan-vsa-main`, both `pass`
(card digests `1c8e6f7d1380552d`, `0bd5c7771e10ce44`).

## Root causes found by the gates (in discovery order)

1. **fp8 quantization is device-sensitive.** main converts weights to fp8 on
   the GPU (post-materialization); quantizing the *identical* bf16 weights on
   CPU produces different fp8 codes often enough to move a full forward by
   ~4e-2 rel (fp8's coarse grid amplifies conversion-tie differences).
   Fix: `FP8Linear` defers quantization to first forward on the serving
   device (`layers/fp8.py`).
2. **0-dim sigma tensors demote the renoise mixing to bf16.** In torch type
   promotion 0-dim tensors act as scalars, so `(1-σ)*x0 + σ*ε` with a 0-dim
   fp32 σ ran in bf16 (two roundings); main's `[B,1,1,1]` fp32 σ promotes the
   arithmetic to fp32 with one final bf16 cast. 3.2e-3 per step, compounding
   to 8.2e-2 over 3 steps. Fix: non-0-dim σ in `WanDMDLoop`.
3. **main's DMD sigma table is NOT the one the code appears to prepare.**
   `DmdDenoisingStage.__init__` hardcodes a fresh internal
   `FlowMatchEulerDiscreteScheduler(shift=8.0)`; the pipeline scheduler that
   `TimestepPreparationStage.set_timesteps(n)` configured is never consulted,
   and the config `flow_shift` is ignored. Lookups run against the 1000-entry
   warped **init** table: σ(1000)=1.0, σ(757)=0.7567567, σ(522)=0.5217391
   (confirmed in the capture manifests). `dmd_inference_table` reproduces
   this exactly; a canary T0 test guards it.

Also confirmed en route: the CPU-generator RNG stream (initial fp32 draw +
bf16 renoise draws), the fp64 x0 math, and the flash/dense forward at full
81-frame geometry are each independently bitwise (triage decomposition in
session evidence).

## Caveats

- Text parity holds for ASCII prompts; main's ftfy cleaning diverges from
  official's on CJK width-folding (see wan21 report — main measured 4.19e-1
  vs official on the Chinese negative prompt). FastWan cards reuse the wan21
  text stage; DMD uses no negative prompt. Add main's clean fn + a CJK golden
  before serving non-ASCII prompts against these cards.
- VAE decode is not bitwise-gated (shared component; wan21 anchors cover it);
  golden videos are in the goldens dirs for SSIM-level comparison.
- Committed goldens are trimmed (per-step model outputs and the reproducible
  step-0 input dropped); the full set regenerates via
  `gates/capture_fastvideo_main.py {qad,vsa}` — one command, pinned config.
