# GPU bring-up — the real torch/CUDA backend

The `cuda` backend (`torch_cuda.py` + `torch_adapters.py` + `torch_kernels.py`) is **written but not
run**: this repo's dev environment has no GPU and no torch, so the torch code is grounded in the
verbatim real `fastvideo` APIs but **could not be executed or verified here**. Every spot the real
API must be confirmed on a box is marked `# BRINGUP` in the source and listed as a risk below.

This doc is the checklist for a human on a GPU box to take it from "resolves" to "generates".

## What it does

On a box with torch + CUDA + the parent `fastvideo` package installed, `Platform.detect()` returns a
`cuda` platform. The existing v2 loops/policies/scheduler/training are **unchanged**; only the
resolved implementations differ:

- **Components** resolve to torch adapters that wrap the real module named by the card's `load_id`
  (`fastvideo.models.dits.wanvideo:WanTransformer3DModel`, `…vaes.wanvae:AutoencoderKLWan`,
  `…encoders.t5`), loading weights from `ComponentSpec.checkpoint`. Each adapter bridges the real
  forward to the mini's duck-typed surface (`dit(latent,text,sigma)->velocity`, `vae.decode/encode`,
  `text_encoder.encode`), marshalling numpy↔torch at its boundary (the loop math stays numpy fp32).
- **Solver ops** (`flow_match_step`/`flow_sde_step`) resolve to plain torch elementwise — there is
  **no fused solver kernel** in fastvideo-kernel (it ships only attention/norm/quant primitives), so
  the honest source is "torch elementwise", registered at arch `generic`.

## Already verified on CPU (see `test_torch_backend.py`)

1. All three cuda components (`dit`, `vae`, `text_encoder`) + both solver ops are registered, with
   honest sources (no claimed `fastvideo-kernel:flow_*`), `available=False` here.
2. Importing the backends **never imports torch** (`torch_adapters`/`torch_kernels` load only inside
   builder bodies) — the CPU mini stays green.
3. On a (forced-available) cuda platform, resolution picks the **real cuda cells, not a silent toy**.
4. Building without torch **fails loudly** (not a quiet numpy toy decoding real latents).

## Ordered bring-up checklist (on the GPU box)

1. **Set weights.** Fill `ComponentSpec.checkpoint` for each component (HF id or local path) — e.g. a
   small helper that stamps a model root onto the wan21 card. *(Risk A — without this, builders raise.)*
2. **Detect.** `Platform.detect()` returns `cuda(smXY)`; confirm the arch string. `component_matrix()`
   shows the three cuda components `available=True`.
3. **Build in isolation.** Build each component; assert type + a single forward's output shape (no
   full denoise yet). Confirm the loader call (`from_pretrained` vs `TransformerLoader().load`). *(A)*
4. **One DiT step.** `dit(x, pe, sigma) -> velocity`; check finite + shape `[C,T,H,W]`. Confirm the
   `timestep = sigma*1000` convention *(B)* and that the output is **velocity (noise−clean)**, not x0
   or epsilon — a sign flip denoises backward. *(C)*
5. **One solver step.** `flow_match_step` finite + right shape (math mirrors `loop/sampler.py`).
6. **VAE.** `encode` → handle the `DiagonalGaussianDistribution` (`.mode()`) + `latents_mean/std`
   normalization to match the DiT's training latent scale; `decode` → video in `[-1,1]`. *(D)*
7. **Text.** Confirm `UMT5EncoderModel` vs `T5EncoderModel`, the tokenizer (`max_length=512`), and
   that the forward runs inside `set_forward_context(...)`. *(E)*
8. **End-to-end.** Full t2v denoise → VAE decode → compare to a known-good fastvideo generation
   (SSIM / the ssim regression harness).
9. **RL / SDE path.** `flow_sde_step` returns a finite `(prev, log_prob, mean, eff_std)`; run a
   rollout. Note: the **training** weight-surface (`mse_grad_step`) is NOT implemented on the cuda
   rung — RL/distill on GPU is a separate workstream. *(F)*
10. **CUDA-graph capture.** Last. The wan21 loop declares `breakable_cudagraph`; capturing a real
    torch graph (vs the numpy `StaticWorkspace` model) is GPU-only work.

## Risks / open unknowns (the `# BRINGUP` points)

| | Risk | Failure mode if wrong |
|---|---|---|
| **A** | checkpoint/config + exact loader call (`from_pretrained` vs `TransformerLoader().load`) | builder raises (blocking) |
| **B** | sigma→timestep scaling (`sigma*1000`, continuous vs discrete index) | silent garbage, not a crash |
| **C** | DiT output is velocity (noise−clean), correct sign/orientation | denoises backward |
| **D** | VAE returns a distribution + `latents_mean/std/shift` normalization | washed-out / saturated video |
| **E** | `UMT5EncoderModel` vs `T5EncoderModel`; tokenizer; `set_forward_context` | wrong/empty conditioning |
| **F** | torch training surface (`mse_grad_step`) not implemented | RL/distill on cuda is a separate workstream |
| **G** | numpy↔torch marshalling per DiT call (H2D/D2H + dtype round-trip) | correct but slow; torch-native surface is the perf follow-up |

Items A–E are correctness; F–G are scope/perf. None block the **CPU mini** (all gated `available=False`).
