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

1. **Set weights + args.** Fill `ComponentSpec.checkpoint` for each component (HF id or local path),
   and provide the `FastVideoArgs` the real loaders need (`_fastvideo_args` builds a minimal one from
   the path — confirm its required fields/precision). *(Risk A — without these, builders raise.)*
2. **Detect.** `Platform.detect()` returns `cuda(smXY)`; confirm the arch string. `component_matrix()`
   shows the three cuda components `available=True`.
3. **Build in isolation.** Build each component via the FastVideo loaders (`TransformerLoader` /
   `VAELoader` / `TextEncoderLoader` + `TokenizerLoader`); assert type + a single forward's output
   shape (no full denoise yet). *(A)*
4. **One DiT step.** `dit(x, pe, sigma) -> velocity`; check finite + shape `[C,T,H,W]`. The
   `timestep = sigma*1000` convention and the velocity (noise−clean) semantics were cross-checked as
   MATCHING the real `forward`/scheduler — confirm numerically. *(B, C)*
5. **One solver step.** `flow_match_step` finite + right shape (math mirrors `loop/sampler.py`).
6. **VAE.** Normalization is now applied in-adapter (`(z-mean)*inv_std` on encode, inverse on decode,
   `latents_std` as reciprocal). Confirm the `shift_factor` placement/sign and dtype on the box. *(D)*
7. **Text.** Class resolution (UMT5 vs T5, from config) and `set_forward_context(...)` are now wired.
   Confirm the exact tokenizer kwargs (`text_len`/`max_length`, special tokens) from the model config. *(E)*
8. **End-to-end.** Full t2v denoise → VAE decode → compare to a known-good fastvideo generation
   (SSIM / the ssim regression harness).
9. **RL / SDE path.** `flow_sde_step` returns a finite `(prev, log_prob, mean, eff_std)`; run a
   rollout. Note: the **training** weight-surface (`mse_grad_step`) is NOT implemented on the cuda
   rung — RL/distill on GPU is a separate workstream. *(F)*
10. **CUDA-graph capture.** Last. The wan21 loop declares `breakable_cudagraph`; capturing a real
    torch graph (vs the numpy `StaticWorkspace` model) is GPU-only work.

## Risks / open unknowns (the `# BRINGUP` points)

Cross-checked against the real source (`crosscheck-gpu-adapters`): the **interface contracts matched**
(DiT returns a bare velocity tensor; `timestep=sigma*1000`; `encode().mode()` + bare `decode`;
`.last_hidden_state`; no fused solver kernel). The **construction layer was wrong and is now fixed**
in code (real loaders instead of the nonexistent `from_pretrained`; UMT5-vs-T5 resolved from config;
`set_forward_context` wired; latent normalization applied). What remains is genuinely box-dependent:

| | Risk | Failure mode if wrong |
|---|---|---|
| **A** | `FastVideoArgs` construction the real loaders need (exact fields/precision); checkpoint path | builder raises (blocking) |
| **B** | sigma→timestep scaling — cross-checked as `sigma*1000`; confirm numerically | silent garbage, not a crash |
| **C** | DiT output velocity (noise−clean) — cross-checked as matching; confirm sign on box | denoises backward |
| **D** | `shift_factor` placement/sign + dtype of the (now-applied) latent normalization | washed-out / saturated video |
| **E** | exact tokenizer kwargs (`text_len`/`max_length`, special tokens) from config — class + `set_forward_context` now fixed in code | wrong/empty conditioning |
| **F** | torch training surface (`mse_grad_step`) not implemented | RL/distill on cuda is a separate workstream |
| **G** | numpy↔torch marshalling per DiT call (H2D/D2H + dtype round-trip) | correct but slow; torch-native surface is the perf follow-up |

Items A–E are correctness; F–G are scope/perf. None block the **CPU mini** (all gated `available=False`).
Multi-GPU FSDP sharding via the loaders also needs on-box verification (single-GPU bring-up first).
