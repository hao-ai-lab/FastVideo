# Handoff — GPU bring-up of the v2 torch backend

**For: an agent on a GPU box, branched from `will/mini-fastvideo`.**
**Your job:** take the *written-not-run* `cuda` backend to *runs-and-generates*, then commit + push.

Everything below is committed on `will/mini-fastvideo` and CPU-tested (**204 tests pass**). The torch
path was authored on a machine with **no GPU and no torch**, so it is grounded in the real
`fastvideo` APIs and cross-checked against the source, but **never executed**. That's what you finish.

---

## 0. Orientation (read these first, in order)

1. **`v2/README.md`** — what the whole v2 mini is (the `(recipe, runtime)` runtime; "architecture is
   real, kernels are toys"). The "Honest scope" paragraph says exactly what's wired.
2. **`v2/platform/backends/GPU_BRINGUP.md`** — *your checklist*: the ordered 10-step bring-up + the
   risk table (A–G), each tied to a `# BRINGUP` marker in the source. **This handoff is orientation +
   process; GPU_BRINGUP.md is the work.**
3. This file — the meta-instructions (verify bar, commit/push, gotchas).

## 1. What's already done (commits on this branch)

```
d6d0580a [fix]  correct GPU adapters against real fastvideo API (cross-check findings)
b8d78f40 [feat] real torch/CUDA backend (written-not-run) behind the cuda cells
27791b51 [feat] static-buffer capture form for the cudagraph step body (Path A)
ae6a170d [feat] piecewise CUDA-graph capture/replay at the step boundary (Path A)
9308d87e [feat] route diffusion loops through the kernel table
7490c590 [feat] multi-backend dispatch substrate (device/arch/kernel registries)
```

The dispatch substrate (two tuple-keyed registries `COMPONENTS(kind,device,variant)` +
`KERNELS(op,device,arch,variant)`, a detected `Platform`, numpy terminal + parity oracle), the
universal kernel seam (diffusion loops go through `model.platform.kernels`), the
piecewise cudagraph lifecycle, and the torch backend cells are all in place. On a GPU box,
`Platform.detect()` returns a `cuda` platform and resolves the torch cells instead of the numpy toys —
**the inference loops/policies/scheduler are unchanged**; only the resolved implementations differ.

## 2. The files you'll touch

| File | What it is |
|---|---|
| `v2/platform/backends/torch_adapters.py` | `TorchWanDiT` / `TorchWanVAE` / `TorchT5Encoder` — wrap the real `fastvideo.models.*` (named by each card's `load_id`) to the mini's duck-typed surface. Built via the real FastVideo loaders. |
| `v2/platform/backends/torch_kernels.py` | torch `flow_match_step` / `flow_sde_step` (plain elementwise — there is **no** fused solver kernel in fastvideo-kernel; don't look for one). |
| `v2/platform/backends/torch_cuda.py` | registers the `cuda` cells as lazy trampolines (torch imported only inside builder bodies). |
| `v2/card/specs.py` | `ComponentSpec.checkpoint` — the per-component weights source (empty on toys; **you fill it in**). |

The surface the adapters must honor (what the loops call):
`dit(latent, text_embed, sigma) -> velocity` · `vae.decode(latent)` / `vae.encode(video)` ·
`text_encoder.encode(text)`. The CPU toys in `v2/models/backend.py` are the reference behavior.

## 3. Your task (the gating items — full detail in GPU_BRINGUP.md)

1. **Env:** install `torch` + the parent `fastvideo` package + weights. (`fastvideo` source lives at
   `/Users/willlin/src/FastVideo`.)
2. **Risk A — the one blocking gap:** the builders call `_load_via_fastvideo(...)` → the real loaders
   need a **`FastVideoArgs`**, which `_fastvideo_args(spec)` builds minimally from `spec.checkpoint`.
   Confirm/extend its fields (model config, precision, parallelism). And stamp `ComponentSpec.checkpoint`
   onto the wan21 card — a tiny helper that maps a model root onto the three components is the cleanest
   way (the toy cards leave it `""`).
3. **Work the risk list (A–G in GPU_BRINGUP.md).** The *interface* contracts were cross-checked as
   matching (DiT returns bare velocity; `timestep=sigma*1000`; `encode().mode()`; `.last_hidden_state`;
   no fused solver kernel) — confirm them numerically. The *construction* layer was fixed (real loaders,
   `set_forward_context`, latent normalization, UMT5-from-config). What's left is box-dependent:
   `FastVideoArgs` fields, `shift_factor` placement/sign, exact tokenizer kwargs, FSDP sharding.
4. **Bring up in order:** build each component in isolation → one DiT step → one solver step → VAE
   decode → full t2v → SDE stochastic sampling → cudagraph capture (last).

## 4. The verification bar (how you know it's right)

- **CPU suite must stay green:** `python3 -m pytest v2/ -q` → still **204 passed**. The torch path is
  gated `available=False` off-GPU; importing the backends must never import torch. If you break either,
  you broke the substrate. (`v2/tests/test_torch_backend.py` pins these.)
- **Parity oracle is the spec:** the substrate's whole point is that a real backend matches the numpy
  reference on the consistency ladder. On GPU, compare a full generation against a known-good fastvideo
  output — use the parent repo's SSIM regression harness (`fastvideo/tests/ssim/`). Target C4 (SSIM /
  artifact quality); component/trajectory parity (C0/C1) is bit-level vs the reference pipeline.
- **Don't trust "it ran" — trust "it matched."** A wrong `timestep` scale or `shift_factor` produces
  plausible-but-wrong video, not a crash (risks B/D). Diff against a reference, don't eyeball.

## 5. Commit + push

- **You are on a GPU branch** (branched from `will/mini-fastvideo`). Commit your bring-up fixes there,
  focused by concern (e.g. one commit per confirmed risk), in the existing style (`[fix]`/`[feat] …`).
- **NEVER add Claude as a co-author** (repo policy, `/Users/willlin/src/.claude/CLAUDE.md`).
- **Do not rewrite or force-push** the six commits above — build on top.
- When the CPU suite is green **and** a GPU generation matches the reference, **push your branch.**
- If you launch inference with wandb logging enabled, log in with the token in the project
  `CLAUDE.md` (`/Users/willlin/src/.claude/CLAUDE.md`) — **do not paste it into any committed file.**

## 6. Gotchas (don't relearn these the hard way)

- **No fused solver kernel exists.** `fastvideo-kernel` ships only attention/norm/quant primitives;
  the cuda `flow_match_step`/`flow_sde_step` are plain torch by design. Don't hunt for a `.cu` solver.
- **`from_pretrained` is not the loader.** `WanTransformer3DModel`/`AutoencoderKLWan` have none — the
  real path is the `*Loader().load(model_path, fastvideo_args)` classes in
  `fastvideo/models/loader/component_loader.py`. The loader resolves the class from the checkpoint
  config (this is what makes UMT5-vs-T5 correct without hardcoding).
- **T5 needs `set_forward_context`.** A bare encoder forward reads stale/None global context.
- **The loop surface stays numpy** for bring-up; adapters marshal numpy↔torch at the boundary. A
  torch-native surface (latent on-device through forward→combine→solver) is the **perf follow-up**
  (Risk G), not bring-up — don't rewrite `cfg.combine`/`precision.cast`/the samplers yet.
- **cudagraph capture is last.** The wan21 loop declares `breakable_cudagraph`; the v2 capturer models
  the lifecycle with a numpy `StaticWorkspace`. Capturing a real `torch.cuda.CUDAGraph` is GPU-only
  work and the riskiest step — leave it until inference is verified.
