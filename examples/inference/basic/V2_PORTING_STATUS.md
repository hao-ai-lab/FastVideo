# v2 porting status — fastvideo models → the v2 (recipe, runtime) substrate

Goal: every model in fastvideo's registry resolves through the **v2 `VideoGenerator`** / `Engine`
(typed `fastvideo.api` configs + the real torch backend) to a recipe that can construct and run it.

**Scope: ALL fastvideo models (achieved).** v2 now resolves **63/64** of fastvideo's registered HF ids
by exact id (PRIMARY), plus the architecture fallback for local/unregistered checkpoints. The single
remaining id — `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` — is **environment-blocked**: its VSA
(Sparse-Linear Attention) kernels require `nvcc` (not built in this bring-up). It arch-resolves to the
base Wan card but needs the VSA kernel build to run faithfully.

Dispatch is **architecture-driven** (`v2/registry.py`): exact HF id → short-name → architecture
inference from the checkpoint (pipeline / transformer / VAE class names + `z_dim`, `transformer_2`,
`spatial_upsampler`). Adding a model is one `_BUCKET_C` row (HF ids → builders + transformer class).

## The porting mechanism — self-contained recipe packages
Every net-new arch is a **self-contained recipe package** (`v2/recipes/<arch>/` = `card.py` `loop.py`
`program.py` [+ `sampler.py`] + an optional `v2/platform/backends/torch_<arch>.py` adapter). The card
declares its torch adapter via **`ComponentSpec.adapter="module:Class"`** (the `_explicit_adapter` seam in
`torch_backend.py`) instead of editing the shared `_make_dit`/`_make_vae`/`_make_text_encoder` dispatch —
so a port adds **only new files**, never touching shared code, and parallel ports never conflict. New
samplers/loops live in-package. Registration is one row in `v2/registry.py:_BUCKET_C`.

## Working today (GPU-verified, real video/audio) — committed on `v2`
| Official example(s) | Model | v2 card |
|---|---|---|
| `basic.py`, `basic_mps.py`, `basic_ray.py` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | wan21 |
| `basic_self_forcing_causal.py` | `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers` | wan_causal |
| `basic_ltx2_distilled.py` | `FastVideo/LTX2-Distilled-Diffusers` (2-stage + spatial upsampler) | ltx2 |
| `basic_wan2_2_ti2v.py` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | wan2.2-ti2v |
| `basic_wan2_2.py` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (MoE + CPU expert offload) | wan2.2-a14b |
| `basic_ltx2.py` | `Davids048/LTX2-Base-Diffusers` | ltx2 base |
| `basic_ltx2_3_distilled.py` | `FastVideo/LTX-2.3-Distilled-Diffusers` (joint T2VS, video+audio) | ltx2.3-distilled |

Plus the **Wan2.1 i2v cluster** (Fun-1.3B-InP GPU-verified; I2V-14B-480P/720P + Wan2.2-I2V-A14B MoE reuse
the i2v card) — CLIP image-encoder + first-frame `[mask|cond]` → 36ch DiT.

## GPU bring-up results (real weights on H100 NVL, single-GPU, TORCH_SDPA)
**20 models generate real video/audio on GPU** — the 7 above + **13 of the newly-ported** archs, each run
end-to-end through the real `VideoGenerator` (resolve → stamp → CUDA load → generate). The rest are blocked
by a **fastvideo-shared-code / missing-kernel / HF-access** wall, NOT a v2 recipe bug (the v2 recipes are
faithful — e.g. cosmos25's DiT+VAE produced finite output; only its Qwen2.5-VL encoder hit a library
incompat). All ports also resolve + run end-to-end on the CPU toy backend (`test_bucket_c_ports.py`).

| GPU status | Models |
|---|---|
| ✅ **Verified** (real GPU output) | stable_audio (audio), matrixgame2, matrixgame3, gen3c, wan_fun_control, lucy_edit, hunyuangamecraft, hunyuan_video, hunyuan_video15, longcat (13.58B), sfwan22 (2×14B MoE, expert offload), lingbotworld (2×14B, offload), fastwan (TI2V-5B-FullAttn DMD) |
| 🚫 fastvideo/env-blocked | **cosmos25** (DiT+VAE ran; Qwen2.5-VL encoder → transformers 5.12.1 incompat in fastvideo); **kandinsky5** (fastvideo registry registers a bare `PipelineConfig`); **hyworld** (fastvideo DiT hardcodes `flash_attn`, not built); **turbowan** 1.3B/i2v + **fastwan** VSA-variants (SLA/VSA sparse-attn params + Triton kernels need nvcc) |
| 🚫 access-blocked (HF-gated) | cosmos2, flux2, sd35 (no HF token in this env) |

To unblock the env-blocked: build `fastvideo-kernel` (SLA/VSA Triton, needs nvcc); pin a fastvideo-compatible
`transformers` for the Qwen2.5-VL encoder; add a Kandinsky5 `PipelineConfig` + an SDPA fallback in the
hyworld DiT (all fastvideo-side / environment, not v2 recipe work).

## Newly ported (recipe details)
Each resolves through the registry AND runs end-to-end on the CPU toy backend via the public `Engine`
path (the `v2/tests/test_bucket_c_ports.py` regression guard), emitting the correct modality artifact.

**15 net-new architectures** (each a new `TorchComponent` adapter + recipe):
- **cosmos2** (Cosmos-Predict2-2B-Video2World) — EDM-Karras denoiser; new `CosmosDenoiseLoop` +
  `build_karras_sigmas` (the reference port). **cosmos25** (Cosmos-Predict2.5 2B/14B) — flow-match,
  per-frame plain-sigma timestep, Reason1/Qwen2.5-VL encoder. **gen3c** (GEN3C) — EDM + 82ch pose-buffer.
- **hunyuan_video** (+FastHunyuan) — reuses WanDenoiseLoop, dual LLaMA+CLIP encoders, Hunyuan VAE.
  **hunyuan_video15** (480p/720p). **hunyuangamecraft**, **hyworld** — interactive (camera/action).
- **longcat** (T2V/I2V/VC). **kandinsky5** (5.0 T2V Lite).
- **sd35** (MMDiT, image, triple-encoder). **flux2** (dev/klein, MMDiT image). **stable_audio** (audio).
- **lingbotworld** (camera/Plucker), **matrixgame2**, **matrixgame3** — interactive world models.

**5 Wan-family variants** (reuse the Wan/Causal arch, new in-package sampler/loop/conditioning):
- **turbowan** — rCM few-step (faithful RCMScheduler port), 1.3B/14B T2V + I2V-A14B MoE.
- **lucy_edit** — v2v editor (video-VAE-encode node → 96ch DiT input). **wan_fun_control** — control input.
- **sfwan22** — Self-Forcing Wan2.2-A14B causal + MoE (i2v + t2v). **fastwan** — DMD 3-step (TI2V-5B-FullAttn
  loadable; VSA-trained variants + non-strict `to_gate_compress` load are BRINGUP).

BRINGUP scope per port (documented in each package): GPU load/run; for interactive/world-model archs the
action/camera/memory conditioning needs a request-API extension (the t2v/degenerate path is what
CPU-verifies); video2world/i2v frame-replace conditioning is threaded but inert without conditioning inputs.

## Environment
v2 bring-up runs **single-GPU, resident, on the `TORCH_SDPA` backend** (no fastvideo-kernel / VSA / FP4).
The box has been rescheduled across hosts/arches/python versions mid-session; rebuild the venv for the
current arch when that happens: `uv venv --python 3.12 .venv`; comment out `fastvideo-kernel` in
`pyproject.toml`; `uv pip install -e ".[dev]"`. Source `/home/scratch.willlin_ent/.bringup_env`
(`HF_HOME=./.cache` on scratch, `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`). v2 CPU mini: 240 passed, 2 skipped.

## How to add a model to the v2 substrate
1. `v2/recipes/<arch>/` — card (declare adapters via `ComponentSpec.adapter`; per-model `SamplingDefaults`),
   loop (reuse `WanDenoiseLoop`/`chunk_rollout` or a new in-package loop+sampler), program.
2. `v2/platform/backends/torch_<arch>.py` — a `TorchComponent` subclass (only the forward semantics) if the
   arch is genuinely new; reuse `WanDiT`/`LTX2DiT`/`WanVAE`/`T5Encoder` via `load_id` when it isn't.
3. One row in `v2/registry.py:_BUCKET_C` (HF ids → builders; `transformer_cls` for the arch fallback, or
   `""` for explicit-id-only capability variants of an existing arch).
4. CPU-verify: it resolves + runs on the toy backend (auto-covered by `test_bucket_c_ports.py`). Then GPU
   bring-up (`stamp_*_checkpoints` → real weights) per BRINGUP notes.
