# v2 porting status — fastvideo models → the v2 (recipe, runtime) substrate

Goal: drive each `examples/inference/basic/` model through the **v2 `VideoGenerator`** (typed
`fastvideo.api` configs + the real torch backend) and generate video, with a matching `v2_*` example.

**Active scope (per request): the Wan family + LTX-2 only.** The Cosmos/Hunyuan/SD3.5/LongCat/Flux2/
audio/interactive families below are kept as an upstream reference map but are out of the current scope.

Dispatch is **architecture-driven** (`v2/video_generator.py`): `from_config` reads the checkpoint's
pipeline / transformer / VAE class names (+ `z_dim`, `transformer_2`) and picks the v2 card — mirroring
fastvideo's `get_pipeline_config_cls_from_name`, with no hardcoded repo-id table.

The v2 mini implements a small set of loops/samplers (**flow-match Euler**, causal **chunk_rollout**,
**FlowGRPO SDE**) and adapters (Wan DiT/VAE/UMT5, causal Wan, LTX-2 DiT/VAE/Gemma/**spatial upsampler**).
A model ports **cleanly** only when its architecture *and* sampler match what v2 implements; otherwise it
needs new per-model work (a card, an adapter for the DiT/VAE/encoder forward, and/or a new sampler), and
some are blocked by the environment (gated weights, local-only weights, non-video modality, no VSA).

The v2 bring-up runs **single-GPU, resident, on the `TORCH_SDPA` backend** (no fastvideo-kernel / VSA /
FP4), so distilled models that depend on VSA/SLA/FP4 kernels or non-flow-match samplers won't match.
(Wan2.2-A14B MoE is the exception to "resident": its two 14B experts are CPU-offloaded and swapped onto
the GPU one at a time at the boundary-timestep transition, so it fits a single 80GB GPU.)

**Environment status (current):** the box was rescheduled from an aarch64 host to an x86_64 host
mid-session (`ipp1a1…` → `ipp2-0493`; `uname -m` = x86_64). The `.venv` is aarch64 wheels
(numpy/torch/fastvideo/CUDA) which cannot execute on x86 (numpy `_core` is aarch64; `uv` hits a
missing-aarch64-loader binfmt error). GPU verification is blocked until the box returns to aarch64 (the
existing venv then works as-is) or the venv is rebuilt for x86. Every model marked *verified* above was
GPU-confirmed **before** the flip; the ‡ models are code-complete, pending GPU re-verify.

## Working today (verified, real video) — committed on `v2`
| Official example(s) | Model | v2 card | v2 example |
|---|---|---|---|
| `basic.py`, `basic_mps.py`*, `basic_ray.py`* | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | wan21 | `v2_basic.py`, `v2_basic_new_api.py` |
| `basic_self_forcing_causal.py` | `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers` | wan_causal | `v2_basic_self_forcing_causal.py`, `v2_basic_new_api.py` |
| `basic_ltx2_distilled.py`, `basic_ltx2_distilled_fast_profile.py` | `FastVideo/LTX2-Distilled-Diffusers` | ltx2 — real `LTX2LatentUpsampler` between base/refine (un_normalize→learned 2× upsample→normalize) | `v2_basic_new_api.py` |
| `basic_wan2_2_ti2v.py` (T2V branch) | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | wan2.2-ti2v | `v2_basic_wan2_2_ti2v.py` |
| `basic_wan2_2.py` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (MoE) | wan2.2-a14b + **CPU expert offload** (one 14B expert resident at a time, swapped at the boundary → 60GB peak) | `v2_basic_wan2_2.py` |
| `basic_ltx2.py` ‡ | `Davids048/LTX2-Base-Diffusers` | ltx2 base (single-stage, request-driven steps) | `v2_basic_ltx2.py` |
| `basic_ltx2_3_distilled.py` ‡ | `FastVideo/LTX-2.3-Distilled-Diffusers` | ltx2 base card (single-stage; pass few steps) | `v2_basic_ltx2_3_distilled.py` |

\* `basic_mps.py` (Apple MPS) and `basic_ray.py` (ray executor) are device/executor variants of the
same Wan2.1 model — the model itself is covered by wan21; those runtime modes are out of v2 scope.
‡ Code-complete + CPU-verified (cards/programs build, dispatch routes, schedule correct); GPU re-verify
pending the env (see **Environment status** below).

## Needs per-model work (architecture/sampler matches partially)
| Official example(s) | Model | What v2 needs |
|---|---|---|
| `basic_dmd.py`, `basic_dmd_new_api.py` | FastWan2.1-T2V-1.3B (WanDMDPipeline) | **Blocked (env):** loading needs a *non-strict* load to skip the VSA-only `to_gate_compress` (root cause: the checkpoint key mis-maps under the generic Wan path, and `TransformerLoader` is strict except Cosmos2.5 — `component_loader.py:1005`). Even loaded, the DMD model is trained for VSA, which is **not built here (no nvcc)**, so SDPA output would be degraded. Needs the fastvideo-kernel VSA build + a non-strict WanDMD load. |
| `basic_turbodiffusion*.py` | TurboWan2.1/2.2 (RCM) | **New sampler:** Reparameterized Consistency Model — v2 has no consistency loop (flow-match Euler won't match). |
| `basic_wan2_2_i2v.py`, `basic_wan2_2_Fun.py` | Wan2.2-I2V / Fun | I2V needs an image-encode node + conditioning (the Wan adapter already accepts `encoder_hidden_states_image`; reuse the A14B offload); Fun needs a control input. |
| `basic_ltx2_3_distilled_i2v*.py` | LTX-2.3-Distilled I2V | Image conditioning on the ltx2 single-stage program (encode the input image + condition the denoise). |
| `basic_lucy_edit.py` | `decart-ai/Lucy-Edit-Dev` | Wan-family video-edit: input-video conditioning. |

## Out of current scope — new families (kept as an upstream reference map only)
Each would need a new card + adapters (DiT/VAE/encoder forwards) + maybe a sampler:
`basic_cosmos2_5_*` (Cosmos-Predict2.5), `basic_gen3c.py` (GEN3C — **local `converted_weights/` only**),
`basic_hy15*.py` / `basic_gamecraft.py` / `basic_hyworld.py` (Hunyuan family),
`basic_longcat_*.py` (LongCat + LoRA), `basic_lingbotworld_base_cam.py`,
`basic_sd35_t2i.py` (SD3.5 image), `basic_flux2*.py` (**gated** black-forest-labs).

## Different modality / interactive (out of the t2v VideoGenerator surface)
`basic_stable_audio*.py` (audio), `basic_matrixgame2*.py` / `basic_matrixgame3.py`
(interactive mouse/keyboard-conditioned world models).

## How to add a model to the v2 VideoGenerator
1. Ensure a v2 card exists (`v2/models/<family>/`) whose `load_id`s resolve to real `fastvideo.models.*`.
2. Dispatch is automatic: add a branch in `v2/video_generator.py:_select_builders` keyed on the new
   architecture (transformer/pipeline/VAE class names) — no HF-id table to edit.
3. If a DiT/VAE/encoder/upsampler forward differs, add an adapter + class-detection in `torch_adapters.py`
   and register the component kind in `torch_cuda.py`.
4. Add `examples/inference/basic/v2_<name>.py` mirroring the official example; generate to confirm.
