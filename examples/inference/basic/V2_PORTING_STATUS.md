# v2 porting status — fastvideo models → the v2 (recipe, runtime) substrate

Goal: drive each `examples/inference/basic/` model through the **v2 `VideoGenerator`** (typed
`fastvideo.api` configs + the real torch backend) and generate video, with a matching `v2_*` example.

The v2 mini implements a small set of loops/samplers (**flow-match Euler**, causal **chunk_rollout**,
**FlowGRPO SDE**) and adapters (Wan DiT/VAE/UMT5, causal Wan, LTX-2 DiT/VAE/Gemma). A model ports
**cleanly** only when its architecture *and* sampler match what v2 implements; otherwise it needs new
per-model work (a card, an adapter for the DiT/VAE/encoder forward, and/or a new sampler), and some are
blocked by the environment (gated weights, local-only weights, non-video modality, interactive input).

The v2 bring-up runs **single-GPU, resident, on the `TORCH_SDPA` backend** (no fastvideo-kernel / VSA /
FP4), so distilled models that depend on VSA/SLA/FP4 kernels or non-flow-match samplers won't match.

## Working today (verified, real video) — committed on `v2`
| Official example(s) | Model | v2 card | v2 example |
|---|---|---|---|
| `basic.py`, `basic_mps.py`*, `basic_ray.py`* | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | wan21 | `v2_basic.py`, `v2_basic_new_api.py` |
| `basic_self_forcing_causal.py` | `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers` | wan_causal | `v2_basic_self_forcing_causal.py`, `v2_basic_new_api.py` |
| `basic_ltx2_distilled.py`, `basic_ltx2_distilled_fast_profile.py` | `FastVideo/LTX2-Distilled-Diffusers` | ltx2 | `v2_basic_new_api.py` |

\* `basic_mps.py` (Apple MPS) and `basic_ray.py` (ray executor) are device/executor variants of the
same Wan2.1 model — the model itself is covered by wan21; those runtime modes are out of v2 scope.

## Needs per-model work (architecture/sampler matches partially)
| Official example(s) | Model | What v2 needs |
|---|---|---|
| `basic_dmd.py`, `basic_dmd_new_api.py` | FastWan2.1-T2V-1.3B (WanDMDPipeline) | **Blocked via generic reuse:** DMD checkpoint's `to_gate_compress` param mapping differs from the generic `WanTransformer3DModel` load; needs WanDMD param-mapping/config handling (+ ideally the DMD/UniPC few-step schedule). |
| `basic_turbodiffusion*.py` | TurboWan2.1/2.2 (RCM) | **New sampler:** Reparameterized Consistency Model — v2 has no consistency loop (flow-match Euler won't match). |
| `basic_wan2_2.py`, `basic_wan2_2_ti2v.py`, `basic_wan2_2_i2v.py`, `basic_wan2_2_Fun.py` | Wan2.2 (A14B MoE / TI2V-5B / I2V / Fun) | **New card:** MoE boundary-timestep expert routing (2 transformers; v2 has `ExpertRouting`), TI2V-5B's distinct VAE, I2V image conditioning, Fun control. Large (~56GB A14B). |
| `basic_ltx2.py` | `Davids048/LTX2-Base-Diffusers` | Base (non-distilled): base sigma schedule + more steps (the v2 ltx2 card hardcodes the distilled 8+3). |
| `basic_ltx2_3_distilled_i2v*.py` | LTX-2.3-Distilled I2V | Image conditioning on the ltx2 program (encode + condition). |
| `basic_lucy_edit.py` | `decart-ai/Lucy-Edit-Dev` | Wan-family video-edit: input-video conditioning. |

## New families — each needs a new card + adapters (DiT/VAE/encoder forwards) + maybe a sampler
`basic_cosmos2_5_*` (Cosmos-Predict2.5), `basic_gen3c.py` (GEN3C — **local `converted_weights/` only**),
`basic_hy15*.py` / `basic_gamecraft.py` / `basic_hyworld.py` (Hunyuan family),
`basic_longcat_*.py` (LongCat + LoRA), `basic_lingbotworld_base_cam.py`,
`basic_sd35_t2i.py` (SD3.5 image), `basic_flux2*.py` (**gated** black-forest-labs).

## Different modality / interactive (out of the t2v VideoGenerator surface)
`basic_stable_audio*.py` (audio), `basic_matrixgame2*.py` / `basic_matrixgame3.py`
(interactive mouse/keyboard-conditioned world models).

## How to add a model to the v2 VideoGenerator
1. Ensure a v2 card exists (`v2/models/<family>/`) whose `load_id`s resolve to real `fastvideo.models.*`.
2. Register the HF id → family in `v2/video_generator.py:_FAMILY_BY_PATH` (or rely on the name fallback).
3. If the DiT/VAE/encoder forward differs, add an adapter + class-detection in `torch_adapters.py`.
4. Add `examples/inference/basic/v2_<name>.py` mirroring the official example; generate to confirm.
