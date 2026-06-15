# FastVideo Next-Generation Inference Runtime — Design Proposal

**Status:** Draft for discussion (v19 — v18 reconciled against the landed RL reference: PR #1450 Wan DiffusionNFT
training and PR #1396 per-method grad-norm regression CI; §8 claims converted from prediction to landed evidence)
**Date:** 2026-06-12
**Scope:** The inference runtime and the shared substrate beneath it. The legacy `fastvideo/training/` stack is
fully out of scope (frozen, per repo policy). The modular `fastvideo/train/` stack is **in scope as a consumer, not
as a redesign target**: its trainer machinery (Trainer, method loss logic, callbacks, FSDP/optimizer/checkpoint/
data) is untouched, but its embedded sampling/rollout paths migrate onto the shared loop substrate, its config
imports the shared layers, and the RL layer is built under it against the engine's rollout surface — §8 defines the
exact boundary.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Goals and non-goals](#2-goals-and-non-goals)
3. [Where we are today](#3-where-we-are-today)
4. [What we must support: the model taxonomy](#4-what-we-must-support-the-model-taxonomy)
5. [Survey of reference systems](#5-survey-of-reference-systems)
6. [Proposed architecture](#6-proposed-architecture)
7. [Worked examples: seven workloads on the new runtime](#7-worked-examples)
8. [Training, post-training, and RL on the same substrate](#8-training-post-training-and-rl-on-the-same-substrate)
9. [The application pull: Dreamverse, ComfyUI, and the workflow layer](#9-the-application-pull-dreamverse-comfyui-and-the-workflow-layer)
10. [Migration plan](#10-migration-plan)
11. [Trade-offs and alternatives considered](#11-trade-offs-and-alternatives-considered)
12. [Risks, concerns, and open questions](#12-risks-concerns-and-open-questions)

---

## 1. Executive summary

FastVideo today is a **pipeline library that runs one request at a time**: a linear list of stages, a 111-field
`ForwardBatch` blackboard (~153 counting the training/preprocess batches in the same file), a 1,381-line
`denoising.py` whose ~670-line `DenoisingStage` owns the sampling loop, and an SPMD executor with no request
queue. This served bidirectional video diffusion well, but the model landscape has moved:

- **Autoregressive/causal video** (Wan-Causal, MatrixGame2/3, LongCat, Self-Forcing lineage) each re-implement
  ad-hoc KV-cache dicts inside their DiT files.
- **Audio** arrived twice — as a joint output modality (LTX-2 video+audio) and as a standalone diffusion model
  (Stable Audio) — and both had to be threaded through video-shaped abstractions (`batch.extra["audio"]`).
- **Image** models (Flux2, SD3.5) inherit a video denoising loop with `prefix == "Flux"` special-casing.
- **Omni models** are the forcing function. The Cosmos3 port — the **full-omni** branch
  [`feat/cosmos3-reasoning`](https://github.com/hao-ai-lab/FastVideo/tree/feat/cosmos3-reasoning), top of the
  stacked chain covering T2V/I2V/T2I, joint video+audio (t2vs), **action**, and text reasoning, all bit-exact
  against the official framework (worktree `/home/william5lin/FastVideo_cosmos3_port`) — proved FastVideo *can*
  run a mixture-of-transformers world model — but only by writing one monolithic stage that owns
  tokenize→encode→denoise→decode, returning audio via `batch.extra`, and running text reasoning with **no KV
  cache** (full re-prefill per token).

The proposal is to evolve FastVideo into an **engine with three cleanly separated planes**:

1. **Request plane** — typed multimodal requests/outputs (`OmniRequest`/`OmniOutput`), an async engine with a real
   queue, streaming, and an OpenAI-compatible server.
2. **Pipeline plane** — pipelines become **declarative graphs of stages**; the two iterative computations
   (**denoise loops** and **AR decode loops**) are *inverted* so the runtime owns iteration and stages expose
   `init/step/finalize`. Model-specific behavior moves from `isinstance` forests into **policy objects** resolved at
   pipeline-build time.
3. **Execution plane** — the existing SPMD worker pool, plus a **step-level scheduler** that multiplexes loop steps
   across concurrent requests — with first-class **cancellation, failure isolation, and memory admission**
   designed alongside it, because multiplexing creates the blast radius it must then contain (§6.3.1) — a
   **cache manager** with per-class budgeted pools (paged text KV, slab chunk-KV, feature caches), and an optional
   **placement layer** (role-disaggregated pools, per-stage devices, connectors) borrowed from sglang
   `multimodal_gen` and vllm-omni. Pools are single-node in v1; fleet scale is Dynamo's plane.

Two systems cut across all three planes. An **extension system** (§6.4) provides versioned hook points: read-only
*observers* for debugging, profiling, and numerical-parity alignment (generalizing the existing
`fastvideo/hooks/activation_trace.py`), and compute-altering *interceptors* for cache accelerations —
[cache-dit](https://github.com/vipshop/cache-dit) (DBCache/TaylorSeer, the library sglang's multimodal serving
already integrates) as the flagship — with per-request, per-CFG-branch state so they stay correct under
concurrency. And **declarative stacked parallelism** (§6.3.4, adapted from vLLM-Omni RFC #4084) makes pool meshes
and engine topology one validated spec instead of scattered flags. Above the engine, fleet scale is explicitly
delegated: **NVIDIA Dynamo is the first-class partner** for discovery, cross-node routing, SLA autoscaling, and
cold starts — FastVideo already runs as a minimal Dynamo video backend, and the design hardens that contract
(§6.3.5) rather than rebuilding any of it. The partnership is two-way and Dynamo is not treated as frozen: §6.3.6
states seven concrete upstream asks (generic affinity key spaces, a request-cost interface, chunked media
streaming, role-graph disagg, an RL weight plane with staleness-aware routing, cache-object tiering, sessions),
each with a fallback. The asks are unfiled proposals on an NVIDIA-governed roadmap — fallbacks are first-class
plan elements, not afterthoughts, and the goal is Dynamo fronting both production serving and RL rollout fleets.

The same substrate answers the training question (§8). Integrated training/post-training is FastVideo's advantage
— structurally, not incidentally: unlike LLMs, where inference optimization is post-hoc on frozen weights, a
*usable* video model is itself a post-training artifact (step distillation is mandatory; low precision needs QAT —
our NVFP4 line; AR/world models need distillation plus self-forcing), so every inference capability is a
**(recipe, runtime) pair** and the training loop embeds the inference loop (§8.3). But today the denoise/sampling
loop exists in **four copies** (inference stages, `train/` distillation methods,
legacy `training/` pipelines — and the RL work, landed as PR #1450, vendored the **fourth** and documented why in
its own docstring: there was no shared loop it could call without binding to pipeline classes, §8.1). Loop
inversion makes the step functions the single shared implementation **for inference, `train/`, and RL** where they
fit — self-forcing ships as a custom training loop consuming substrate pieces (§8.2), and the
legacy `training/` copy stays frozen and retires per-family on the Phase-5 schedule, not by in-place unification;
the engine doubles as the RL rollout engine
(trajectory/log-prob capture, weight sync, sleep/wake) under a strict `engine never imports train` rule. Two
production RL systems sharpen the rule (§8.4): the industry's pain is **two runtimes with different kernels**
(Megatron/FSDP training vs vLLM/sglang rollout) — verl-omni re-implements Wan2.2 inside vLLM-Omni and corrects the
numerics afterward; miles' headline features (TIS/MIS, bitwise log-probs, R3 routing replay, unified FP8) are all
mismatch patches. We answer with one model definition, one kernel set, and an explicit **consistency ladder**
(C0 corrected / C1 kernel-pinned / C2 bitwise, §8.5) — measured by ParityAligner, never assumed. The fragmented
config story — `FastVideoArgs`/`TrainingArgs`/`TrainingConfig`/`SamplingParam` plus the in-flight `fastvideo/api/`
schema and its 651-line `compat.py` — collapses into four owned layers with generated views (§6.6). And the design
has a shipping product pulling on it (§9): Dreamverse, the vibe-directing app in `apps/dreamverse/`, today
hand-rolls its own GPU pool, queue, warmup, and streaming relay at a cost of one B200 per user session — the
engine's first production customer, and (via preference data from directing sessions) the eventual data engine for
the RL layer. The same section sets the ComfyUI strategy (§9.4): embed (custom nodes, shipping), **compile** (a
clean-room workflow→PipelineSpec compiler + weight-fleet cache powering an accelerated alternative to running
workflows on stock runtimes), and productize (Dreamverse Studio) — sequenced, with the custom-node long tail
explicitly out of scope.

The single most important architectural claim: **multi-engine DAG composition (vllm-omni / sglang-omni style) is
necessary but not sufficient.** It handles separable omni models (thinker → talker → vocoder, distinct weights per
stage). vllm-omni's `bagel_single_stage`/`lance` prove one resident instance *can* run both loops on shared
weights in one request — AR `generate_text` and diffusion `generate_image` over co-resident `*_moe_gen` experts
(`bagel_transformer.py:718,726`; `pipeline_bagel.py:677,808`). What a DAG of **separate** engines cannot do is
share the weights; what vllm-omni does **not** do is make those interleaved loops runtime-visible — it buries them
in one opaque request-scheduled `DIFFUSION` stage (`RequestScheduler`, `max_num_running_reqs=1`,
`diffusion_engine.py:166-169`), so the scheduler never sees the AR+denoise interleaving. FastVideo's
differentiation is therefore not *expressing* MoT (vllm-omni already does) but making the loops first-class:
a runtime where *one model instance* serves interleaved AR steps and denoise steps that are runtime-visible,
step-scheduled, and batchable — and that requirement, not serving features, is what dictates the stage/loop
abstraction below. Stated with open eyes: **no surveyed system makes runtime-owned diffusion iteration the
always-on default at scheduler granularity — there is only narrow, opt-in precedent: vllm-omni's
`SupportsStepExecution` (`prepare_encode`/`denoise_step`/`step_scheduler`/`post_decode`,
`diffusion/models/interface.py:44-67`) is runtime-owned diffusion iteration at step granularity, but
Qwen-Image-only and off in every deploy; this design generalizes it to an always-on universal contract**
(diffusers' loop blocks, by contrast, own their own iteration; §6.2.3). The risk is
retired by Phase-1 bit-identical parity gates and a measured falsifier (§10), not by borrowed validation. The
contract is also kept deliberately small: families may ship custom step bodies — the runtime owns *iteration*,
never the factoring of step internals.

Migration is incremental: seven phases (−1 to 5), each independently shippable, with SSIM/parity regression gates
and a compatibility adapter so the existing 20+ model families never break. The deprecation commitment is backed
by **enforcement the prior freeze lacked** — the `training/`-vs-`train/` "lesson" is not invoked as a moral but as
data: that freeze was broken 19 times including a new legacy pipeline nine days after it was forbidden — so this
plan carries CI path gates, CODEOWNERS, named owners, a calendar target for Phase 5, and an explicit inflow rule
for the ~1–2 new model families arriving every month (§10).

---

## 2. Goals and non-goals

### Goals

- **G1 — One runtime for all generation paradigms**: bidirectional diffusion (video/image/audio), causal/AR video
  with KV cache, few-step distilled, multi-stage refinement, control/world models, AR token decoding (text, audio
  codecs), and hybrid AR+diffusion (Cosmos3 MoT, BAGEL/Transfusion-style).
- **G2 — Omni-native I/O**: requests and outputs are typed sets of modality parts (text, image, video, audio,
  action, latents). Video+audio+text outputs are first-class, not `extra`-dict passengers.
- **G3 — A real serving runtime**: async engine, request queue, concurrency, cross-request batching where shapes
  permit, streaming (tokens, audio chunks, frame chunks), warmup, metrics — at parity with sglang `multimodal_gen`,
  which is FastVideo's own descendant.
- **G4 — Cosmos3-class MoT support as a first-class citizen**: packed factored sequences, dual-pathway attention,
  per-request hybrid KV+denoise state, reasoner KV cache, chunked world-model rollout.
- **G5 — Zero regression for existing models**: every shipped pipeline keeps working (SSIM-gated — coverage is
  completed for *all* families in Phase −1, since ~7 currently have no SSIM test and would otherwise pass
  vacuously; substrate-wide PRs in Phases 0–2 run the **full** GPU matrix on Modal L40S as a budgeted cost,
  because for exactly those PRs "touched families" means all of them) throughout the migration; the offline
  `VideoGenerator` API survives.
- **G6 — Keep the hot loop fast**: abstractions must resolve at pipeline-build time; the per-step path stays
  compile/cudagraph-friendly; batch-of-1 video on N GPUs (SP/TP today, CFG-parallel once added) must not get
  slower.
- **G7 — One substrate for inference, training, and RL**: models, loaders, configs, schedulers, parallel state, and
  the generative loop bodies are single-source; `train/` and RL consume them in-process or via the engine as a
  rollout service; the engine never depends on training code (§8).

### Non-goals

- **N1 — Datacenter orchestration.** Multi-node routing, SLA planners/autoscaling, fleet health, and cold-start
  orchestration are not built here — **NVIDIA Dynamo is the named first-class partner for that layer** (`dynamo/`
  checkout in repo root; FastVideo already runs as a minimal Dynamo video backend). The engine's job is to be an
  excellent Dynamo citizen: registration, health/drain, cost metrics, affinity events, NIXL-compatible disagg —
  the full contract and separation-of-concerns table is §6.3.5. We never schedule across nodes ourselves.
- **N2 — Trainer internals.** Legacy `fastvideo/training/` is frozen and untouched. In `fastvideo/train/`,
  everything that makes it a *trainer* — `Trainer`, method loss logic, callbacks, FSDP wrapping, optimizers,
  checkpointing, data loading — is not redesigned. §8 changes only what `train/` *consumes*: its embedded
  sampling/rollout loops (today private copies) move onto the shared substrate, its `distributed:`/model config
  blocks import the shared layers (§8.8), and the RL layer lands under it. Dependency rule: `train/` imports the
  substrate/engine; never the reverse.
- **N3 — Replacing the model porting methodology.** Bit-exact per-component parity vs official frameworks (the
  Cosmos3 rules) remains the porting bar; this design changes where ported components plug in, not how they are
  validated.
- **N4 — Migrating all 20+ model families at once.** The plan defines a per-family migration checklist; legacy
  pipelines run unmodified behind an adapter until migrated.
- **N5 — Standalone LLM-serving excellence.** We host AR computations because **omni and interactive models
  require them** — Cosmos3's reasoner, thinkers/talkers, audio-codec decode, causal-video chunk loops — not to
  compete with vLLM/sglang at LLM serving. The priority order is: **omni and diffusion done well — bidirectional,
  causal, and interactive — first**; AR machinery enters only at the sophistication those workloads pull. Concretely:
  paged KV and continuous batching for AR *pathways*, yes; radix prefix trees, speculative decoding, and
  chunked-prefill sophistication, only when an omni workload demonstrates the need (they are flagged as open
  questions, not commitments); a big *separable* thinker delegates to an external engine (§6.3.4) rather than
  pulling its serving problems into ours. This is also the standing guard against over-indexing on LLM-engine
  borrowings (§5): every vLLM/sglang adoption in this document is admitted under a diffusion/omni justification,
  not on its LLM merits.

---

## 3. Where we are today

### 3.1 Current architecture

```
VideoGenerator / StreamingVideoGenerator / CLI / OpenAI entrypoints   (fastvideo/entrypoints/)
        │  one ForwardBatch per call (synchronous)
        ▼
Executor ABC ── MultiprocExecutor (1 proc/GPU, SPMD) ── RayDistributedExecutor   (fastvideo/worker/)
        │  collective_rpc / execute_forward
        ▼
ComposedPipelineBase  (fastvideo/pipelines/composed_pipeline_base.py)
   loads modules from model_index.json via ComponentLoaders
   runs a LINEAR list of stages:
        for stage in self._stages: batch = stage(batch, fastvideo_args)
        ▼
PipelineStage ABC  (fastvideo/pipelines/stages/base.py:29)
   __call__ → verify_input → forward(batch: ForwardBatch, args) → verify_output
        ▼
ForwardBatch  (fastvideo/pipelines/pipeline_batch_info.py, 335 lines)
   111 fields: prompts, embeds, latents, timesteps, camera_states, mouse_cond,
   ltx2_*, trajectory…, plus `extra: dict[str, Any]` (line 238) as escape hatch
```

Parallelism today: TP and SP (Ulysses all-to-all only — ring-attention SP is *not* wired on main) — CFG-parallel
exists in the sglang `multimodal_gen` fork and in
cosmos-framework (CFGP) but **not on main**; this design adds it as an axis (§6.3.4). Plus per-component
CPU/layerwise offload, per-component `torch.compile`, and a strong attention-backend registry (FA, SDPA, Sage,
VSA, V-MoBA, BSA, SLA) selected via `fastvideo/attention/selector.py`.

### 3.2 What strains (evidence-backed)

| # | Pain point | Evidence |
|---|-----------|----------|
| P1 | **Monolithic denoising loop.** A 1,381-line `denoising.py` (the base `DenoisingStage` is ~670 lines; 35 `isinstance/hasattr/getattr` model probes file-wide); **22** model/variant denoising-stage classes across files, 23 with the base (Causal, CausalDMD, LongCat ×3, LTX2, GameCraft, HYWorld, MatrixGame2/3, Gen3C, Cosmos ×5, DMD, SR, SD35, Magi-class, +2 more). Flux gating via `getattr(config, "prefix", "") == "Flux"`; the Wan2.2 expert switch and TI2V inline VAE encode are hardcoded in the shared loop; Cosmos2.5's per-step clamping lives in its in-file subclass. | `fastvideo/pipelines/stages/denoising.py:129,237,352-376,441-475` |
| P2 | **The loop is owned by the stage**, so nothing outside it can schedule, preempt, interleave, stream, or batch steps. Causal/AR models bolt token/chunk loops inside stage subclasses; Cosmos3 had to bypass stages entirely with a free-function `denoise()` engine. | `causal_denoising.py:44,427`; worktree `cosmos3_stages.py` |
| P3 | **Blackboard state.** `ForwardBatch` mixes user intent, derived state, and per-model fields (`mouse_cond`, `camera_states`, `ltx2_*`, `pose`…); multi-modal outputs leave via `batch.extra["audio"]`. No stage declares what it reads/writes. | `pipeline_batch_info.py:238` |
| P4 | **No runtime.** One request per call; `assert latent_model_input.shape[0] == 1, "only support batch size 1"`; no queue, no async engine, no concurrency, no cross-request batching, no step interleaving. Streaming exists only as a bespoke `StreamingTask` path for game models. | `denoising.py:237`; `worker/multiproc_executor.py` |
| P5 | **AR support is per-model folklore.** `causal_wanvideo.py` threads a raw `kv_cache: dict` through forward (`:80,100,143`); LongCat has its own `LongCatKVCacheInitStage`; MatrixGame2 its own; Cosmos3 text reasoning re-prefills the full sequence per token (no cache at all). No allocator, no reuse, no eviction, no paging. | `models/dits/causal_wanvideo.py:80-143` |
| P6 | **No graph, no placement.** Multi-stage models (LTX-2 stage-1→upsample→stage-2 refine→audio decode; Hunyuan15+SR) are hand-wired linear lists; all stages run on all ranks; encoders cannot live on different GPUs; no fan-out for multi-modality outputs. | `pipelines/basic/ltx2/ltx2_pipeline.py` |
| P7 | **Task modes are inferred by heuristics.** Cosmos3 infers T2I from `num_frames==1 and image is None`; TI2V/V2V/LucyEdit ride boolean flags through the shared loop. | worktree `cosmos3_stages.py:101` |
| P8 | **Component set is static.** Loaders are keyed by fixed module names; Cosmos3's sound VAE had to be lazy-loaded inside `forward()` behind an env var. | worktree `cosmos3_stages.py:262-277` |

The Cosmos3 port is the best stress test we have: it succeeded (150/150 parity tests, every modality bit-exact),
but its friction list (monolithic stage, `extra["audio"]`, order-sensitive flat-latent slicing, no reasoning KV
cache, lazy sound VAE, mode heuristics) reads as a requirements document for this design.

---

## 4. What we must support: the model taxonomy

Every paradigm below is either shipped, in-flight, or explicitly requested (Cosmos 3, omni):

| Paradigm | Examples in/around tree | Loop shape | State needed | Output |
|---|---|---|---|---|
| Bidirectional video diffusion | Wan 2.1/2.2, Hunyuan(15), LongCat, Cosmos 2/2.5, LTX-2, TurboDiffusion | N denoise steps over full clip | latents, CFG branches, step/block caches | video |
| Few-step distilled | DMD/FastWan, TurboDiffusion, rCM | 1–4 denoise steps | latents | video |
| **Causal/AR video** | Wan-Causal(-DMD), MatrixGame2-Causal, MatrixGame3, LTX-2 causal, LongCat-VC | outer loop over chunks × inner denoise per chunk | **DiT KV cache** (per-chunk K/V, sink/local window), rolling latents | video (streamable by chunk) |
| Interactive world models | MatrixGame, GameCraft, HYWorld, Gen3C, LingBotWorld, Waypoint | chunk loop driven by live user actions | KV cache + action/camera conditioning per chunk | video stream |
| Image | Flux2, SD3.5, Qwen-Image-class | N denoise steps, small | latents | image (batchable!) |
| Audio diffusion | Stable Audio | N denoise steps over audio latents | audio latents | audio |
| Joint A/V diffusion | LTX-2 (video+audio in one denoise), Cosmos3 t2vs | one joint denoise over concatenated modality latents | multi-modality latents, per-modality CFG scales | video + audio |
| Multi-stage refinement | LTX-2 (base→upsample→refine), Hunyuan15-SR | pipeline of loops | inter-stage latents | video |
| **AR token decode** | Cosmos3 reasoner (Qwen-lineage und pathway); omni thinkers/talkers (Qwen2.5/3-Omni-style) | token loop until EOS | **paged text KV cache**, sampling state | text / codec tokens |
| Vocoder / one-shot heads | LTX-2 vocoder, audio codec → wav | single forward | none | audio |
| **Hybrid MoT (omni)** | **Cosmos3** (und/gen dual-pathway, shared attention, joint vision+action+sound denoise, AR reasoner on same weights), BAGEL/Transfusion lineage | AR loop **then/while** denoise loop, same resident weights | text KV + multimodal denoise state + packed sequence layout, in one request | text + video + audio + action |

### 4.1 The Cosmos3 forcing function (what "support omni" concretely means)

From the official `cosmos-framework` (`cosmos_framework/model/vfm/`):

- **MoT layer** = per-modality expert weights — separate QKV/O projections, norms, and MLPs for the
  *understanding* (text, causal) and *generation* (vision/action/sound, full or NATTEN-sparse) pathways — with
  **shared attention over one packed sequence** (`unified_mot.py`: `PackedAttentionMoT`, `MoTDecoderLayer`; gen MLP
  optionally MoE).
- **Packed factored sequence**: per request, `[text (causal) | vision patches | action | sound]` with per-segment
  attention modes, sample offsets, and 3D-MRoPE position ids (text 1D; vision T/H/W with FPS modulation; sound at
  fixed 25 latent fps; `temporal_modality_margin` at boundaries).
- **One `velocity_fn` denoises all generated modalities jointly**; CFG = two packed forwards (cond/uncond) with
  interval gating; sampler pluggable (UniPC/EDM/fixed-step).
- **AR reasoner runs on the und pathway of the same weights** with a per-layer `ReasonerKVCache` (prefill +
  1-token decode), used standalone (text out) and as **prompt upsampling before diffusion in the same request**.
- **World-model rollout**: `video_temporal_causal` masks + chunked continuation (generate chunk, re-encode,
  condition, continue).
- Inference batching in the framework is **greedy token-budget packing** of heterogeneous samples
  (`cosmos_framework/inference/inference.py:66`, `_iter_packed_batches`) — diffusion requests batched by total
  token count, not by fixed shapes.

A runtime that can serve this can serve everything else in the table. A runtime that composes separate engines per
stage **cannot** serve this (the reasoner and denoiser share weights; splitting them doubles 30B+ of weights and
breaks the interleaved KV/denoise state).

---

## 5. Survey of reference systems

What we take and what we reject, per system. (Full notes from the code review are in the per-repo paths cited.)

| System | What it is | Take | Reject / not enough |
|---|---|---|---|
| **sglang `multimodal_gen`** (`~/sglang/python/sglang/multimodal_gen/`) — FastVideo's serving-hardened descendant | HTTP/OpenAI server (images/videos/meshes), rank-0 `Scheduler` event loop + `GPUWorker`s over ZMQ, `Req` (= ForwardBatch + SamplingParams delegation), **N:M:K role disaggregation** (`disaggregation/orchestrator.py`: encoder/denoiser/decoder pools, capacity-aware dispatch, request state machine, tensor-transfer with P2P/RDMA), warmup requests, RL weight updates | Server/API shape; scheduler-process topology; **role affinity on stages** + disagg state machine; warmup; `Req` w/ sampling-param delegation | Request-level scheduling only (one request per step, no batching/interleave); diffusion-only — no KV cache, no AR loop, no token scheduling; stage/batch abstractions inherited our P1–P3 problems |
| **sglang `srt`** (`~/sglang/python/sglang/srt/`) | LLM serving: continuous batching (`Req`/`ScheduleBatch`, EXTEND/DECODE), radix prefix cache, paged KV (`ReqToTokenPool` + allocators), CUDA-graph decode, overlap scheduling, PD disaggregation with bootstrap/transfer queues | **Concepts**: paged KV pool + allocator design, scheduler event loop w/ overlap, chunked prefill, PD-disagg queue pattern, multimodal-embedding cache keyed by content hash | Wholesale embedding: srt is token-centric; its scheduler, radix tree and attention backends assume LLM decode. We borrow designs, not the stack |
| **vllm-omni** (`~/vllm-omni/`, HEAD #4168) | Multi-stage omni serving: frozen declarative `PipelineConfig`/`StagePipelineConfig` (stage_id, `execution_type ∈ {LLM_AR, LLM_GENERATION, DIFFUSION}`, `input_sources` DAG, `final_output_type`), YAML deploy (devices/TP/memory/connectors per stage), per-stage engines in own processes (stock vLLM core for AR; `DiffusionEngine` for DiT, inline-or-remote), `OmniConnectorBase` put/get + `chunk_ready` readiness signals (SHM + RDMA via Mooncake/Mori/Yuanrong — *not* NIXL/NCCL), readiness-set parking + bounded-window-K + preemption for flow control (**no credit-based flow control anywhere**), global orchestrator + per-stage schedulers, `async_chunk` streaming | **Declarative pipeline spec separate from deploy YAML** (the frozen `PipelineConfig` + editable `deploy/<model>.yaml` `DeployConfig` fused by `merge_pipeline_deploy` — adopted here verified, §6.1); connector interface with readiness signals; inline-vs-remote stage execution; per-stage independent schedulers; streaming-input (resumable) requests; **`SupportsStepExecution` as loop-inversion prior art** (opt-in, Qwen-Image-only — we generalize it, §6.2.2); its **3 separate cache subsystems** (`OmniTensorPrefixCache`, `OmniKVTransferManager`, diffusion/cache TeaCache/MagCache/cache-dit) confirm our per-class-pools thesis (§6.3.2) | vllm-omni **does** express shared-weight MoT (`bagel_single_stage`/`lance`: co-resident `*_moe_gen` experts, one resident instance running AR `generate_text` + diffusion `generate_image` in one request) — but only as **one opaque `DIFFUSION` stage the scheduler never sees inside** (`RequestScheduler`, `max_num_running_reqs=1`), so the AR+denoise interleaving is invisible to scheduling/batching; cross-stage KV is a TP-rank/CFG-branch-aware **copy** (`OmniKVTransferManager`), not a shared live cache; **separate-stage** weight sharing is absent (correctly); diffusion batching exists but is opt-in, homogeneous-`SamplingParamsKey`-gated, Qwen-Image-only, and off in every deploy; streaming is linear-chain in practice (routing hardwired `src+1`, `orchestrator.py:1122`) though the schema permits a DAG |
| **sglang-omni** (`~/sglang-omni/`) | Extension over stock sglang: `StageConfig` DAG (factory, gpu, tp_size, process colocation, `next/route_fn/wait_for/merge_fn/stream_to`), per-stage schedulers (`OmniScheduler` wrapping sglang by composition, `SimpleScheduler`, `DllmScheduler` for the LLaDA2-Uni diffusion-LLM thinker — Ming-Omni's DiT talker runs on a different scheduler — plus streaming variants), **Relay** transport (NIXL/SHM/NCCL/Mooncake, credit-based flow control), hidden-state capture thinker→talker, segmenter-based streaming | Relay abstraction + credit flow control; fan-in `merge_fn` / fan-out / `stream_to` edge semantics; scheduler-composition pattern; hidden-state capture as a declared stage output | Same gap: stages own disjoint weights/caches; hybrid AR+diffusion only as AR stage → DiT stage; per-model bootstrap duplication |
| **vLLM-Omni RFC #4084 — "Composable Parallel Strategies"** ([issue](https://github.com/vllm-project/vllm-omni/issues/4084), open, 2026-06) | Proposal: every parallelism scheme (TP/DP/SP/CP/EP/PP/CFG/VAE-PP/stage replicas) becomes a declarative `StrategySpec` — `mesh_axis` (name+degree), L1 routing/aggregation pattern, optional L2 per-layer hooks and L3 kernel factories — stacked into a mesh tree with generic collective-group construction; thin adapter translates onto omni's existing runtime (`l1_owner` marks who routes each axis, preventing double-routing) | **Stacked-axis spec as single source of truth** + generic group construction; the **L1/L2/L3 intervention vocabulary** (we reuse it for both parallelism and the extension system); **pre-flight topology validation**; **CPU-testable composition** via a fake process backend; explicit per-axis ownership; fail-loud on unsupported combos instead of silent fallback | Adapter/translation indirection is their constraint (retrofitting vLLM), not ours — we implement the spec natively in `fastvideo/distributed/`; parallelism-only scope (no loops, caches, or hybrid-model answers); the schemes hardest for us (CP variants, affinity routing) are exactly the parts deferred to follow-up RFCs — design proven, implementation not |
| **verl-omni** (`~/verl-omni/`) | RL post-training for diffusion/omni models (verl lineage): Ray single-controller, vLLM-Omni rollout servers + FSDP2/VeOmni trainers, DanceGRPO/FlowGRPO/DiffusionNFT/Diffusion-DPO recipes (Qwen-Image, SD3.5, Wan2.2), async multi-reward serving, "rollout correction" | Reward-loop architecture (weighted multi-reward, HTTP scorers, async overlap with rollout); SDE-with-logprob scheduler formulation (`FlowMatchSDEDiscreteScheduler`); component-granular sleep levels; group-relative advantage utils; bypass-vs-decoupled log-prob modes | Pays the full two-runtime tax (§8.4): Wan2.2 re-implemented inside vLLM-Omni for rollout *and* in diffusers/FSDP2 for training, stitched by `rollout_correction.py` (IS + rejection masks) — the machinery a single-runtime design exists to delete |
| **miles** (`~/miles/`) | Production LM RL (slime fork, SGLang RL team): SGLang rollout + Megatron training; the deepest extant treatment of train-inference mismatch (bitwise log-probs, TIS/MIS, R3 routing replay, unified FP8, INT4 QAT) | The mismatch toolkit → our consistency ladder C0–C2 + Behavior Record (§8.5); three weight transports behind one interface + per-sample `weight_versions` staleness (§8.6); the custom-rollout-function contract; health-monitored engines; log-prob-diff debug tooling | Dual model definitions + Megatron↔HF weight-name bridges by construction; Megatron is required at *its* scale (≥70B MoE) — which confirms our FSDP2 single-runtime bet only holds at FastVideo's 1–30B scale (§8.4 boundary condition) |
| **cosmos-framework** (`./cosmos-framework/`) | Official Cosmos3 (§4.1) | The model-side requirements; token-budget **packed batching for diffusion**; sampler pluggability; CFG interval gating; CP/CFG-parallel patterns | It is a research framework: no queue, no serving, FSDP-centric; not a runtime to copy |
| **NVIDIA Dynamo** (`./dynamo/`) | Rust/Python fleet orchestration above engines: OpenAI frontend, KV/affinity-aware router (NATS cache events → radix index), prefill/decode disagg coordination with NIXL transfer, SLA Planner (replica autoscaling via K8s CRDs), KVBM multi-tier KV offload, ModelExpress cold-start weight streaming. **Already fronts our ecosystem**: a FastVideo video worker exists (`examples/diffusers/worker.py`), plus sglang-diffusion and vllm-omni backends | The **first-class partner** for everything fleet (§6.3.5): registration/health/drain contract, cost metrics for the Planner, affinity events, NIXL-aligned disagg — production scaling we never build | Rebuilding any of it in-engine; its router keys on token prefixes today and media streaming/cost-models/role-graphs have gaps — addressed as concrete upstream asks A1–A7 (§6.3.6), not workarounds |
| **xDiT / xfuser** (`~/xDiT`) | The de-facto diffusion-parallelism library: USP (`sp = ulysses × ring`), **PipeFusion** (displaced patch pipeline-parallelism that reuses *stale KV from the previous denoise step* — mandatory warmup steps), CFG-parallel (`cfg ≤ 2`), DistVAE (VAE on its own process group), TeaCache/FBCache adapters; world = dp×cfg×sp×pp×tp with nested group init | Two axis-vocabulary updates (§6.3.4): **`pp_patch`** with declared applicability conditions (non-causal only; wins at high resolution / weak interconnect; tolerable for few-step distilled), and the recognition that DistVAE ≡ our decode role pool — no new axis needed; their PipeFusion×SP hybrid rule (KV replication required) as a composition constraint | Wrapper-per-model integration (pipeline + transformer + attention-processor wrappers per family) is exactly the per-model maintenance shape we're eliminating; no serving/scheduler; PipeFusion is architecturally incompatible with causal/AR video (stale KV breaks causality) — a pre-flight validation rule, not a footnote |
| **cosmos-rl** (`~/cosmos-rl`) | NVIDIA's RL framework for physical AI — **now deprecated as Cosmos3 unifies the model families**: single controller + Redis command queues, POLICY/ROLLOUT replicas (colocated or disagg), pluggable rollouts (vLLM, TRT-LLM, diffusers `NFTRollout`, in-process `WFMRollout`), GRPO with **AIPO** one-sided clipping + dual-clip + KL-threshold off-policy masking, **buffer-model async weight sync** on a side CUDA stream, elastic `HighAvailabilitylNccl` mesh rebuild, `weight_version` in every payload | AIPO + off-policy masking into the §8.7 objectives (C0-mode stability kit); the buffer-model async-sync pattern (§8.6); per-payload weight versioning (independently validates our staleness design); elastic-mesh fault tolerance as the recovery reference | Dual model definitions per role (weight mappers, per-role `parallelize`) — the two-runtime tax again; and no production video rollout engine (`NFTRollout` is generic diffusers) — confirmation that the §8 rollout engine is the missing piece in this space, not a duplicate of an existing one (our own landed DiffusionNFT, PR #1450, confirms it from the other side: trainer-side library code sampling through a vendored bare-model loop, §8.1 — still no serving-grade video rollout engine anywhere) |
| **diffusers Modular Pipelines + Guiders** (`~/diffusers`, pulled to head) | HF's blessed re-architecture of pipelines: block composition (`Sequential`/`Conditional`/`Auto`), `LoopSequentialPipelineBlocks` (marked experimental), untyped `PipelineState`/`BlockState`, `ComponentSpec` + `ComponentsManager` (auto-offload), `modular_model_index.json` hub format, and **Guiders** (CFG/PAG/APG/SkipLayer… with start/stop step scheduling); 12+ families incl. Wan, LTX, HunyuanVideo1.5, Flux2 | **Convergent validation of the policy/components pillars only**: Guiders are exactly pluggable CFG policies, and `ComponentSpec` matches our `ModelSpec` components — adopt compatible serialization and a `BaseGuidance` adapter (§6.2.3, §6.6). **It is *not* loop inversion**: `LoopSequentialPipelineBlocks.__call__` raises `NotImplementedError`; every concrete family hand-writes `for i, t in enumerate(timesteps)` inside its own wrapper (`wan/denoise.py:434`, `sdxl/denoise.py:701` — SDXL ships four such wrappers, the subclass forest again). Iteration is block-owned and invisible to any runtime — equally consistent with the alternative this design rejects ("keep loops in stages, make bodies pluggable") | No serving, batching, scheduler, AR loops, or parallelism; untyped blackboard state (their version of our P3); class-defined blocks vs our serializable graph IR — export is clean, import is lossy |
| **torchtitan** (`~/torchtitan`) | PyTorch-native pretraining reference: `ParallelDims` building multi-axis **DeviceMesh** (pp/dp_replicate/dp_shard/cp/tp/ep) with product-of-degrees pre-flight validation and cached submeshes; FSDP2 per-block + MoE prefetch; selective activation checkpointing by FQN; async DCP checkpointing; per-block compile; per-mesh-dim deterministic seeding; `set_batch_invariance()` utilities; an RL experiment (PolicyTrainer → vLLM generator) syncing weights via **TorchStore push/pull with direct RDMA** | The substrate answer for §6.3.4: compile the stacked-axis spec to DeviceMesh via a `ParallelDims`-style builder; TorchStore/DCP as the PyTorch-ecosystem weight-sync precedent for §8.6 (caveat: structure-matched fill, **not** automatic cross-mesh resharding — `WeightSyncPlan` still owns layout math); trainer patterns for `train/` (selective AC, async ckpt, determinism utils); upstream batch-invariance feeding C2 (§8.5) | LLM-pretraining-shaped; no real inference engine (Forge is minimal); full-DTensor everywhere is overkill for engine hot paths — adopt the pattern, not the stack |
| **LiveKit agents / WebRTC** (`~/agents`) | Realtime AI agents over a WebRTC SFU: `rtc.VideoSource.capture_frame()` raw-frame publishing (RGBA/I420; encoding in a native thread — non-blocking to asyncio), `AVSynchronizer`, data channels + named RPCs for low-latency control (interrupts, turn-taking), avatar `VideoGenerator` protocol | The §9 transport trigger matrix: fMP4/WS suffices for segment streaming; **WebRTC is mandatory for interactive world models** (<100 ms motion-to-photon ≈ 20–40 ms action RTT + ≤40 ms/frame inference + ~10 ms encode/decode), with data channels as the action path; engine contract: `OmniEvent` optionally carries **raw frames + PTS**, sessions accept an async action stream + flush RPC | Real operational weight (SFU, TURN, tokens) — adopt only when triggers fire; frames must be CPU-side buffers (GPU→CPU copy on the hot path) |
| **vLLM core — RFC #42770 "Changes in vLLM Model Development"** ([issue](https://github.com/vllm-project/vllm/issues/42770), Woosuk Kwon, 2026-05, open; `~/vllm` pulled to head) | A philosophical reversal by the largest engine: **drop the full-graph `torch.compile` requirement** (manual "big fused ops" with hardware dispatch written directly in model code; local compile kept for adjacent-op fusion; **breakable CUDA graphs** — the SGLang technique, prototyped in PR #42304 — replacing compiler-managed piecewise capture; `forward_context` named as a compiler-compat hack to remove); per-vendor model code for head models; an explicit **`Model`/`ModelConfig` interface** where models own their state and `prepare_inputs` (model runner V2's `ModelState`) and config decouples from HF (#24384). After community pushback (the compile lead's detailed dissent; OOT vendors' CustomOp concerns), the RFC settled on a **two-tier resolution**: hand-fused vendor-specific head models + shared portable definitions for the long tail | Validations and one technique: model-owned state + model-agnostic runner converges with this design (diffusers Modular converges on the components/policy half — not the runner, §6.2.3); HF-decoupled `ModelConfig` ≈ our `ModelSpec` manifest; the two-tier head/tail resolution ≈ our flagship-vs-family split (§6.3.3); and **breakable CUDA graphs** as the capture mechanism for loop step bodies with interceptor branches. Plus a direct work item: FastVideo's `fastvideo/forward_context.py` is inherited from exactly the hack vLLM is deleting — staged retirement starting Phase 1 (shim until Phase 5, §6.3.3). **From the v1 code itself** (beyond the RFC): the unified-progress scheduling structure — adapted to a **cost-model currency**, since bidirectional denoise steps and causal AR tokens are incommensurable work units (§6.3.1) — plus in-step encoder budgeting, the KVCacheGroups hybrid allocator studied and *not* adopted (its single `BlockPool` requires uniform page bytes across groups — `get_uniform_page_size` — which our 150–500× KV-granularity spread violates; we keep per-class budgeted pools, §6.3.2), `EncoderCacheManager` semantics for feature caches, `KVConnectorBase_V1` as the connector protocol shape (§6.3.4), Punica batched multi-LoRA evaluated for the workflow cloud (adopted only for fixed-adapter fast modes — its one-id/baked-scale semantics cannot express ComfyUI's stacked-LoRAs-with-strengths, §9.4), `CuMemAllocator` tag-based sleep (§8.6), and the persistent-batch/staged-write/delta-output CPU discipline behind the ≤2% step budget | The dissent is worth keeping in view: manual fusions don't scale across quant schemes/hardware, and per-fused-op kernel selection pressure re-creates IR-like autotuning ("starts to look very similar to IROps") — arguments for keeping our policy/registry layer rather than going fully manual; vLLM core remains LLM-scoped (diffusion/omni stays in vllm-omni) — and per N5, every borrowing from this repo is admitted on a diffusion/omni justification, never on LLM-serving merits |

---

## 6. Proposed architecture

### 6.0 Overview

```
┌────────────────────────────── Request plane ───────────────────────────────┐
│  OpenAI-compatible server (videos/images/audio/chat, SSE/WebSocket)        │
│  VideoGenerator (offline sync API, unchanged)   AsyncEngine (new)          │
│            OmniRequest ──► admission ──► request queue ──► OmniOutput      │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Pipeline plane ──────────────────────────────┐
│  PipelineSpec (declarative graph, per model family)                        │
│    nodes: Stage (one-shot) | LoopStage (DenoiseLoop, ARDecodeLoop)         │
│    edges: typed Artifacts (Embeddings, Latents[modality], Tokens, KVRef…)  │
│    policies: CFGPolicy, ExpertRouting, AttnMetadata, Precision, FlowShift  │
│    components: shared module registry (one resident copy per worker pool)  │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Execution plane ─────────────────────────────┐
│  Engine scheduler (per-request, graph-node dispatch, role pools)           │
│  StepScheduler (per worker pool: multiplex denoise steps + AR steps)       │
│  Worker pool(s): SPMD procs, TP/SP groups (as today) + CFG-parallel (new)  │
│  CacheManager: paged text-KV ▪ chunked causal-video KV ▪ feature caches    │
│  Connectors (in-proc → SHM → NCCL/NIXL) for cross-pool edges               │
│  Attention backends, torch.compile / cudagraph capture per (node, bucket)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Default deployment is exactly today's: one SPMD pool, all graph nodes co-located, synchronous offline call. Every
serving feature is additive configuration, not a different code path for models. Two cross-cutting surfaces thread
through all planes: the hook/plugin extension system (§6.4) and the declarative parallelism spec (§6.3.4).

### 6.1 Request plane: typed omni I/O

```python
@dataclass
class OmniRequest:
    request_id: str
    task: TaskType                    # explicit: T2V, I2V, TI2V, V2V, V2W, T2I, T2A, T2VS, A2W, REASON, ...
    inputs: list[ModalPart]           # TextPart | ImagePart | VideoPart | AudioPart | ActionPart | LatentPart
    sampling: SamplingParams          # AR knobs: max_tokens, temperature, top_p, stop, seed
    diffusion: DiffusionParams        # steps, guidance (per-modality scales), sigmas/shift, resolution, frames
    outputs: OutputSpec               # requested modalities + streaming flags + return_latents/trajectory
    node_params: dict[str, NodeOverrides] = field(default_factory=dict)
                                      # per-graph-node overrides, validated against the PipelineSpec
    priority: int = 0

@dataclass
class OmniOutput:
    request_id: str
    artifacts: dict[str, Artifact]    # "video": VideoArtifact, "audio": AudioArtifact(sample_rate=…),
                                      # "text": TextArtifact, "action": TensorArtifact, "latents": …
    metrics: RequestMetrics
```

Decisions this encodes:

- **Task is declared, never inferred** (kills P7). Heuristics may *suggest* a default at the API boundary, but the
  pipeline graph branches on `request.task`.
- **Outputs are named artifacts** (kills the `extra["audio"]` pattern, P3). Audio carries its sample rate; text
  carries token ids + decoded string; everything carries provenance (which node produced it).
- **Streaming is part of the request**: `outputs.stream = {"text": True, "audio": chunk_ms=200, "video": per_chunk}`.
  The engine returns an `AsyncIterator[OmniEvent]` when any stream flag is set.
- **Multi-loop graphs are parameterized per node** — decided here, in Phase 0, because external consumers build
  against this schema first: `sampling`/`diffusion` are *defaults*; `node_params["refine"].steps` or
  `node_params["talker"].max_tokens` override per graph node, validated against the node-parameter schema each
  `PipelineSpec` declares. This is what LTX-2's refine loop needs (its own step count and guidance scale are
  first-class fields *today*, `fastvideo_args.py:204-205`) and what any thinker/talker pair needs; the rejected
  alternative — per-model fields on the universal schema — is exactly the `ltx2_*` leakage P3 indicts.
- `VideoGenerator.generate_video(prompt=…)` remains and constructs an `OmniRequest` internally — the offline API is
  a thin shim, preserved verbatim for G5.

### 6.2 Pipeline plane

#### 6.2.1 State: replace the god-batch with Request / RequestState / typed edges

`ForwardBatch` conflated three things. We split them:

- **`OmniRequest`** — immutable user intent (above).
- **`RequestState`** — per-request mutable execution state, owned by the runtime: artifact slots keyed by graph
  edge, loop states, cache handles, RNG state, per-request metrics. A small **typed core** (`latents:
  dict[Modality, Tensor]`, `cond: ConditioningSet`, `timesteps`, `step_idx`) plus a **per-model extension
  dataclass** registered by the pipeline (e.g. `Cosmos3State(packed: PackedSeq, reasoner_kv: KVHandle, …)`,
  `MatrixGameState(mouse, keyboard, kv: KVHandle)`).
- **Edges carry typed `Artifact`s** (Embeddings, ImageLatents, VideoLatents, AudioLatents, TokenIds, HiddenStates,
  KVRef, ActionTensor). Stages declare input/output artifact types; the graph validates at build time — which fields
  a stage reads/writes stops being archaeology (P3, P6).

Trade-off (discussed in §11.5): full typed dataflow everywhere is verbose; a pure blackboard is unverifiable. The
hybrid — typed edges between nodes + a typed per-model state extension inside nodes — is the deliberate middle.
`ForwardBatch` survives during migration as an adapter view over `RequestState` for unmigrated stages.

#### 6.2.2 Stages and loop inversion (the core change)

```python
class Stage(Protocol):                 # one-shot transforms: encoders, VAE en/decode, packers, vocoders
    def forward(self, ctx: StageContext, state: RequestState) -> None: ...

class LoopStage(Protocol):             # iterative computations: the runtime owns the loop
    def init(self, ctx, state) -> LoopState: ...
    def step(self, ctx, loop: LoopState) -> StepResult: ...   # ONE step: one solver step / one token / one chunk
    def finalize(self, ctx, loop: LoopState) -> None: ...

@dataclass
class StepResult:
    done: bool
    emit: list[Artifact] | None = None   # streamed chunks: tokens, audio chunks, decoded frame chunks
```

Two `LoopStage` families cover every paradigm in §4:

- **`DenoiseLoop`** — `init` builds timesteps/sigmas, allocates latents, resolves policies; `step` performs one
  solver step (model forward(s) + CFG combine + scheduler step). Sub-variants by composition, not subclassing:
  joint multi-modality latents (LTX-2, Cosmos3 t2vs), chunked-causal (`step` = one denoise step of the current
  chunk; chunk advance handled by an outer `ChunkRollout` loop node holding the KV handle), few-step distilled
  (timesteps from config).
- **`ARDecodeLoop`** — `init` runs prefill into a `KVHandle`; `step` decodes one token (or one speculative bundle),
  samples, appends, optionally emits; `done` on EOS/max_tokens. Used by Cosmos3 reasoner, omni thinkers/talkers,
  audio-codec generators.

Why inversion is the keystone:

1. **Scheduling**: only the runtime sees step boundaries, so only inversion enables continuous batching of AR steps,
   interleaving denoise steps of concurrent requests, preemption between steps, and fair sharing between a 50-step
   video job and a 4-step image job (P2, P4).
2. **Streaming** falls out: `StepResult.emit` is the universal chunk channel (text tokens, talker codec chunks,
   causal-video decoded chunks) instead of three bespoke mechanisms (P4).
3. **Hybrid models** become a graph of loops over shared components: Cosmos3 = `ARDecodeLoop(reasoner)` →
   `Pack` → `DenoiseLoop(joint)` → decoders — same transformer module bound to both loop nodes (G4).
4. **Checkpointing/elasticity**: a `LoopState` is a serializable resume point (future: migration between workers,
   crash recovery for 1000-step jobs).

Cost: the loop body must be re-entrant and the per-step dispatch must stay cheap (§12, risk 2). `DenoiseLoop.step` for
batch-of-1 video compiles/captures exactly like today's inner loop body; nothing tensor-level changes.

Note the contract's deliberate smallness — this is the design's main simplicity lever: the runtime requires only
`init/step/finalize`. *How a family implements `step` is free* — composed from policies by default, or a custom
body when the math is genuinely entangled (§6.2.3). Loop inversion is about **who owns iteration**, not about how
step bodies are factored; everything the scheduler, streamer, and parity harness need comes from the small
contract, never from the factoring.

#### 6.2.3 Policies: dissolving the `isinstance` forest

By default, a model-conditional currently probed inside `DenoisingStage` becomes a small strategy object **resolved
once at pipeline build** and injected into the loop (kills P1 without 22 subclasses) — *default*, not admission
requirement: families whose step bodies braid across slot boundaries keep custom steps under the escape hatch below:

| Policy | Replaces | Examples |
|---|---|---|
| `CFGPolicy` | chunked cond/uncond logic, adaptive-gate delta reuse, LongCat batched-CFG, per-modality scales, Cosmos interval gating, embedded-guidance (Flux ×1000) | `ClassicCFG`, `BatchedCFG`, `AdaptiveGateCFG`, `PerModalityCFG`, `EmbeddedGuidance` (cfg-parallel is a parallelism *axis*, not a policy — see below) |
| `ExpertRouting` | Wan2.2 `boundary_ratio` transformer switch | `BoundaryTimestepRouting(transformer, transformer_2)` |
| `AttnMetadataProvider` | VSA / V-MoBA / STA builder `isinstance` chain in the loop | registered per backend; built in `init`, refreshed per step only if needed |
| `PrecisionPolicy` | `prefix == "Flux"` autocast hacks, `scheduler_step_in_fp32`, cast flags | declared in model config |
| `FlowShiftPolicy` | Cosmos3 resolution-bucket shift, per-task sigmas | config-driven lookup table |
| `ConditioningInjector` | camera/action/pose/audio conditioning plumbed through batch fields | per-model injectors that write into `RequestState.cond`; the loop is agnostic |

**The step skeleton and the policy contract.** A three-phase step (forward → CFG → scheduler step) does not
factor the shipped loops; they need more slots, with dependencies that cross policy boundaries. The default
`step` body is a fixed skeleton of ordered, typed slots:

```
prepare_inputs → [per guidance branch: pre_forward → forward → post_forward] → combine
              → scheduler_step → post_step → inter_step
```

with three rules. **(1) Binding vs state.** Policy *bindings* resolve at pipeline build (which classes, which
slots); policy *state* is allocated per-request in `LoopState` at `init` — the same scoping rule as plugins,
because the same failure applies: `AdaptiveGateCFG`'s cached delta (`denoising.py:338-343, 507-551`) is
per-request mutable state, and a build-time singleton would smear request A's CFG delta into request B the moment
interleaving lands. **(2) Observation.** Policies read a typed per-step `StepContext` (timestep, branch, active
expert/model id, sampler coefficients) — the channel by which `AdaptiveGateCFG` observes `ExpertRouting`'s switch
to invalidate its delta (today an inline `id(model)` check), and by which the guidance **branch vocabulary is
policy-declared per request at `init`** (plugin per-branch state scopes over the *declared* set — LTX2's 1–4
runtime-decided passes declare theirs up front, not a fixed cond/uncond pair). **(3) Slots cover the common
shapes, not all shapes** — verified against shipped code: TI2V's post-`scheduler.step` latent clamp is a
`post_step` policy (`denoising.py:570-573`); CausalDMD's between-step renoising with expert-boundary-dependent
noise selection is `inter_step` × `ExpertRouting` (`causal_denoising.py:268-301`); Cosmos2.5's per-frame timestep
vectors and per-step GT re-clamp are `prepare_inputs`/`pre_forward`. But Cosmos's conditioning-frame injection
consumes the *sampler's* EDM coefficients inside each CFG branch with an x0-space combine
(`denoising.py:845-933`), and LTX2's guidance passes alter the network via forward kwargs
(`ltx2_denoising.py:503-605`) — braids no slot factoring owns cleanly.

**CFG-as-policy is proven by precedent, not speculative.** vllm-omni's `CFGParallelMixin` already unifies
sequential-2-forward, batched, *and* cfg-parallel under one pair — `predict_noise_maybe_with_cfg` +
`combine_cfg_noise` (`vllm_omni/diffusion/.../cfg_parallel.py:76-212`) — so one shared denoise-step body plus a
swappable `CFGPolicy` expressing classic (2-forward), batched (1-forward stack+chunk), adaptive-gate (cached-delta
reuse with model-id invalidation), per-modality, and embedded-guidance (single-branch) is an existence proof, not
a hope. The right reading of vllm-omni's "three CFG mechanisms" is **one unified in-loop pair + one parallelism
axis + one orchestrator layer**, not three independent CFG systems. We make that boundary explicit in **three
layers**: **(1)** `CFGPolicy` is *in-loop* — it owns the branch vocabulary, the combine formula, and per-request
mutable state (the adaptive-gate cached delta is the canonical state case); **batched-vs-2-forward is a dispatch
detail *inside* one policy, not a separate mechanism** — which is why there is no standalone `BatchedCFG` axis to
collide with parallelism. **(2)** cfg-parallel is a *parallelism axis* — it shards `policy.predict`'s branches
across ranks and runs the **same rank-invariant combine** on every rank, and **composes under any policy**; a
request never owns both a batched-CFG policy variant *and* a `cfg` parallel group (declaring both is rejected,
§6.3.4). **(3)** companions are an *orchestrator pattern* — a request is split into companion sub-requests
*upstream* of diffusion, and the loop receives already-computed conditioning unchanged. Two caveats the slots must
respect: the combine runs in the **step body's numeric space** (Cosmos combines in **x0-space** post-EDM, not
noise-space — `denoising.py:845-933`); and **embedded-guidance/Flux is a degenerate single-branch identity-combine
policy** (guidance rides inside the forward kwarg), *not* "no CFG".

Which is why the scheduler's contract is deliberately smaller than the policy system: **a family may ship a
custom `step` body** — a hand-written function using samplers, CFG math, and conditioning utilities as a library
(the Cosmos3 port's denoise-engine pattern, legitimized). A custom step is still a `LoopStage`: scheduled,
budgeted, preempted, streamed, observed, and parity-gated exactly like a composed one, and the skeleton-owned
plumbing (offload dance, attention metadata, autocast, trajectory capture) is provided either way. Policy
factoring is the default because it deletes duplication; it is **not an admission requirement**. So the honest
form of the claim: a new model contributes *at minimum* a graph spec and a step body — policies where they fit —
and never edits shared loop code.

External validation, stated at its true scope: diffusers' **Modular Pipelines** independently converged on the
*policy* pillar — **Guiders** (CFG, PAG, APG, SkipLayer…, each with start/stop step scheduling) are exactly
pluggable CFG policies, and `CFGPolicy` stays adapter-compatible with their `BaseGuidance` interface so model
ports and guidance research transfer in both directions (§6.6). It does **not** validate loop inversion: their
loop blocks own their own iteration, invisible to any runtime (see §5). The honest position: **there is only
narrow, opt-in precedent — vllm-omni's `SupportsStepExecution` (`prepare_encode`/`denoise_step`/`step_scheduler`/
`post_decode`, `diffusion/models/interface.py:44-67`) is runtime-owned diffusion iteration at step granularity,
but Qwen-Image-only and off in every deploy; no surveyed system makes it the always-on default. This design
generalizes that contract to a universal, always-on one** (diffusers' loop blocks, by contrast, own their own
iteration), and the Phase-1 bit-identical parity gates are what carry that risk (risk 3).

#### 6.2.4 PipelineSpec: declarative graphs

Frozen, validated, registered per model family (the vllm-omni `PipelineConfig` shape, extended with loop nodes,
branches, and fan-out):

```python
COSMOS3_OMNI = PipelineSpec(
    family="cosmos3",
    components={"transformer": …, "vae": …, "sound_vae": Component(optional_for={…}, required_for={"t2vs"}),
                "vision_encoder": …, "tokenizer": …},          # declared, eagerly loadable per task (kills P8)
    nodes=[
        Node("validate",  InputValidation()),
        Node("tokenize",  Cosmos3Tokenize()),                                    # text → TokenIds
        Node("vis_cond",  VAEEncodeConditioning(), when=task_in(I2V, V2W)),      # image/video → clean latents
        Node("reason",    ARDecodeLoop(model="transformer", pathway="und"),      # prompt upsampling / REASON
                          when=task_in(REASON) | opt("upsample_prompt")),
        Node("pack",      Cosmos3Pack()),                                        # → PackedSeq artifact
        Node("denoise",   DenoiseLoop(model="transformer",                       # SAME module as "reason"
                          modalities=[VISION, ACTION?, SOUND?],
                          cfg=PerModalityCFG(interval_gated=True), sampler=FromConfig())),
        Node("vdec",      VAEDecode("vae")),                                     # fan-out ↓
        Node("adec",      AudioVAEDecode("sound_vae"), when=task_in(T2VS)),
        Node("assemble",  AssembleOutput()),                                     # → OmniOutput artifacts
    ],
    edges=[…typed…],
)
```

Properties:

- **Branches** on `request.task` and request options replace geometry heuristics (P7).
- **Fan-out/fan-in** are real: video and audio decode in parallel after a joint denoise; thinker hidden-states can
  feed both a text detokenizer and a talker (sglang-omni `wait_for/merge_fn` semantics).
- **Components are shared by reference.** Two loop nodes binding `"transformer"` get the same resident module —
  this is the MoT requirement, impossible in engine-per-stage designs.
- **Linear pipelines are the degenerate case**; the existing stage lists translate mechanically.
- A separate **deploy config** (YAML) maps nodes → pools/devices/parallelism (vllm-omni's split of frozen pipeline
  vs deployment), defaulting to "one pool, everything colocated".

### 6.3 Execution plane

#### 6.3.1 Engine and two-level scheduling

**Level 1 — engine scheduler (per request, per graph node).** Async intake → admission (validate, resolve
PipelineSpec, allocate request state) → dispatch nodes to worker pools as their inputs become ready; a request
state machine generalizing `multimodal_gen`'s `RequestTracker` (waiting → node-running → node-done → …), with
role-pool capacity dispatch (N:M:K encode/denoise/decode) available when configured.

**Level 2 — StepScheduler (per worker pool).** The new capability. Each pool runs an event loop over **ready loop
steps**:

```
while True:
    work = select(ready_loops, policy)        # rank0 decides, broadcast to pool (SPMD-consistent, as today)
    ├─ AR group:       batch decode steps of all active ARDecodeLoops (continuous batching;
    │                  paged KV; chunked prefill for long prompts/vision contexts)
    ├─ Denoise group:  pick one (or a shape-compatible batch of) DenoiseLoop step(s)
    └─ interleave by priority/deadline; preemption only at step boundaries
```

Batching semantics are **paradigm-aware** (this is where "one scheduler" earns its keep):

- **AR steps**: classic continuous batching (sglang-style), batched across requests every iteration. CUDA-graph
  decode per batch-size bucket.
- **Image diffusion**: cross-request batching with resolution buckets — the workload where batching pays most
  (Flux2/SD3.5 serving).
- **Video diffusion**: batch-of-1 with TP/SP (plus the new CFG-parallel axis) stays the default (memory-bound); concurrency comes from
  *interleaving* steps of multiple requests (a 4-step distilled job no longer queues behind a 50-step job) and,
  later, Cosmos-framework-style token-budget packing of heterogeneous samples as an opt-in.
- **Hybrid (Cosmos3)**: reasoner decode steps join the AR group; the joint denoise joins the denoise group; both hit
  the same resident weights, so the StepScheduler is also the **mode multiplexer** the MoT model needs. The
  multiplexer needs a *parallelism* answer too, because the two loop types want disjoint layouts — denoise wants
  SP(+CFG); AR decode is sequence-length-1, where SP has nothing to shard and CFG doesn't exist. The stated
  answer, rather than a deferred contradiction: on a `[cfg × sp]` pool, **AR decode runs data-parallel across the
  cfg×sp weight-replica axes** (every replica already holds full weights; the decode batch shards across them,
  paged text-KV partitioned per replica — not duplicated), optionally over per-pathway TP subgroups (§6.3.3);
  and **pure-REASON traffic is routable to differently-shaped pools** via cost classes and affinity routing, so
  reasoner token latency is not gated by indivisible denoise steps except when a single request genuinely
  interleaves both on the same weights. The Phase-4 gate measures pool efficiency (reasoner tokens/s/GPU at
  target denoise throughput), not just the re-prefill strawman.

**Mechanics adopted from the vLLM v1 scheduler** (`vllm/v1/core/sched/scheduler.py` — production-proven, and a
cleaner formulation than our group split above):

- **Unified progress accounting — but the budget currency is cost, not counts.** v1 has no prefill/decode phases:
  every request is `num_computed_tokens` catching up to a target under a per-iteration token budget, and that
  works because a *token* maps near-linearly to work in the LLM regime. **A step does not.** A bidirectional
  denoise step re-attends over the entire latent sequence at O(L²) with zero KV amortization — every step pays
  full price — while AR prefill is causal-masked incremental work against a cache (O(n·C) for n new tokens on
  context C), a decode step is ~O(C) per layer, and a chunked-causal video step sits between (one chunk vs cached
  context). Counting "steps" would put items three orders of magnitude apart in one bucket. So we keep v1's
  *accounting structure* — `steps_done`/`steps_target` progress counters, one admission/budget loop — but the
  budget is **predicted GPU-time from a per-(model, phase, shape) cost model**: each schedulable item (a denoise
  step at given L and CFG mode, an AR prefill chunk, a decode token-batch, an encoder forward) converts to the
  common currency before admission. The cost model is policy-aware (batched-CFG ≈ 2× batch vs 2 forwards; VSA/
  NATTEN sparsity changes the attention term; cache-dit changes the block count) and **it is the same cost model
  we publish to Dynamo's Planner and router (ask A2, §6.3.6)** — one model, two consumers, calibrated online by
  the Profiler observer.
- **Encoder work budgeted inside the same step — scoped to what diffusion actually needs.** v1 interleaves
  encoder forwards with decode because VLM prefill demands it. Our shape is different: for diffusion, encoding is
  a one-shot prologue per request, so the value here is narrower and specific — *waiting* requests' encode work
  fills budget headroom alongside *running* requests' loop steps on the same pool (hiding encode latency under
  denoise), with the full interleaved pattern reserved for omni workloads that genuinely stream multimodal
  context mid-request. We take the budgeting mechanism, not the VLM workload assumption.
- **CPU discipline, applied where it bites.** v1's machinery (persistent `InputBatch` state allocated once with
  sliced H2D copies, staged writes flushed once per step, **delta-only scheduler outputs**, async dispatch via
  Futures) exists because LLM decode steps are ~10 ms and CPU time is the bottleneck. Our pressure is uneven: it
  bites for the **AR decode group and batched image steps** (short, frequent) — adopt fully there — while video
  denoise steps are 50–500 ms+ and tolerate a simpler dispatch path; we take the async-Future overlap everywhere
  but do not pre-build LLM-grade batch plumbing for loops that don't need it.
- **Same-step speculative contract.** Draft tokens ride the scheduler output and are verified by rejection
  sampling within the same step — the proven shape for `ARDecodeLoop` multi-token bundles (§12, open question 10).

Two properties make cost-currency scheduling *easier* here than vLLM's problem, and one makes it harder. Easier:
diffusion **baseline** costs are static and known at admission — (resolution, frames, steps, CFG mode, model)
determine per-step cost, flat across steps (no growing context) — with the honesty caveat the doc owes itself:
**admission always uses the conservative baseline**, because the design's own flagship features make realized cost
runtime-dependent (cache-dit skip decisions are residual comparisons unknowable in advance; VSA tile selection is
content-dependent; AR decode lengths are unbounded — budgeted at the `max_tokens` cap with refund on early EOS).
Runtime telemetry refines calibration and utilization accounting; it never licenses admission optimism. Harder:
**a denoise step is indivisible**, so the largest step bounds
iteration latency — one 30 s-1080p step can block an interactive Dreamverse segment regardless of budgets.
Mitigations, in order: don't co-schedule jumbo steps with latency-class work (cost classes map to pools — which
model-affinity pooling §9.4 wants anyway); shrink jumbo step wall-time with SP *within a node* (pools are
single-node in v1, §6.3.4 — SP degree caps at intra-node GPU count, stated rather than implied); and
admission-time SLO classes so the Planner (§6.3.5) scales pools per class instead of one pool absorbing both.

**Failure isolation, cancellation, and memory admission — the soundness conditions of multiplexing.**
Step-multiplexing changes the failure class categorically: today, one request per pool means one request's CUDA
error is its own problem; on a shared pool, a mid-step illegal access poisons the CUDA context and any in-flight
collectives for *every* co-scheduled tenant, including resident session caches. This is designed **with** Phase 2,
not after it:

- **Cancellation is first-class and common-path.** Vibe directing makes abandoning in-flight generation the
  *normal* user action, not an edge case. Cancel takes effect at the next step boundary: queued steps drop,
  `LoopState` and cache handles release, partial artifacts report with explicit `cancelled` status. The
  flush/interrupt RPC (§9.1) is this mechanism, not a separate one.
- **Failure classification.** *Request-fatal* (admission/step allocation failure, NaN flagged by NaNWatch, errors
  attributable to one request's step): abort that request via an **SPMD-consistent abort broadcast** — the exact
  dual of the rank-0-decides scheduling broadcast — release its state; fan-out graphs deliver the named artifacts
  completed so far plus a structured error (partial-artifact semantics). *Pool-fatal* (illegal memory access,
  NCCL desync/timeout, context corruption): pool re-init — kill and restart workers (the **sentinel-fd
  worker-death watch is inherited from Dreamverse's `gpu_pool.py:542-586` and explicitly preserved**, so Phase 2
  is not a reliability regression for the flagship customer), re-establish collectives, invalidate all
  pool-resident caches (sessions re-prefill from durable inputs; requests resume from serialized `LoopState`
  where one exists, else requeue), and report through the Dynamo health surface (§6.3.5). vLLM v1 needed exactly
  this machinery (`EngineDeadError`, `abort_requests`); we build it at the same layer.
- **Memory is the second budget axis.** vLLM's token budget is half its admission story; the other half is
  `allocate_slots` in the same loop. We keep both halves: every schedulable item carries (predicted time,
  **resident bytes, peak-activation bytes**) — latents, conditioning sets, CFG duplicates, and activation peaks
  live in `RequestState` *outside* the CacheManager, so an **admission planner** bounds Σ resident + worst-case
  peak (profiled per model × shape, vLLM-style) against the pool budget. Two requests that fit individually but
  jointly OOM are rejected at admission, not discovered at step 37.
- **Preemption semantics, by state size.** "Preemption only at step boundaries" now says what happens to the
  loser's multi-GB state: small resident states stay on-GPU; medium offload to pinned host memory; large or idle
  drop and resume from serialized `LoopState` — with the re-encode/re-prefill economics priced by the cost model
  (different economics from token-KV recompute, stated rather than assumed).

The non-serving path bypasses level 1 entirely: `VideoGenerator` submits one request and drives the pool to
completion synchronously — preserving today's latency profile and debuggability (G5, G6).

#### 6.3.2 CacheManager (unifying P5)

One manager, handle-based. First, the proportionality note (N5): **KV is the minority case here** — bidirectional
diffusion, the majority of today's traffic, allocates *no* KV at all, so the KV machinery below is lazy: pools
materialize only when a `PipelineSpec` declares KV-bearing loops (AR pathways, causal video, omni), and a pure
bidirectional deployment carries none of it.

For that KV-bearing minority, adversarial review caught this design importing a property vLLM does not have: the
single-`BlockPool`/no-fragmentation guarantee exists *because* physical bytes-per-block are uniform across all
groups — `kv_cache_utils.py` asserts a single page size (`get_uniform_page_size`), its docstring calls breaking
this "non-trivial due to memory fragmentation concerns", and groups differ only in tokens-per-block at equal byte
size. Our classes differ by **150–500× in natural granularity** (a text-KV page ≈ 64 KB/layer; a causal-Wan
latent-chunk slab is 9.6–32 MB/layer), and the demand classes are *workload-decoupled* (text-KV and chunk-KV vary
independently with request mix), so one unified pool either strands slab-sized memory under token traffic or
degenerates into static partitioning anyway — which is exactly what vLLM's one multi-page-size path (DeepseekV4)
does at startup. **Decision — simpler and honest: separate per-class pools with static budgets.** Each cache
class gets a budget carved at pool init from the deploy config (cost-model defaults, driven by what the
`PipelineSpec` declares): the text-KV class runs a vLLM-style paged block allocator; the chunk-KV class runs a
**slab allocator** (one latent-frame chunk per slab). Both sit behind the same `KVHandle` interface. The
fragmentation/deadlock argument: classes are statically partitioned, so cross-class fragmentation is impossible
by construction; within a class, granularity is uniform, so the standard arguments apply; rebalancing budgets is
a pool-drain operation — rare (class demand follows deployment workload mix, not per-request noise) but not free:
a drain is priced by the cost model like any quiescing barrier (§6.3.2's weight-transition pricing), and a slab
budget shrink must wait out or evict session-resident slab caches, so rebalancing is a scheduled operation, not a
hot-path one. (This is the original two-pool sketch returning with the justification it lacked — the single-pool detour
was built on a misread of vLLM.) Our cache classes:

- **Paged text KV** (paged class): used by `ARDecodeLoop` (Cosmos3 reasoner — replacing the
  re-prefill-per-token loop; omni thinkers/talkers). No radix prefix tree in v1 of our build; block-hash prefix
  reuse later (the allocator design accommodates it).
- **Chunked causal-video KV** (slab class): per-DiT-layer K/V for latent-frame chunks with
  sink/local-window semantics — the formalization of `causal_wanvideo.py`'s ad-hoc dicts (Wan-Causal,
  MatrixGame2/3, LongCat-VC, Cosmos3 world-model rollout). vLLM's `ChunkedLocalAttentionSpec` /
  `SlidingWindowSpec` managers (out-of-window recycling, per-request admission caps) are the reference
  *semantics*, re-expressed over slabs. A declared **training mode** disables mid-rollout recycling and keeps
  grad-aware index snapshots (the `wan_causal.py:119-120,405-431` behavior, §8.2) — the inference recycling rules
  are not silently applied under autograd. **MoT models fall out naturally**: the und pathway draws from the
  paged class, the gen pathway from the slab class or nothing — independent budgets, no interference.
- **Feature caches**: text-embedding cache (content-hash keyed; reused across CFG branches, retries, and requests),
  vision-encoder/deepstack cache (Cosmos3 image-conditioned reasoning), cache-dit (DBCache/FBCache) residual caches scoped
  per `LoopState`. The embedding/encoder caches adopt vLLM's `EncoderCacheManager` design
  (`mm_hash`-keyed, reference-counted with a freeable FIFO, budget-aware allocation). Two invalidation regimes,
  not one: RL `update_weights` bumps a **weight epoch** that invalidates wholesale; but workflow-cloud traffic
  patches *text encoders* per request (ComfyUI LoRAs carry `lora_te*` keys and a separate `strength_clip`), so
  the cache key is **partitioned, not flushed**: `content-hash × encoder identity × adapter-set hash
  (ids + strengths) × weight epoch`. Without the adapter component, two workflows sharing a prompt but differing
  in te-LoRA stacks would silently serve each other stale embeddings — in the exact product whose trust claim is
  reproducibility (§9.4).
- **Weight/adapter cache** (Phase 3+, pulled by multi-tenant workflow execution, §9.4): disk→CPU→GPU checkpoint and
  LoRA tiers, LRU, content-hash-keyed `ModelSpec` resolution, fast patch/unpatch with stacked-LoRA semantics — and
  **model-affinity scheduling** in the engine (route requests to pools where the checkpoint is already resident,
  the diffusion analog of prefix-affinity routing). Crucially, **weight state is schedulable state**: components
  are one resident copy per pool, so a patch/unpatch transition is a *pool-quiescing barrier* — drain in-flight
  steps, apply/undo `W += scale·BA` shard-consistently across ranks (undo via CPU master-copy restore or 2×
  backup, chosen per deploy memory budget), re-admit. Two interleaved loops requiring different patch states
  cannot coexist on one pool; the StepScheduler therefore treats transitions as scheduled, costed operations —
  the cost model carries a **weight-transition term**, and affinity scheduling exists precisely to minimize how
  often it is paid (a quiesce-vs-queue policy decides whether a lone divergent request waits or forces the
  barrier). Until this lands, the engine serves homogeneous model families per pool.

Handles (`KVHandle`, `FeatureRef`) live in `RequestState`; lifetimes are tied to the request (or to an explicit
session for interactive world models, where the chunk KV must persist across many requests of one game session).

#### 6.3.3 Model layer: MoT and packed sequences as first-class citizens

To make Cosmos3-class ports native rather than heroic (per §4.1):

- **`PackedSeq`** core type: modality segments, per-segment attention mode (causal / full / sparse-NATTEN /
  temporal-causal), sample offsets, position-id planes (3D-MRoPE with FPS modulation), cond masks. Pack/unpack
  utilities replace order-sensitive flat-latent slicing (one of the documented Cosmos3 port frictions, §3.2/P3).
- **Dual/multi-pathway layer conventions** in `fastvideo/layers/`: `MultiPathwayAttention` (per-pathway QKV/O +
  shared attention over `PackedSeq`), `PathwayMLP` (dense or MoE per pathway), modality-routing by segment — TP
  sharding defined per pathway. The Cosmos3 DiT port refactors onto these; future MoT/Transfusion-lineage models
  reuse them.
- **Attention backend contract extended** to accept per-segment modes/masks, so backends (FA varlen, flex, NATTEN)
  plug into MoT models the same way they plug into plain DiTs.
- **Sampler registry** (UniPC, EDM, Euler-FM, fixed-step distilled) with strict config validation that **rejects or
  normalizes** incompatible checkpoint scheduler configs (the karras-sigmas NaN incident becomes a load-time error,
  not a black video).

**Compile & graph position (informed by vLLM RFC #42770 and its discussion).** The largest engine just reversed
its compiler-first philosophy: full-graph `torch.compile` is being dropped in favor of manual fused ops in model
code, local compile for adjacent-op fusion, and **breakable CUDA graphs** (the SGLang technique, vLLM PR #42304 —
piecewise capture with no Dynamo/Inductor involvement). Our position, stated explicitly:

- **Never full-graph across the engine.** Per-component/per-block compile where it pays (today's per-component
  flags, kept); manual fused ops are *permitted in model code* — the RFC's lesson — rather than forced through
  shared abstractions.
- **Breakable CUDA graphs for loop step bodies — as an optimization tier, not a foundation.** The technique lets
  cache-dit skip decisions and policy branches become break points between captured segments instead of
  graph-invalidating recompiles, and video's batch-1 fixed shapes are the friendly case for capture. But it is
  marked experimental in vLLM's own tree (`vllm/envs.py:724`), and interceptors × CFG branches × expert routing ×
  observers-needing-eager multiply graph variants. The contract therefore is: **the eager path is the always-correct
  baseline**; graphs are captured for a deliberately small matrix (per model, shape-bucket, branch-profile), and
  interceptor-heavy configurations may simply run eager with the cost surfaced in metrics. The ≤2% Phase-2 latency
  budget is gated on the Phase-1 loop-inversion overhead measurement, not on breakable graphs maturing upstream.
- **Two tiers, like vLLM's post-discussion resolution**: flagship (recipe, runtime) pairs (a FastWan/LTX-2 NVFP4
  build on B200) may carry hand-fused, hardware-specific module variants, swapped in per `DeployConfig` without
  touching the `PipelineSpec`; the 20+-family tail stays on shared layers + policies. The dissent in that RFC is
  our guardrail for the tail: manual fusion doesn't scale across quant schemes and hardware, which is exactly what
  the policy/registry layer is for.
- **Retire `fastvideo/forward_context.py` — staged, not deleted in place.** It is inherited from precisely the
  global the RFC names as a compiler-compat hack, but today it is also the live attention-metadata transport for
  `fastvideo/attention/layer.py:93` *and* the frozen legacy `training/` stack
  (`training_pipeline.py:405`, `distillation_pipeline.py:672`) — deleting it in Phase 1 would violate N2's frozen
  boundary. Staging: Phase 1 removes it from the **migrated inference path** (new loops carry `StageContext` +
  `AttnMetadataProvider` + model-owned `RequestState` extensions — the same "model owns its state" conclusion
  vLLM's `Model`/`ModelState` interface reaches), while the module itself survives as a compatibility shim fed by
  the new context, so attention layers and every unmigrated/legacy caller are untouched; the file is deleted in
  Phase 5 with the legacy paths.

*What a media engine would feed back to RFC #42770's `Model` interface* (we have standing to comment — our
experience is the evidence): (1) `prepare_inputs`/`forward` assume one phase per request; diffusion needs
multi-phase requests (encode once, then N denoise forwards) with **memoized cross-step state** (text embeddings
reused across all steps) — a small contract change that would let vllm-omni's diffusion stages implement the same
interface; (2) formalize encoder-output caching in the interface (their `EncoderCacheManager` exists but sits
outside the `Model` contract); (3) keep the interface tensor-type-agnostic (the vllm-metal request) so non-torch
runners can implement it.

#### 6.3.4 Placement, connectors, and external engines

- **Default**: one SPMD pool; node→pool mapping trivial; "connector" is an in-proc handoff (zero overhead).
  **Pools are single-node in v1** — a load-bearing decision made explicitly: the engine never owns a cross-node
  NCCL mesh inside one pool (no cross-node collective bring-up, no NCCL-timeout fault domains, no multi-node
  drain protocol); multi-node scale is *multiple pools* fronted by Dynamo (§6.3.5). Consequences accepted: SP
  degree ≤ intra-node GPUs, which bounds the jumbo-step mitigation (§6.3.1) and caps single-pool model scale —
  the revisit trigger is a model or resolution that needs more. `RayDistributedExecutor`, today's multi-node
  path, is **frozen with the legacy adapter** and retires at Phase 5; its use case is superseded by
  Dynamo-fronted pools.
- **Role pools** (opt-in): encode / denoise / decode pools with capacity dispatch — port of `multimodal_gen`'s
  disaggregation, now expressible per graph node rather than per hardcoded role.
- **Per-node placement** (opt-in): deploy YAML pins nodes to devices/pools (vision encoder on its own GPU; vocoder
  on CPU); cross-pool edges select a `Connector` (transport: SHM/Mooncake/Mori/Yuanrong today, with NIXL as a
  target — *not* "vllm-omni's NIXL/NCCL", which it does not use). The `chunk_ready` readiness signal is adopted
  from vllm-omni's `OmniConnectorBase`; **credit-based flow control is sglang-omni's Relay design, not
  vllm-omni's** (vllm-omni uses readiness-set *parking*, no credits) — so credit-flow here is a FastVideo
  addition *over* vllm-omni's readiness model, composed with sglang-omni's proven design. For KV/cache-bearing
  edges specifically, the
  connector speaks a protocol shaped like vLLM's `KVConnectorBase_V1` (scheduler-side query/alloc/finish roles +
  worker-side async `start_load`/`wait_for_layer`/`save` with layer-pipelined loading and `invalid_block_ids`
  fault reporting) — the same interface NIXL/LMCache/Mooncake already implement, which is also what makes Dynamo
  ask A4 (§6.3.6) an extension of an existing contract rather than a new one. **Adopt vllm-omni's TP-rank-aware +
  CFG-branch-aware KV payload slicing** (`OmniKVTransferManager` `build_rank_aware_send_keys`/`recv_keys`): a
  shared-weight MoT split across parallel groups must slice KV by rank *and* by CFG branch on the wire, not ship a
  monolithic blob.
- **External-engine node** (opt-in, later): an `ARDecodeLoop`-shaped adapter that delegates a *separable* AR stage
  (a 235B thinker) to a colocated sglang/vLLM instance over the connector interface. Useful for separable omni
  models; never required for MoT models (which must stay native).
- **Declarative parallelism (adapted from vLLM-Omni RFC #4084).** Pool meshes and engine topology are described by
  one stacked axis spec — e.g. `parallel: [dp(2), cfg(2), sp(4), tp(2)]` per pool — with collective-group
  membership computed generically (the RFC's mesh-tree construction) instead of hand-wired per combination in
  `fastvideo/distributed/parallel_state.py`. We adopt three properties outright: **(1) pre-flight validation** —
  rank counts, group membership, and *ownership conflicts* are build errors; each axis records who realizes it
  (e.g. CFG branches can be realized by a batching `CFGPolicy` (1-forward stack+chunk) *or* a `cfg` parallel group —
  declaring both is rejected, the RFC's `l1_owner` / no-double-routing rule applied to our policy-vs-axis split); **(2) fail-loud** on
  unsupported combinations rather than silent fallback; **(3) CPU-testability** — engine scheduler, graph dispatch,
  and placement planning run against a fake worker pool so topology logic is unit-tested without GPUs (FastVideo CI
  is GPU-starved; this is how the composition layer stays tested). The RFC's L1/L2/L3 intervention vocabulary —
  batch routing / per-layer behavior / kernels — maps cleanly onto this design (L1 = engine + StepScheduler, L2 =
  TP/SP/pathway-sharded layers, L3 = the attention-backend registry) and is reused for the extension system below.
  We implement the spec natively rather than depending on `composable_parallel` (§11.11).
- **Axis vocabulary v1 + substrate** (informed by xDiT and torchtitan). Axes: `dp`, `cfg` (≤2), `sp = ulysses ×
  ring`, `tp`, and **`pp_patch`** — xDiT's PipeFusion displaced-patch pipelining, which exploits diffusion's
  step-to-step redundancy by reusing stale KV from the previous denoise step (mandatory warmup steps; wins at high
  resolution and on weak interconnects; tolerable for few-step distilled models) and is **invalid for causal/AR
  models** — an applicability condition the pre-flight validator enforces per `PipelineSpec`, since stale KV breaks
  causality. xDiT's DistVAE needs no new axis: it is our decode role pool. Composition constraints travel with the
  axes (e.g., `pp_patch × sp` requires KV replication inside attention). Implementation substrate: the spec
  compiles to **PyTorch DeviceMesh** via a `ParallelDims`-style builder (torchtitan's pattern — product-of-degrees
  validation, cached submeshes); the trainer consumes the mesh natively (FSDP2/DTensor), while engine hot paths
  keep raw process groups where DTensor indirection isn't earned.

#### 6.3.5 Production scale: Dynamo as the first-class fleet partner

Everything above this line is a **single-deployment engine**: one pool (or a few colocated role pools) on one
node-group. Truly scaling — multi-node fleets, SLA-driven autoscaling, cross-node routing, cold starts, fleet
fault-recovery — is a different discipline, and the design's position is that **FastVideo never builds it**:
NVIDIA Dynamo (`./dynamo/`) is the named first-class partner for that layer. This is not aspirational — the
integration already exists in embryo: `dynamo/examples/diffusers/worker.py` registers FastVideo with
`register_llm(ModelInput.Text, ModelType.Videos, …)` and serves `VideoGenerator` through `@dynamo_endpoint`
(today behind an `asyncio.Lock`, one request at a time — precisely the limitation Phase 2 removes), and Dynamo
already fronts our sibling stacks (sglang's diffusion worker via `components/src/dynamo/sglang/init_diffusion.py`,
vllm-omni under `components/src/dynamo/vllm/omni/`). First-class means hardening that contract, not inventing one.

**Separation of concerns — the hard line:**

| Dynamo owns (FastVideo must not build) | FastVideo engine owns (Dynamo does not reach into) |
|---|---|
| Service discovery + model registration (etcd/NATS, K8s) | model execution, loops, policies, samplers |
| Cross-node request routing (KV/affinity/load-aware) | StepScheduler: step interleaving, AR batching, in-pool placement |
| SLA Planner: replica autoscaling (per-pool, prefill/decode ratios, K8s CRDs) | caches (paged KV, chunk KV, feature, weight fleet) and their in-node lifecycle |
| Fleet health, canaries, request migration, graceful-drain orchestration | consistency contract (C0–C2), parity gating, plugins |
| OpenAI-compatible fleet frontend + preprocessing | the per-pool API surface and `OmniEvent` streaming semantics |
| KVBM multi-tier token-KV offload; ModelExpress weight streaming (cold starts) | which weights/caches are worth offloading, and when |

**The contract (what "first-class backend" concretely requires of us — Phase 2/3 work items):**

1. **Worker surface** (Phase 2): the Dynamo worker wraps `AsyncEngine` instead of a locked `VideoGenerator` —
   registration with `ModelType.Videos | Images` (+ `Chat` once omni lands), `worker_type` mapping our roles onto
   theirs (Aggregated; Prefill/Decode for disagg), streaming handler (our `OmniEvent` chunks → their annotated
   stream), health-check endpoint, graceful drain (finish in-flight loops, release pools) wired to
   `DistributedRuntime`'s signal handling.
2. **Metrics for routing + Planner** (Phase 2): queue depth, pool utilization, and — important for video —
   **per-request cost estimates** (a 5 s 480p distilled job and a 30 s 1080p job differ by orders of magnitude;
   the Planner's SLA math needs the cost model we can derive from `PipelineSpec` × resolution × steps; LLM
   backends get this implicitly from token counts, we must publish it). This is the same cost model the
   StepScheduler uses as its internal budget currency (§6.3.1) — built once, consumed in-pool and at fleet level.
3. **Affinity events** (Phase 3, with the weight-fleet cache): Dynamo's KV-aware routing rests on engines
   publishing cache events (`lib/kv-router/src/protocols.rs:627-646`:
   `KvCacheEventData::Stored{parent_hash, blocks[{block_hash, tokens_hash}]}` / `Removed` / `Cleared`) into a
   radix index. The index is hash-generic, but today's router derives keys from *token prefixes* — our two
   affinity signals (**checkpoint residency** for the workflow cloud §9.4, **session residency** for Dreamverse
   §9.1) need either a custom router policy (Dynamo exposes the router entrypoint for this) or an upstream
   extension — formalized as **ask A1** (§6.3.6).
4. **Disagg alignment** (Phase 3): our encode/denoise/decode role pools map onto Dynamo's prefill/decode
   coordination — the worker returns `disaggregated_params` in its terminal chunk and transfers tensors over
   NIXL, which is already in our connector menu (§6.3.4); cross-node disagg then comes from contract
   compatibility, not new infrastructure.

**What this buys, concretely:** multi-node serving without writing a router; SLA autoscaling that makes the
$5-subscription workflow cloud's unit economics work (engine-level multiplexing × fleet-level bin-packing);
ModelExpress weight streaming for fast cold-starts — which composes directly with the weight-fleet cache (§9.4)
when checkpoints migrate across the fleet; KVBM's GPU→CPU→SSD tiering pattern as the template for offloading
idle Dreamverse session caches (sessions are idle-heavy between user directions); and production ops (operator,
observability, fault tolerance) we never staff. Dreamverse's planned local-first `controller/` (Runpod/Modal
provisioning) remains the single-box story; at fleet scale it delegates to Dynamo rather than growing its own
orchestration.

#### 6.3.6 Asks of Dynamo (the partnership is two-way)

We have a direct working relationship with the Dynamo team, so Dynamo is **not treated as frozen**: where the
ecosystem design improves by changing Dynamo rather than working around it, we say so. Each ask below states the
gap, the proposed upstream change, who benefits beyond FastVideo (these are ecosystem gaps — vllm-omni,
sglang-diffusion, and every media/RL backend hit the same walls), and our fallback if it doesn't land. Roughly in
priority order:

- **A1 — Generic affinity routing (key spaces beyond token prefixes).** *Gap:* the KV router's radix index is
  hash-generic, but keys are derived from request token prefixes — useless for our affinity signals. *Ask:*
  generalize the event protocol (`lib/kv-router/src/protocols.rs`) to namespaced affinity events —
  `{key_space: "checkpoint" | "session" | "lora_set" | "weight_version" | "kv_prefix", key_hash, resident,
  capacity_hint}` — with router policies pluggable per key space. *Beneficiaries:* any diffusion/workflow backend
  (checkpoint/LoRA residency — the §9.4 fleet cache), any session workload (world models, omni chat, Dreamverse),
  and LLM serving itself (LoRA-set affinity). *Fallback:* custom router policy via Dynamo's router entrypoint;
  worst case, least-loaded routing + engine-side redirects.
- **A2 — Heterogeneous request cost interface.** *Gap:* router load-accounting and Planner profiles infer cost
  from token counts; a 5 s 480p distilled clip and a 30 s 1080p clip differ by orders of magnitude with identical
  "prompt lengths." *Ask:* a standard per-request cost annotation (predicted GPU-seconds / memory class) that
  engines publish and the router + Planner consume, with Planner profiling keyed on engine-declared cost features
  (model, resolution, steps) instead of tokens. *Beneficiaries:* every media backend Dynamo fronts. *Fallback:*
  encode cost classes as separate registered model names (crude bucketing).
- **A3 — Streaming media through the fleet frontend.** *Gap:* `ModelType.Videos` today returns one base64 MP4 blob
  (the `examples/diffusers/worker.py` pattern); Dreamverse-class UX needs chunked fMP4/audio segments and progress
  events end-to-end. *Ask:* first-class chunked-binary streaming (segments + progress + keep-alive) through the
  frontend for media model types, mapping 1:1 onto our `OmniEvent` channel. *Fallback:* bypass the frontend for
  streams (direct WebSocket to pools) and use Dynamo for routing/placement only — workable but forfeits unified
  auth/observability.
- **A4 — Role-graph disaggregation (N roles, not two).** *Gap:* disagg coordination is hardwired to the
  prefill→decode pair; our role pools are encode→denoise→decode (and omni graphs fan out further). The
  `disaggregated_params` payload is already opaque, so the protocol nearly exists. *Ask:* generalize PrefillRouter
  into a role router over engine-declared role graphs with NIXL hand-offs per edge. *Beneficiaries:* vllm-omni and
  sglang-omni each hand-built exactly this orchestration (§5) — Dynamo could own the pattern once for the whole
  ecosystem. *Fallback:* keep multi-role inside one Dynamo-visible worker (our engine-internal pools, §6.3.4) and
  expose only aggregated workers to the fleet.
- **A5 — The RL weight plane: versioned update broadcast + staleness-aware routing.** *Gap:* ModelExpress streams
  weights for cold starts; RL needs *hot* incremental updates — trainer publishes version N+1, the rollout fleet
  converges, and the router knows who's on what. *Ask:* (a) ModelExpress incremental in-place weight-update
  broadcast to a live worker set with per-worker `weight_version` tracking; (b) version/staleness-aware routing
  (route rollouts to ≥version-K workers; drain stale ones) — A1's `weight_version` key space; (c) a Planner
  objective plugin for **train↔rollout GPU elasticity** (scale the rollout pool against trajectory-queue depth and
  the staleness bound rather than serving SLAs). *Beneficiaries:* every RL stack that fronts rollout with Dynamo —
  this is miles' weight-transport + `weight_versions` machinery (§8.6) offered as fleet infrastructure.
  *Fallback:* our own transports (§8.6) with Dynamo unaware of versions; works colocated, weakens fleet-scale
  async RL.
- **A6 — KVBM generalization to engine-defined cache objects.** *Gap:* KVBM tiers token-KV blocks; our expensive
  idle state is latent session caches and chunk-KV (Dreamverse sessions between directions, world-model games).
  *Ask:* a generic cache-object API (register/pin/unpin, tier hints, eviction callbacks) so engine caches ride
  KVBM's GPU→CPU→SSD machinery under fleet policy. *Fallback:* engine-side offload within the CacheManager
  (single-node only).
- **A7 — Session lifecycle as a frontend/router primitive.** *Gap:* no session concept at the fleet layer; ours
  (Dreamverse scenes, interactive world models, multi-turn omni) need create/attach/expire semantics pinned to
  session-resident workers. *Ask:* session IDs as a first-class routing primitive (pairs with A1's `session` key
  space + A6 for idle offload). *Fallback:* sticky routing by consistent hash at our API gateway (sglang-omni's
  pattern), losing fleet-level visibility.

Sequencing with our phases: A2 + A3 matter by Phase 2 (Dynamo-grade worker + Dreamverse streaming); A1 + A4 by
Phase 3 (fleet cache + disagg); A5–A7 by Phase 4 (RL hardening + session serving). Each ask is independently
useful to Dynamo's existing backends — that's the test we apply before asking.

### 6.4 Extension system: hooks and plugins (cross-cutting)

Requirement: drop-in **optimization plugins** — [cache-dit](https://github.com/vipshop/cache-dit)
(DBCache, FBCache, TaylorSeer calibration) and successors — and drop-in **instrumentation** for debugging, profiling,
and numerical-precision alignment, without forking loop or model code per technique. Seeds already in-tree:
`fastvideo/hooks/` (`ModuleHookManager`/`ForwardHook`, plus `activation_trace.py` — zero-overhead-when-off,
regex-selected module taps with a `trace_step()` step-index context, built for port-parity debugging), and a
cautionary tale: `enable_teacache` is plumbed through the entire API surface (`pipeline_batch_info.py:182`,
OpenAI protocol, sampling params) yet has **no consumer in the shared denoise loop** — bolting optimizations into a
monolithic loop is exactly what didn't ship. The extension system makes such techniques additive.

Two tiers with different contracts:

**Tier 1 — Observers (read-only).** An event bus with stable, named hook points:

| Plane | Hook points |
|---|---|
| Request | `request_start/end`, `artifact_emitted`, `schedule_decision` |
| Pipeline | `node_start/end`, `loop_init/finalize`, `step_start/end` (step idx, timestep, state ref) |
| Execution | `model_forward pre/post`, module-level forward taps by name regex (existing `ModuleHookManager`), cache events (alloc/hit/evict), connector transfers |

Contract: observers cannot mutate state; the chain is assembled at pipeline build, so an unused hook point is
*literally absent* from the hot path (the activation-trace "zero overhead when off" rule, generalized); each
observer declares `needs_eager`, and the runtime drops compile/cudagraph capture only for the scopes it watches,
logging the cost. Built-ins shipped with the runtime:

- **ActivationTrace** — the existing module, promoted from env-var global to a per-request-scoped observer (env-var
  UX preserved).
- **ParityAligner** — formalizes the Cosmos3 oracle methodology (N3) as infrastructure: *record mode* dumps named
  taps per step/block from a reference run (official framework, or pre-change FastVideo); *compare mode* replays a
  request with fixed seeds and reports the first divergence beyond per-tap tolerances. This is the engine behind
  the Phase-1 "old loop vs new loop, bit-identical" gate, and the standing tool for every future port and precision
  change (FP8/FP4 work included). The training side already runs the coarse version of this pattern in CI: PR #1396
  pins per-method, **per-GPU-model** layer-0 grad-norm references at 10% tolerance
  (`fastvideo/tests/train/methods/grad_norm_regression.py` — the causal-DFSFT refs differ ~9% between L40S and
  GB200 while the finetune refs differ only ~0.1%: cross-device reduction order is a real *and method-dependent*
  consistency variable, §8.5);
  ParityAligner is that idea taken to per-tap, per-step resolution.
- **Profiler** — per node/step/block wall+CUDA timings, NVTX ranges, on-demand `torch.profiler` windows, memory
  watermarks; exported through engine metrics.
- **NaNWatch** — first-NaN/Inf localization (module × step) for debugging.

**Tier 2 — Interceptors (compute-altering).** Explicit extension points — never monkeypatching (§11.10):

```python
class StepInterceptor(Protocol):      # step-skip / cached-prediction schemes
    def before_step(self, loop: LoopState) -> StepOverride | None: ...  # may supply a predicted output, skipping the forward
    def after_step(self, loop: LoopState, out) -> None: ...             # update residual/calibration state

class BlockInterceptor(Protocol):     # cache-dit class: FBCache, DBCache, TaylorSeer
    def plan(self, loop: LoopState) -> BlockPlan: ...                   # which transformer blocks run vs skip this forward
    def on_skipped_block(self, idx, h) -> torch.Tensor: ...             # substitute cached / Taylor-extrapolated residual
```

- Block interception needs one model-side convention: in-tree DiTs execute their block stack through a
  `run_blocks(self.blocks, h, ctx)` helper — a plain loop when nothing is bound (compile-friendly) — instead of
  inlined `for` loops. Honest cost accounting: this is a **per-family refactor, not a rename** — today's DiTs
  inline divergent block loops with different arguments and gradient-checkpointing paths (`wanvideo.py:712`,
  `flux_2.py:1057`, `ltx2.py:2648`, `causal_wanvideo.py:560`). Adoption is per-family with ParityAligner gates
  (Phase 1 scope = the Phase-1 families, Wan + Flux2); a family that hasn't converted simply **rejects block-level
  plugins at build time** via capability negotiation — step-level interceptors still work everywhere.
- **State scoping is the correctness fix**: interceptor state lives in `LoopState.plugin_state[plugin_id]`, keyed
  per request **and per CFG branch**. Reference implementations (cache-dit on diffusers, TeaCache forks) keep
  residual state in module-level globals, which silently corrupts under concurrent requests — here
  concurrency-safety is structural, which is what makes these optimizations usable in the serving engine at all.
- Build-time composition: ordered chain; conflicting interceptors (two block-skippers) rejected pre-flight — the
  same validation stance as the parallelism spec; per-model compatibility declared in the `PipelineSpec` (a 4-step
  distilled model rejects step-skip caches rather than producing garbage).
- Graph contract: interceptors declare `graph_safe`; graph-safe ones express skip decisions as captured tensor ops
  or bucketed graph variants; others force eager for their scope only.

**cache-dit is the reference integration target.** It is the library sglang's multimodal serving already uses
(`multimodal_gen/runtime/cache/cache_dit_integration.py`: `enable_cache_on_transformer` /
`enable_cache_on_dual_transformer` built on cache-dit's `BlockAdapter` + `DBCacheConfig` /
`TaylorSeerCalibratorConfig`, configured by a `cache_dit_config` server arg). Two scars in that integration are
instructive: sglang must **monkeypatch `CachedContextManager.similarity`** to behave correctly on
sequence-parallel-sharded tensors, and must manually call `refresh_context_on_transformer` per request. Both needs
are structural in our interceptor contract — the `BlockInterceptor` adapter sees the parallel groups (so residual
comparison is group-aware by construction) and its state is `LoopState`-scoped (so per-request/per-branch lifecycle
is automatic, no manual refresh). The Wan2.2 dual-expert case (sglang's `dual_transformer` path) composes with our
`ExpertRouting` policy: one cache context per expert. We consume cache-dit as a dependency — its configs,
calibrators, and skip math — behind our extension point, keeping algorithm parity with the sglang fork.

**Packaging — and the trust boundary.** A `fastvideo.plugins` entry-point group + registry: third-party packages
(the cache-dit adapter, a vendor profiler) register without forking. **Enablement is DeployConfig-scoped only** —
entry points resolve at deploy load, never per request. The reasons are load-bearing in a multi-tenant engine:
plugins execute arbitrary code, and since the OpenAI protocol is *generated from the request schema*, a
per-request `plugins=[...]` field would derive third-party-code selection straight into the public API. Requests
therefore only *parameterize pre-enabled* plugins through per-plugin validated schemas
(`diffusion.plugin_params={"cache_dit": {"Fn": 8, "Bn": 8}}`), with two guards: exact-mode requests reject
`distribution_altering` parameterization outright (the §9.4 trust claim), and per-request plugin overhead —
including a `needs_eager` observer de-capturing scopes shared with co-scheduled tenants — is attributed in that
request's cost metrics, so noisy neighbors are visible and chargeable. Hook-point names are a versioned public
surface.

In RFC #4084's vocabulary: observers and interceptors are the L1/L2 intervention levels applied to *computation
shape* rather than parallelism (step/loop ≈ L1, block/layer ≈ L2), and L3 remains the attention/kernel backend
registry. One taxonomy, three uses — parallelism, optimization, instrumentation.

### 6.5 Entrypoints and compatibility

- `VideoGenerator` / `StreamingVideoGenerator` / CLI: unchanged signatures, now building `OmniRequest`s.
- New `AsyncEngine` (`generate(request) -> OmniOutput | AsyncIterator[OmniEvent]`).
- OpenAI-compatible server extends the existing `fastvideo/entrypoints/openai/`: `/v1/videos/generations`,
  `/v1/images/generations`, `/v1/audio/speech`, `/v1/chat/completions` (omni: text+media in, text+media out —
  lands with Phase 4),
  SSE/WebSocket streaming. API-shape parity with `multimodal_gen` is an explicit goal so a future
  upstream/downstream story (§11.1) stays open.
- **Workflow frontend** (Phase 3): the ComfyUI-workflow compiler (§9.4) is a peer entrypoint — workflow JSON parses
  into `PipelineSpec` + `OmniRequest`s and enters the same admission path as every other client.
- **Legacy adapter**: `LegacyPipelineNode` wraps any existing `ComposedPipelineBase` as a single opaque graph node
  (request → ForwardBatch shim → stages → artifacts). Every unmigrated model runs day one.

### 6.6 Model loading, configs, and the typed API

The user is mid-migration to a typed API (`fastvideo/api/`) and has flagged it — and config specification and model
loading generally — as open for redesign. Today there are **seven overlapping config surfaces**: arch configs
(`fastvideo/configs/models/`, with `param_names_mapping` — good, unchanged); pipeline configs
(`fastvideo/configs/pipelines/`); `FastVideoArgs` (81 fields) and its legacy subclass `TrainingArgs` (90 own + 81
inherited = 171); the
new-trainer YAML `TrainingConfig` (whose `distributed:` block re-declares the same parallelism fields); legacy
`SamplingParam` (75 fields, now at `fastvideo/api/sampling_param.py`); and the in-flight typed schema
(`fastvideo/api/schema.py`: `GenerationRequest`, `GeneratorConfig`, `ServerConfig`, `ParallelismConfig`,
`OffloadConfig`, `CompileConfig`, `InputConfig`, `OutputConfig`, …). The seams are load-bearing code:
`fastvideo/api/compat.py` is **651 lines of bidirectional translation** (`request_to_sampling_param`,
`generator_config_to_fastvideo_args`, `legacy_generate_call_to_request`, dotted-path explicit-field tracking via
`request_metadata.py`), and per-model knobs leak into the universal surface as side files
(`api/matrixgame2.py`, `api/matrixgame3.py`).

The in-flight work points the right way — typed request, generator config, and streaming event types
(`api/results.py`: `VideoProgressEvent`/`VideoPartialEvent`/`VideoFinalEvent` prefigure `OmniEvent`). Its problem
is that it sits *beside* the legacy surfaces, translating, rather than replacing them. Target: **four layers, one
owner each, everything else derived**:

| Layer | Owns | Absorbs / retires |
|---|---|---|
| `ArchConfig` | model structure + `param_names_mapping` | (unchanged) |
| `ModelSpec` + `PipelineSpec` (§6.2.4) | components and weights, graph, policies, samplers | pipeline configs' behavior knobs; registry name-detectors; `model_index.json` implicitness |
| `DeployConfig` | placement, stacked parallelism axes (§6.3.4), memory/offload, compile, plugins | the runtime thirds of `FastVideoArgs` and `GeneratorConfig`; `TrainingConfig.distributed` |
| `OmniRequest` (§6.1) | everything per-call, incl. per-model `ModelOptions` blocks | `GenerationRequest` (evolves into it), `SamplingParam`, the request half of `ForwardBatch` |

Derived views are *generated*, never hand-maintained: the OpenAI protocol models, CLI flags, and presets
(`api/presets.py` becomes per-model default `OmniRequest` fragments) all derive from the request schema; per-model
options are registered typed blocks rather than universal-schema fields (the `api/matrixgame2.py` pattern,
formalized). `compat.py` is migration scaffolding, but its retirement must be honest about who depends on it:
`VideoGenerator` itself currently routes through `generator_config_to_fastvideo_args` /
`legacy_generate_call_to_request` / `request_to_sampling_param` (`entrypoints/video_generator.py:31`), and the doc
promises those public signatures stay. So the policy is: **Phase 0 freezes the file** (no new entries, ever);
each phase deletes the translations its migrations obsolete (the `SamplingParam`/`ForwardBatch` bridges die as
families migrate); what legitimately remains until Phase 5 is the thin, generated legacy-signature shim — and at
Phase 5 the file reaches zero lines. Monotonic shrink per phase is the tracked metric, not a Phase-1 cliff.

**Model loading.** Replace implicit `model_index.json` + name-detector resolution with an explicit **`ModelSpec`
manifest** per checkpoint family: declared components with `required_for`/`optional_for` task sets (the Cosmos3
lazy-sound-VAE fix, P8), accepted weight layouts (diffusers-style, native, quantized variants), dtype/precision
policy, and the default `PipelineSpec` binding. Component loaders stay as they are (`TransformerLoader`,
`VAELoader`, … unchanged); resolution becomes manifest-first, with today's detectors as fallback for unmanifested
HF checkpoints. The same manifest is what `train/` `ModelBase.init_from` consumes — one loading path for inference,
training roles (student/teacher/critic), and RL rollout (§8).

**Hub interchange (diffusers Modular).** `ModelSpec` component entries serialize compatibly with diffusers'
`ComponentSpec` / `modular_model_index.json` (the modular-repo hub format), and guidance configs map onto their
Guider classes — so a modular-diffusers port and a FastVideo `PipelineSpec` share vocabulary, models published
either way load both ways, and HF remains the distribution rail for `(recipe, runtime)` artifacts (see the
artifact-registry known gap, §12). The same shape is appearing in vLLM core (RFC #42770): an HF-decoupled
`ModelConfig` plus a `Model` interface where the model owns its state and input preparation — the third
independent arrival at manifest-first loading and model-owned state, after diffusers Modular and this design.

**Schema stability tiers.** By Phase 3, at least four serialized surfaces have external consumers: `ModelSpec`
manifests (hub interchange, format co-owned with diffusers), compiled-workflow `PipelineSpec`s (content-hash-keyed
in the weight-fleet cache — a schema change silently changes hashes and invalidates fleet affinity), the
`OmniEvent` stream (Dreamverse's frontend; proposed as Dynamo ask A3's wire format), and per-node parameter
schemas. Every one carries a `schema_version`; each is **experimental** until the phase its first external
consumer ships (`OmniEvent`: Phase 2; `ModelSpec` + workflow specs: Phase 3), then
**stable-with-a-deprecation-window** (one minor release); workflow-spec hashes include the schema version so
affinity invalidation is explicit, never silent. The migration plan gates *behavior* at every phase — this is the
corresponding gate for *interfaces*.

---

## 7. Worked examples

**(a) Wan 2.2 T2V (today's bread and butter).** Graph: validate → text-encode → `DenoiseLoop(transformer,
cfg=AdaptiveGateCFG, routing=BoundaryTimestepRouting, attn=VSAProvider)` → vae-decode. Single pool, SP as today
(CFG-parallel available once the axis lands). Behavior and perf identical to today; the boundary-switch and VSA
hacks become declared policies.

**(b) Wan-Causal streaming T2V.** `ChunkRollout` outer loop holding a `ChunkKVPool` handle; inner
`DenoiseLoop(chunk)`; each finished chunk → incremental VAE decode → `emit` → SSE frames. The same graph with a
session-scoped KV handle serves MatrixGame-style interactive world models (actions arrive as streaming-input
updates, vllm-omni-style resumable requests).

**(c) LTX-2 two-stage A/V.** validate → text-encode → `DenoiseLoop(joint video+audio latents)` — likely a
**custom step body** under the §6.2.3 escape hatch, since LTX2's 1–4 guidance passes alter the network via forward
kwargs, a braid the slot skeleton doesn't own cleanly; per-modality CFG scales and STG remain policy-shaped if the
factoring proves out → upsample → `DenoiseLoop(refine, lora-swapped)` → fan-out {vae-decode video,
audio-vae+vocoder} → assemble (video+audio artifacts). Today's 11 hand-ordered stages become a graph with one fork
either way — the graph and loop contract don't care which way the step body lands.

**(d) Qwen-Omni-style thinker→talker→vocoder (separable omni).** Three nodes: `ARDecodeLoop(thinker,
emit=hidden_states)` →(stream)→ `ARDecodeLoop(talker, emit=codec_chunks)` →(stream)→ `Vocoder`. Single pool or
per-node placement; `async_chunk`-style streaming via `StepResult.emit` + connector readiness. Optionally the
thinker delegates to an external engine node.

**(e) Cosmos3 full omni (the forcing function).** The §6.2.4 spec: optional `ARDecodeLoop(reasoner)` with paged KV
(replacing re-prefill-per-token) → pack → joint `DenoiseLoop(vision[+action][+sound])` → fan-out decoders →
assemble {video, audio, action, text}. One resident MoT transformer bound to both loops; REASON task short-circuits
to text-only; world-model rollout = `ChunkRollout` with temporal-causal masks and clean-latent chunk KV.

**(f) Flux2 / Stable Audio serving.** Same `DenoiseLoop` with image/audio latents; StepScheduler batches
shape-bucketed requests; embedded-guidance and precision quirks are policies. FastVideo becomes a credible image/audio
server for free.

**(g) DiffusionNFT RL (landed: PR #1450).** The landed method is already *library-mode shaped* — rollouts run
in-process under the trainer's own kernels — but against a vendored loop: `DiffusionSampler` walks the full
25-step ODE (or re-noising `sde_reflow`) schedule over the frozen **old-policy** copy, conditional-only,
`attn_kind="dense"` pinned, and the objective is **likelihood-free**: only final clean latents survive the
rollout; training re-noises them at shuffled schedule timesteps and contrasts the student against an *implicit
negative policy* in prediction space, so per-step log-probs exist nowhere in the method. Under this design the
same method drives the shared `DenoiseLoop` with a sampler policy and `OutputSpec(return_final_latents=True)` —
deleting `rl/common/sampling.py` — and gains what the vendored loop cannot offer: CFG policies, distilled few-step
samplers, caches, and batched scheduling for the rollout phase (today's rollouts run with zero serving-grade
optimizations). Engine-client mode: seeded rollout `OmniRequest`s fan out to a rollout pool running the same loop,
and the trainer pushes `update_weights` between iterations — note the weights shipped are the *old policy*, a
decay-blended copy, a concrete `WeightSyncPlan` case (§8.6) — with sleep/wake reclaiming rollout memory during the
gradient phase. Consistency per §8.5's own rules: NFT is the *easy* rung — likelihood-free means no ratio exists
for drift to poison, and the landed discipline (pinned attention kind, pinned autocast dtype, seeded generators)
is C1 by construction; **the full ladder is for the GRPO-class methods that come next** — C1-pinned rollout (cache
acceleration off) bounds per-step log-prob drift to ε, trajectory *identity* is C2's conditioned replay guarantee,
not C1's; alternatively C0 with cache-dit on buys throughput and pays for it with TIS correction — never both at
once.

---

## 8. Training, post-training, and RL on the same substrate

FastVideo's differentiation — and its complexity tax — is that training and post-training live in the same repo as
inference, with RL now **landed** (PR [#1450](https://github.com/hao-ai-lab/FastVideo/pull/1450), DiffusionNFT for
Wan — no longer a theoretical reference: a 3,500-line PR on main spanning the 1,075-line method, a vendored
sampler, reward scorers, trainer hooks, tests, and three `.agents/skills` codifying its extension contract). The
question: does a fully unified design make sense? **Answer: unify the substrate, not the stacks** — and
for video models this is less a choice than a recognition: unlike LLMs, diffusion's inference optimizations are
themselves post-training artifacts (§8.3), so the loop inversion of §6.2.2 is the missing shared piece. One honest
qualification, learned from studying two production RL frameworks (`~/verl-omni` for diffusion RL, `~/miles` for LM RL): the industry's pain is not really
"many repos." It is **two runtimes with different kernels** — Megatron/FSDP-class training optimizations on one
side, vLLM/sglang-class inference optimizations on the other — and an entire mismatch-correction industry has grown
around that split (§8.4). That is simultaneously the strongest argument for what we are building and the sharpest
warning about it: one repo does not buy numerical consistency. One model definition + one kernel set + an
**enforced, measured consistency contract** does (§8.5).

### 8.1 What exists today (and what is duplicated)

Three consumers of the generative computation, with partial sharing:

| Layer | Shared today | Duplicated today |
|---|---|---|
| Models, component loaders + FSDP load, arch configs (`param_names_mapping`), schedulers, `fastvideo/distributed/`, attention backends, offload hooks | ✓ single-source across inference, `train/`, legacy `training/` | — |
| Validation generation | new stack's `ValidationCallback` instantiates the real inference pipeline and calls `forward()` | legacy stack rolls its own; the landed RL method adds a third path — `TrainingMethod.on_validation_begin()` + `rl/common/validation.py` re-implement prompt sharding, seeded sampling, and sample logging precisely because (per the hook's own docstring) these are "methods that intentionally avoid inference pipelines" |
| **Denoise/sampling loop** | — | **4 copies**: `pipelines/stages/denoising.py` (inference); `DMD2Method._student_rollout()` / `SelfForcingMethod._student_rollout_streaming()` (`train/methods/distribution_matching/`); inlined per-pipeline in legacy `training/*_distillation_pipeline.py` (~1200-line monoliths); `DiffusionSampler` in `train/methods/rl/common/sampling.py` (RL, landed with PR #1450) |
| CFG / uncond handling | — | 3 copies: `stages/denoising.py` (cond/uncond branching + CFG gating; `conditioning.py`'s forward is a no-op that only validates flags) vs `ModelBase.predict_noise(cfg_uncond=…)` vs legacy inline |
| Preprocessing | `pipelines/preprocess/*` already subclass `ComposedPipelineBase` and reuse `TextEncodingStage` → parquet latents/embeddings | trajectory extraction (`preprocess_pipeline_ode_trajectory.py`) is a separate path from `return_trajectory_latents` |

The fourth copy is no longer a prediction — PR #1450 landed it. `train/methods/rl/common/sampling.py` vendors
`DiffusionSampler`: an ODE walk delegating per-step math to the shared `FlowMatchEulerDiscreteScheduler.step`, plus
a re-noising `sde_reflow` mode that is a near line-for-line re-derivation of DMD2's rollout loop
(`dmd2.py:484–520`). Its own docstring states the reason: it *"intentionally does not call FastVideo's full
inference pipelines"* because that would bind the method to model-family pipeline classes like `WanDMDPipeline` —
and the accompanying skill (`.agents/skills/rlhf-training-abstractions/SKILL.md`) elevates this to standing policy
("do not bind RL methods to inference pipeline classes"), with a test asserting the RL YAML never references
`WanDMDPipeline`. The authors are right to refuse that dependency, and that is precisely the indictment: today's
only consumable units are family-bound pipeline classes, so every new post-training method must choose between a
wrong dependency and a private loop. The `add-rl-method` skill now mandates the vendored sampler for future RL
methods — which contains the duplication *within* `train/methods/rl/` but cements it outside the substrate, where
no inference optimization, CFG policy, or scheduler improvement will ever reach it.

Concretely the "fourth copy" is already **two** vendored RL/distill loops, not one: `DiffusionSampler`
(`rl/common/sampling.py:92-99` docstring, `:128-147` hand-rolled `for`-loop, `conditional=True`, `attn_kind=dense`)
*and* DMD2's `_student_rollout` (`dmd2.py:430-550`, `attn_kind=vsa`) — a second, independently vendored loop. What
each of these vendored loops **forgoes** by living outside the substrate is itemizable: (1) cross-group /
cross-request batching; (2) content-hash embedding/feature-cache reuse; (3) step scheduling and interleaving;
(4) cost-aware admission; (5) component-granular sleep/wake (it holds student + old + reference resident
simultaneously); (6) cache-dit / interceptor skip acceleration (forced dense, full 25-step ODE); (7) distilled
few-step sampler selection (a 4-step card collapses 25→4). A `DenoiseLoop` over substrate
step bodies is the third option this design exists to provide — and, per §8.4, it is the moat: rollout forward
*is* serve forward + capture, so every serving optimization is automatically a rollout optimization.

### 8.2 The decision: shared substrate, layered dependencies

Full unification — one framework that both trains and serves — is the wrong target: trainers own
optimizers/FSDP/grad-accumulation/checkpointing; the engine owns scheduling/caches/serving. What they genuinely
share is the **generative computation** and the **model substrate**:

```
substrate (shared):  models/ + loaders/manifests + configs + schedulers + distributed/ + attention/
                     + PipelineSpec graphs, Stage/LoopStage step bodies, sampler & CFG policies, PackedSeq
engine  (inference): StepScheduler, queues, caches, serving, placement   — depends on substrate
train   (training):  Trainer, methods, callbacks, data                   — depends on substrate (+ engine as client)
```

Rules and consequences:

- **`engine` never imports `train`** (the reverse is allowed). A substrate change is a change to both worlds — and
  is gated as such (§12, risk 5).
- **Distillation rollouts call the shared loop — with one honest exception.** DMD2/KD methods drive
  `DenoiseLoop.step` directly (grad-enabled where needed — step bodies are plain tensor programs, so autograd
  composes) instead of re-implementing timestep iteration. **Self-forcing does not fit the inference step
  contract** and is not forced into it: its rollout samples per-block exit indices broadcast across ranks, runs
  no-grad steps to the exit, exactly *one* grad-enabled forward, then a separate no-grad `store_kv=True`
  context-caching pass — grad-window flags and per-step cache-write control that would be training-only surface
  in substrate code. It ships as a **custom training `LoopStage` in `train/`** (the §6.2.3 escape hatch),
  consuming substrate pieces — model forwards, samplers, and `ChunkKVPool` in its declared **training mode**
  (no mid-rollout recycling; grad-aware index snapshots so activation-checkpoint recompute doesn't double-advance
  the cache, per `wan_causal.py:119-120, 405-431`). The dedup arithmetic stated honestly: shared step bodies where
  they fit; custom loops stay custom but consume the substrate instead of vendoring their own sampling stack.
  The trainer already concedes this shape in landed code: PR #1450 added `TrainingMethod.manages_optimization()` /
  `managed_train_step(data_stream, …)` so DiffusionNFT's sample→score→inner-train outer epoch could own its own
  optimizer cadence — method-owned loops are a supported trainer contract today, not a workaround this design
  must invent.
- **Trajectory capture becomes one feature, not four.** `return_trajectory_latents`, the ODE-trajectory
  preprocessing pipeline, RL rollout recording (the landed `SamplingResult` in `rl/common/sampling.py` —
  latents + timesteps + sigmas — is the third ad-hoc capture path), and future log-prob capture for GRPO-class
  methods all become `OutputSpec` options on the loop.
- **Preprocessing pipelines are loop-less graphs** of one-shot encode stages emitting parquet artifacts — they
  already use the pipeline machinery, so this is a rename, not a rewrite.
- **Observers carry over for free**: ParityAligner / NaNWatch / Profiler attach to training rollouts identically —
  first-class tooling for divergence hunts and precision (FP8/FP4, QAT) validation during training.
- The legacy `training/` stack stays frozen (per repo policy) and is unaffected until its models migrate to
  `train/` on their own schedule; nothing here forces that migration.

### 8.3 Why diffusion is different: inference optimization *is* post-training

In the LLM world, inference optimization is mostly post-hoc and training-free: a lab ships frozen weights, and the
serving stack optimizes *around* them — PTQ quantization, KV-cache management, continuous batching, speculative
decoding, kernels. The artifact crossing the trainer→engine boundary is a checkpoint. That is why the two-runtime
split is tolerable there: the inference frontier can advance without touching training.

Video generation does not work that way. **The deployable model is itself a post-training artifact**, and nearly
every inference capability FastVideo ships is really a **(recipe, runtime) pair**:

| Inference capability | Training-side half | Runtime-side half |
|---|---|---|
| Few-step generation — the difference between unusable and usable | **Step distillation** (DMD2, consistency/rCM) — an active research frontier; recipes change quarterly | distilled samplers / timestep policies |
| Streaming + world models (AR video) | step distillation **plus a self-forcing variant on top** — the causal student *is created by training* | `ChunkKVPool`, causal rollout, streaming |
| Low-precision inference | **QAT** (our NVFP4 Attn-QAT line, upstreaming now) — PTQ alone doesn't hold quality; training must see deployment numerics | FP4/FP8 kernels + `PrecisionPolicy` |
| Sparse attention | **VSA** — trained sparsity (tile selection learned during training) | VSA kernel backend |
| Reward alignment | RL — DiffusionNFT landed (#1450, likelihood-free: needs only final-sample capture); GRPO-class next | samplers (ODE/SDE), final-latent + trajectory/log-prob capture |

Two structural consequences:

1. **The training loop embeds the inference loop — and that collocation is the central, durable moat.** Distillation
   runs deployment-grade rollouts *inside* training: DMD needs student rollouts; self-forcing is literally "train
   against your own KV-cached causal rollout" — the runtime feature must exist before the model class does; QAT's
   fake-quant forward must match the deployed quant kernels. There is no valid sequencing of "build the inference
   engine first, training later" — each is a dependency of the other, which is exactly why the substrate (§8.2) has
   to come first, and why the denoise loop keeps getting copied (§8.1): every post-training method needs an
   inference loop inside it, and absent a shared one, each grows its own. The payoff of collocating
   inference + post-training/RL on one substrate: **rollout forward is serve forward + capture** — same loop, same
   caches, same batcher, same numerics — so every serving optimization is automatically a rollout optimization and
   there is exactly **one numerics surface** (no rollout-vs-train kernel gap to patch, §8.4).

   **RL rollout is a strictly *better* cross-request batching case than open-world serving** — the killer point. A
   GRPO/NFT group is K identical-config samples of one prompt: `num_video_per_prompt:24`,
   `sample_train_batch_size:6`, `num_batches_per_epoch:48`, `num_steps:25`, single-frame 448×832, 4 GPUs
   (`diffusion_nft_pick_clip.yaml:33-37,52-55,72-91`; `distributed_k_repeat_indices` with `repeats_per_prompt=24`,
   `diffusion_nft.py:331-338`). Same shape, same schedule, same CFG branch, K=24-wide → **zero bucketing needed**
   (serving must bucket heterogeneous resolutions/steps/CFG). The K samples also share *one* prompt embedding, so a
   content-hash feature cache encodes the text **once** and reuses it across all 24 — a **24× text-encode
   reduction** the vendored loop cannot express (it carries the embedding per-sample, `diffusion_nft.py:308-309`).
   Per GPU per outer epoch = 6 prompts × 48 batches = 288 prompt-slots, each a 24-wide homogeneous denoise batch:
   near-ideal co-batchable work the vendored Python loop runs **one sample at a time**.

2. **Train/serve skew generalizes the RL mismatch.** Any sampling-in-the-loop post-training optimizes the student
   *under the rollout implementation used during training*. Serve it under different kernels/sampler/caching and
   quality silently degrades — the diffusion analog of RL's off-policy bias, except it applies to essentially all
   post-training and surfaces as quality regression rather than reward collapse (slower to notice, harder to
   attribute). The consistency ladder (§8.5) is therefore **not an RL feature**: C1 (kernel-pinned) is the default
   contract for *every* sampling-in-the-loop method — distill what you serve, serve what you distilled. (miles
   reaches the same place from the LLM side: its INT4 QAT exists because rollout efficiency at 1TB scale required
   training to see inference numerics.)

This reframes the unification question (§11.12): in LLMs, an inference engine can ignore training because
optimization is training-free; in diffusion, an inference engine that cannot run inside training loops is **cut
off from the actual optimization frontier**, and a training framework without a serving-grade rollout distills
models toward a toy sampler. FastVideo's "advantage that is also its disadvantage" is precisely that it occupies
the one position where the (recipe, runtime) pair can be co-designed and parity-gated. The RL argument of §8.4 is
a special case of this.

### 8.4 The lesson from RL systems: the two-runtime tax

**Why the split exists.** Training forwards want FSDP/Megatron sharding, activation checkpointing, high-precision
gradient accumulation, and sequence packing. Rollout forwards want KV caches, continuous batching, CUDA graphs,
quantization, and inference kernels. Run the *same weights* through both and the log-probs differ — different
kernels, precision paths, batching, reduction orders. The policy that generated the trajectories (π_rollout) is no
longer the policy being optimized (π_train); PPO-style ratios are computed against the wrong baseline; at scale
(especially MoE + FP8) this is a documented cause of training collapse. Everything below exists to patch that
wound:

| The tax | verl-omni (diffusion RL) | miles (LM RL) |
|---|---|---|
| **Dual model definitions** | Wan2.2 implemented *twice*: a rollout pipeline inside vLLM-Omni (`pipelines/wan22_dance_grpo/vllm_omni_rollout_adapter.py`, with trajectory/log-prob capture and dual-expert switching) and a diffusers/FSDP2 trainer engine (`workers/engine/fsdp/diffusers_impl.py`, VeOmni variant) | model exists in sglang *and* Megatron, bridged by weight-name iterators (Megatron fused params → HF names) |
| **Mismatch corrections** | `trainer/diffusion/rollout_correction.py`: *bypass mode* (trust rollout log-probs) vs *decoupled mode* (recompute under trainer + truncated importance sampling + rejection-sampling masks) | TIS/MIS in `backends/training_utils/loss_hub/corrections.py`: clamp `exp(logp_train − logp_rollout)` to configurable `[tis_clip_low, tis_clip]` bounds (CLI defaults `[0, 2.0]`; their MIS example config uses `[0.5, 2.0]`) or hard-mask out-of-range tokens (MIS); `tis_clipfrac` monitored as a health metric |
| **Determinism engineering** | deterministic rollout/reward/trainer modes; per-step seed derivation | a whole program for **bit-wise identical log-probs**: FlashAttention-3 + DeepGEMM batch-invariant kernels + deterministic sglang, both sides pinned to TransformerEngine |
| **MoE routing drift** | — | **R3 rollout routing replay**: capture per-token expert choices in sglang (`(seq−1, layers, top_k)` int32 — ~60 MB/sample at 32K×60×8) and replay them in the Megatron forward |
| **Precision drift** | — | **Unified FP8**: identical quantization recipe (blockwise / MXFP8) on both sides; weights dequantized to BF16 for transfer, re-quantized identically on arrival |
| **Weight-sync machinery** | FSDP2→vLLM-Omni LoRA collection + RPC; sleep levels (1 = keep VAE/text-encoder resident, 2 = full offload) | three transports behind one interface: Ray-IPC zero-copy (colocated), bucketed NCCL broadcast, P2P-RDMA with pinned-CPU staging; per-sample `weight_versions` staleness tags |

The punchline for FastVideo: verl-omni's Wan2.2 rollout adapter is a **re-implementation of code this repo already
owns** — external RL frameworks must rebuild our models inside other engines to RL-train them, then correct the
numerics. Doing RL natively means the model exists once and the corrections become a fallback, not a foundation.

**The moat made concrete.** Under this design, rollout = the *same* shared `DenoiseLoop` + a `CFGPolicy` (for NFT:
conditional-only / identity — `predict_noise` already carries `conditional:bool` + `cfg_uncond` as the policy knob,
`train/models/base.py:142-152`) + an `OutputSpec(capture=behavior)` + the step scheduler + the caches — so rollout
forward *is* serve forward + capture. Every serving optimization is automatically a rollout optimization, and there
is one numerics surface, no rollout-vs-train kernel gap to patch. The alternative is the **two-runtime tax** the
table above enumerates: verl-omni re-implements Wan inside vLLM-Omni and stitches it with `rollout_correction.py`'s
IS / rejection masks; miles' TIS/MIS, bitwise log-probs, R3 routing replay, and unified-FP8 are *all* mismatch
patches. FastVideo's own landed NFT confirms it from the other side — a vendored bare-model loop with zero
serving-grade optimizations (§8.1). Collocation at FastVideo's 1–30B FSDP2 scale **deletes the second runtime, the
correction layer, and the 3rd/4th denoise-loop copies** — and it makes the `(recipe, runtime)` flywheel (§8.3,
§9.5) honest, because preferences are then collected under the very serving profile that ships. This collocation of
inference + post-training (including RL) on one substrate is the central, durable moat. Three caveats the design
already names keep this from being hand-waving: old-policy weights are a `WeightSyncPlan` **role** (rollout samples
from `self.old`, decay-blended `diffusion_nft.py:858-862`); the dense-attn pin is a Precision/Attn **policy** on the
card; and likelihood-free NFT is the **C2 behavioral rung** (no log-probs to match — `old_deviate` / ref-MSE in
prediction space, `diffusion_nft.py:777-778`).

**What stays regardless** (intrinsic to RL, not artifacts of the split): trajectory buffers and rollout
management, group-relative advantage computation, multi-reward serving with async overlap, staleness control,
evaluation-during-training, checkpointing, fault tolerance (health-monitored engines). We adopt these; we delete
the rest. Two of them already exist landed in-repo as the baseline to build from: #1450's gathered per-prompt
group-advantage normalization (`diffusion_nft.py::_compute_advantages`) and its weighted `MultiRewardScorer`
(in-process, GPU-resident PickScore/CLIP — the async/HTTP serving half is the part still to come).

**The boundary condition, stated honestly.** Miles defaults to Megatron because its targets are ≥70B MoE models
(1TB-scale K2-Thinking with INT4 QAT). FastVideo's training targets are 1–30B DiTs/MoTs on FSDP2/HSDP — *that
scale assumption is what makes the single-runtime bet viable.* If a Megatron-class trainer ever becomes necessary
(100B+ MoT training), those models re-enter the two-runtime world — and the toolkit below (C0 corrections,
Behavior Record, weight transports) is designed to still apply as the fallback. This is a trigger to monitor, not
a reason to build for it now.

### 8.5 The bet: one model definition, two execution profiles, a consistency ladder

Even in one runtime there are honestly **two execution profiles**: the training forward (grad-enabled, activation
checkpointing, FSDP-gathered params, no caches) and the rollout forward (no-grad, CUDA graphs, KV/feature caches,
TP/SP-sharded, possibly quantized, possibly cache-dit-accelerated). Same repo ≠ same numerics — and the precise
mechanism matters. Thinking Machines' ["Defeating Nondeterminism in LLM
Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) is the definitive
analysis: forward passes contain no atomics and are *run-to-run deterministic for a fixed batch*; what breaks is
**batch invariance** — kernels switch reduction strategies with batch/shape (split reductions in RMSNorm, split-K
and tile/instruction selection in matmul, KV-split counts in attention), so a request's numerics depend on what it
was batched with. Compose that with variable load (or, in our case, with rollout-vs-training profile differences)
and you get nondeterminism — their headline: 1,000 temperature-0 completions of Qwen3-235B produced **80 unique
outputs**, first diverging at token 103; with batch-invariant kernels, 1,000/1,000 identical. Miles' bitwise
program (FA3 + DeepGEMM) and sglang's deterministic mode descend from this work. One codebase makes parity
tractable, not automatic — so consistency is a **declared, measured contract**, per run:

- **C0 — corrected.** Profiles free to differ: quantized rollout, cache-dit on, CUDA graphs. Trajectories store
  rollout log-probs; objectives apply TIS/MIS (miles' clamp/mask semantics ported to per-step diffusion log-probs,
  with verl-omni's rejection-sampling mask as an option); `logprob_diff` and clip-fraction are always-on training
  telemetry. This is the throughput mode and the graceful-degradation mode.
- **C1 — kernel-pinned (default for RL).** Rollout-for-training pins the profile to the trainer's: same attention
  backend, same `PrecisionPolicy`, distribution-altering interceptors disabled, eager or shape-matched graphs.
  ParityAligner gates per-step log-prob diff ≤ ε in CI for likelihood-based methods (ε set per model family from
  Phase-1 harness data, not guessed); the drift metric should sit at ~0 during training. "Trust the rollout
  log-probs" (verl-omni's bypass mode) is then actually sound instead of hopeful. The landed DiffusionNFT (#1450)
  is C1 by construction — in-process rollouts under the trainer's kernels, `attn_kind="dense"` hard-pinned on
  rollout and all three training forwards, `FASTVIDEO_ATTENTION_BACKEND` defaulted to `FLASH_ATTN` in the launch
  script — and it
  already paid a C1-class bug tax that proves the discipline isn't free: `WanModel.predict_noise` had to stop
  deriving autocast dtype from caller tensors because RL-created fp32 intermediates silently changed numerics
  between rollout and gradient forwards (the `_get_training_dtype()` fix + `test_wan_training_dtype.py`). The CI
  enforcement pattern also has a landed coarse precedent: #1396's per-method, device-keyed grad-norm regression
  refs (10% rtol) — the same pin-a-golden-and-gate shape ParityAligner applies at per-step resolution.
- **C2 — bitwise.** Batch-invariant kernel set + deterministic mode + Behavior Record replay → bit-identical
  log-probs between rollout and training forwards. The recipe is known (per the Thinking Machines post): fixed
  reduction strategy for norms, one pinned matmul kernel config across shapes (~20% off cuBLAS peak, acceptable),
  and attention with **fixed split-size** (not split-count) KV reductions plus a consistent KV layout (page-table
  update before the kernel) so numerics are independent of how many tokens are processed at once — implementable
  as a `torch.Library` op-swap (their `batch_invariant_ops`) behind our attention-backend registry as a
  `deterministic` backend mode — and torchtitan already carries `set_batch_invariance()` utilities, so the kernel
  set is landing in the PyTorch ecosystem proper, not just a vendor library. Measured cost in their vLLM demo:
  26s → 42s (~1.6×) with the improved attention kernel. The payoff — stated with its conditions: in their RLVR
  experiment, no correction collapses mid-run, importance weighting trains with KL ≈ 10⁻³, and bitwise
  sampler-trainer identity holds KL flat at exactly 0. For diffusion, **kernels alone do not confer that**:
  C2's "true on-policy" holds only as the conjunction of batch-invariant kernels *plus* the full Behavior Record
  (sampler noise/RNG, guidance state, routing) *plus* C1's preconditions (matched precision, interceptors off,
  same weight version, identical SDE log-prob arithmetic) — and it is a per-trajectory replay guarantee, verified
  by ParityAligner, not a blanket property. C2 is our golden-run, divergence-debugging, and MoE-routing-parity
  mode; C1+TIS covers daily training.

**Where batch variance actually bites FastVideo** (the post's lens applied to our workloads): the video denoise
path is effectively batch-1 with fixed shapes — accidentally near-batch-invariant already, our structural
advantage. The exposure is elsewhere: (a) **AR pathways** (Cosmos3 reasoner, omni thinkers) under continuous
batching, where batch composition changes every decode step — the blog's exact scenario, and why C1/C2 for AR
segments needs either batch-invariant decode kernels or frozen-batch rollout; (b) **image serving** with bucketed
cross-request batching; (c) **parallelism-degree differences** between rollout (SP/TP-sharded) and training
(FSDP-gathered) forwards — sharding is a batch-like axis that changes reduction layout, so C1 pins matched degrees
and C2 requires invariance across them; (d) **chunked causal video attention** (`ChunkKVPool`), where the
fixed-split-size principle applies directly: KV-chunk reductions must be invariant to how many chunks are resident,
or world-model rollouts won't replay bit-identically.

**Behavior Record (R3, generalized) — budgeted at real bytes.** Trajectories optionally capture *behavior
artifacts* beyond latents/log-probs: per-step RNG state (cheap — seeded generators in one shared loop reproduce
draws, so seeds suffice), sampler branch choices (cheap), and MoE routing decisions (**not cheap for diffusion**:
miles' ~60 MB/sample is per-token routing with one forward per token; diffusion re-routes the entire packed
sequence at *every denoise step, per CFG branch* — steps × branches × tokens × sparse-layers × top_k, which for a
Cosmos3-class request (~50K packed tokens, ~24 sparse layers, top_k 4, 35–50 steps × 2 branches) is **~1.3–1.9
GB/sample** int32, ~0.3–0.5 GB as uint8 — tens of GB per GRPO group). Consequently routing capture is an
**opt-in instrument for C2 goldens, MoE-parity debugging, and divergence hunts — not a training-scale default**;
routine RL runs at C1 (or C0+TIS) without it. If it is ever wanted at fleet scale, the GB-trajectory
streaming/storage design it forces is acknowledged as undesigned (§12 known gaps). The plumbing is an
`OutputSpec` feature of the loop (§6.2.2); the bytes are the cost, and they are stated.

**Plugins are policy-relevant.** cache-dit during rollout changes the sampled distribution — trajectories then come
from a behavior policy β ≠ π_θ. Contract: interceptors declare `distribution_altering: bool`; at C1+ such plugins
are disabled for rollout-for-training; at C0 they are permitted and stored log-probs must reflect the actual
(cached) computation, with IS correction in the objective. Enforced declaratively at pipeline build, not by
convention.

**Precision unification** is the same contract one level down (miles' FP8 lesson): rollout and trainer share one
`PrecisionPolicy` / quantization recipe — "train BF16, roll out FP8" is only legal at C0. This dovetails with the
in-repo Attn-QAT/FP4 work: QAT is exactly what makes low-precision rollout consistent with training by
construction.

### 8.6 Weights, parallelism duality, and placement

The trainer mesh (FSDP/HSDP) and the engine mesh (TP/SP/CFG) are different layouts over the same parameters — both
expressed in the stacked-axis spec (§6.3.4), e.g. `[hsdp_replicate(2), hsdp_shard(4)]` vs
`[dp(2), cfg(2), sp(2), tp(1)]`. The **`WeightSyncPlan`** is computed from **three inputs, not two** — axis specs
alone are necessary but not sufficient: (a) the pair of mesh specs; (b) **per-model weight-layout adapters**
covering what specs cannot express — fused QKV/MLP packing, MoE expert placement, quantized storage formats, LoRA
deltas, FSDP flattening, and checkpoint key translation (the `param_names_mapping`/state-dict-adapter lineage this
repo already maintains for conversion; torchtitan reaches the same conclusion — its fused-SwiGLU override needs
model-level state-dict adapters, and its own notes mark some redistributions as inexpressible,
`spmd_types.py:45`, `overrides/fused_swiglu.py:34`); (c) the transport. The plan is validated pre-flight and
CPU-testable with fake pools — replacing hand-written per-combination code (the bug class miles' weight-name
iterators and verl's resharding exist to manage), while being explicit that the per-model adapter is authored, not
derived. Phasing: Phase 2 ships the **colocated** cases (same-layout zero-copy; in-place reshard for the Phase-1
families' FSDP↔engine layouts, adapters hand-checked); the general cross-mesh plan over the declarative spec lands
with the spec in Phase 3.

- **Transports** (miles' menu, one interface): colocated same-layout → zero-copy views — the unified-runtime
  dividend: often a literal no-op; colocated different-layout → in-place reshard; disaggregated → bucketed
  collective broadcast or P2P-RDMA with pinned-CPU staging. `multimodal_gen`'s RL weights-updater is the in-family
  precedent for the RPC surface. At fleet scale, weight distribution and version tracking should ride Dynamo's
  ModelExpress rather than our own fan-out — that is ask A5 (§6.3.6): versioned incremental broadcast +
  staleness-aware routing, with our transports as the colocated/fallback path. Two further precedents to reuse:
  torchtitan's RL experiment syncs PolicyTrainer→vLLM via **TorchStore** push/pull with direct-RDMA state-dict fill
  — the PyTorch-ecosystem transport to evaluate first for the disaggregated path (with its own caveat: DCP and
  TorchStore do structure-matched fill, *not* automatic cross-mesh resharding — the `WeightSyncPlan` still owns the
  layout math); and cosmos-rl's **buffer-model async sync** (a background thread on a side CUDA stream writes
  incoming weights into a parameter-only clone; the live model swaps at safe points) is the pattern for
  non-blocking colocated updates.
- **Sleep/wake is component-granular** (verl-omni's levels, generalized by our component registry): a rollout pool
  can drop KV/feature caches and DiT weights while keeping VAE/text-encoder resident (level 1) or release
  everything (level 2). Implementation reference: vLLM's `CuMemAllocator`
  (`vllm/device_allocator/cumem.py`) — tag-based unmap/offload where our tags are component names (finer than
  vLLM's global `weights`/`kv_cache` tags), with its documented caveats inherited: CUDA graphs invalidate across
  sleep (recapture on wake), woken KV pools are zero-initialized (sessions re-prefill), and encoder/embedding
  caches must `reset()` on weight update.
- **Staleness is first-class** (miles): every trajectory carries `weight_version`; `update_weights_interval` and a
  max-staleness bound govern the async overlap loop (generate batch N+1 while training on N — the
  `train_async.py` pattern); over-sampling with abort/truncation statuses handles stragglers, and loss masks zero
  out incomplete samples. The landed NFT method (#1450) adds a wrinkle the plan must carry: its behavior policy is
  not the student but a decay-blended **old** copy (and validation optionally swaps in the EMA shadow) — so a
  `WeightSyncPlan` ships a declared *role* (student / EMA / blended old policy), never implicitly "the weights",
  with the blend applied trainer-side before push or pool-side on arrival.

### 8.7 The RL layer on the engine

Components (all under `train/`, consuming the substrate and the engine — never the reverse). These are now the
**migration target for landed code**, not a blank-slate plan: `train/methods/rl/` exists on main (#1450) as a
1,729-line package — a 1,075-line `TrainingMethod` with advantages, objective math, and inner optimization
inlined, driving its own reusable-but-package-private sampler and reward modules — no `RolloutFn`, no
`TrajectoryBuffer`, consuming neither substrate loops nor engine — and its three `.agents`
skills (add-rl-method, add-reward-model, rlhf-training-abstractions) codify the current pre-§8.7 conventions.
The list below is what that module refactors into as the substrate and engine land:

- **`RolloutFn`** — miles' single best interface lesson: *users write one function*. Contract:
  `(ctx: RolloutContext) -> list[Trajectory]`, where `ctx` provides an engine client (seeded generation, with
  log-prob capture where the method's `OutputSpec` declares it — NFT-class methods don't), the data source, and
  tool hooks. Covers single-shot diffusion sampling, N-sample GRPO groups,
  multi-turn/agentic episodes, and future interleaved omni episodes — without touching the trainer.
- **`Trajectory`** (our `Sample`): request + seeds, per-step records as declared by the method's `OutputSpec`
  (final latents only for NFT-class; latents + timesteps + log-probs for GRPO-class), Behavior
  Record, rewards (scalar or dict), loss masks, `weight_version`, status (`COMPLETED/TRUNCATED/ABORTED`).
- **`TrajectoryBuffer`**: collection, filtering, group-by-prompt, conversion to training batches.
- **Reward layer** (verl-omni's shape): weighted multi-reward aggregation; sync in-process scorers and async HTTP
  scorers; reward computation overlapped with rollout; reward models loadable as engine components — and later as
  engine graph nodes scoring decoded artifacts. The landed `train/methods/rl/rewards/` is the right seed: thin
  PickScore/CLIP scorers loading from HF (the #1438 draft's vendored HPSv3/VideoAlign repos were dropped before
  landing — good hygiene to keep, §12 open question 13), a duck-typed `RewardScorer = Callable[[media, prompts],
  Tensor]` contract, and weighted aggregation in `MultiRewardScorer` — what's missing is the async/HTTP/offload
  half, since today's scorers sit GPU-resident on every training rank for the whole run.
- **Objectives library**: FlowGRPO/DanceGRPO (clipped PPO over per-step SDE log-probs; SDE-with-logprob is a
  sampler policy, implemented once), DiffusionNFT (final latents + policy/ref forwards — extracted from the landed
  #1450, whose objective math today lives inlined in the monolithic method class rather than as a reusable
  module), Diffusion DPO (offline pairs), and AR GRPO/GSPO for token pathways — with cosmos-rl's stability kit
  (AIPO one-sided clipping for async off-policy rollouts, dual-clip, KL-threshold off-policy masking) as the
  C0-mode options. Shared group-relative advantage utils.
- **Two rollout modes, one set of trajectories**: *library mode* — the method drives substrate loops in-process
  (required when backpropagating through sampling; the landed #1450 is library-mode *shaped* — in-process,
  no-grad, trainer kernels — but drives its private `DiffusionSampler` instead of substrate loops, which is the
  Phase-1 migration); *engine-client mode* — `RolloutClient` submits seeded
  `OmniRequest`s to a colocated or disaggregated engine, inheriting queueing, batching, step interleaving, and
  (at C0) cache acceleration. At fleet scale the disaggregated rollout pool is a **Dynamo-fronted fleet**: rollout
  requests route through Dynamo (staleness/version-aware once ask A5 lands, §6.3.6), the Planner shifts GPUs
  elastically between trainer and rollout pools, and weight updates propagate via ModelExpress. The consistency
  ladder (§8.5) plus seeded RNG in `RequestState` is what makes the modes interchangeable without retuning.
- **Loops**: sync (rollout → reward → advantage → update → weight-sync) and async (overlapped, staleness-bounded)
  — thin scripts over these components, deliberately the same shape as miles' `train.py`/`train_async.py`.

**Omni-RL — expressibility is established; the algorithm is not.** A hybrid episode over a MoT model (Cosmos3):
one trajectory containing AR reasoner segments (token log-probs) *and* denoise segments (per-step SDE log-probs),
rewarded jointly — one model, one trajectory, two loop types. Engine-per-stage RL frameworks cannot express this
without two engines and two weight copies; that structural advantage is real. But "optimized with mixed
objectives" names an **unsolved algorithm design**, stated as three open problems (open question 16): (1) *scale
mismatch* — token log-probs are O(1–10) nats over 10²–10³ tokens while per-step diffusion log-probs are Gaussian
densities over 10⁶–10⁷ latent dims; a joint clipped-ratio objective needs principled per-segment normalization
that none of the cited recipes (FlowGRPO/DanceGRPO/NFT/AIPO/GSPO) provides, and getting it wrong lets one
modality silently dominate the shared trunk; (2) *credit assignment* — the reasoner influences video reward only
through sampled discrete tokens re-entering as conditioning, a non-differentiable boundary, so token segments get
sparse trajectory-level signal while denoise segments get dense per-step ratios, both updating shared attention
weights with no interference analysis; (3) *reasoning regression* — RL-updating the und pathway on
video-correlated reward risks degrading reasoning; hybrid episodes need reference-model KL anchoring.
Accordingly, the Phase-4 deliverable is **hybrid trajectory capture demonstrated plus an objective design study**
— not a pilot that presumes the objective exists.

**What we deliberately do not build**: verl's general Ray hybrid-controller choreography (actor/critic/ref/reward
worker groups with cross-runtime resharding). At our scale the trainer drives a simple loop; if critic/ref models
arrive they are additional engine pools, not a new controller paradigm.

### 8.8 Config unification with training

`train/utils/training_config.py` keeps what is genuinely training's (optimizer, data, loop, checkpoint, tracker
sections) and **imports the shared layers** for everything else: the deploy/parallelism spec (§6.3.4) replaces its
`distributed:` block (today a field-for-field duplicate of `FastVideoArgs.tp_size/sp_size/hsdp_*`), and model/
pipeline identity comes from the same `ModelSpec`/`PipelineSpec` the engine loads (§6.6). One description of "what
model, how sharded" serves inference, training roles (student/teacher/critic), and rollout.

---

## 9. The application pull: Dreamverse, ComfyUI, and the workflow layer

Dreamverse (`apps/dreamverse/` — Next.js frontend + FastAPI runtime; open-sourced with the
[vibe-directing](https://haoailab.com/blogs/dreamverse/) and
[release](https://haoailab.com/blogs/fastvideo-dreamverse-release/) posts) looks out of place in an inference-runtime
design at first glance. It is the opposite: it is the **proof of the thesis and the generator of the requirements**.
"Vibe directing" — steering video through continuous natural-language revision, 30-second scenes composed of chained
5-second clips, ~4.55 s per 5 s 1080p clip on one B200 (faster than real-time playback) — is a product category that
exists *only because of (recipe, runtime) pairs* (§8.3): faster-than-realtime clips come from NVFP4 (QAT-backed) +
FA4 + `torch.compile` on the runtime side and from distillation recipes on the training side, streamed as fMP4 over
WebSocket with frame/audio conditioning carrying continuity between segments. The blog's own one-liner — *"better
creative work comes from a faster feedback loop, not just a better model"* — is §8.3 in product language.

### 9.1 What Dreamverse proves about the runtime gap

Read `apps/dreamverse/arch.md` and the division of labor is striking: the **app server owns "GPU assignment and
worker lifecycle," "actual generation queue semantics," warmup, and "stream chunk emission"**
(`dreamverse/gpu_pool.py`, `main.py`). The application layer is hand-building the engine plane of §6 because the
runtime doesn't provide it — P4 made manifest in a product, with a price tag: **one B200 per active user session**.
Dreamverse is simultaneously the best evidence for the engine (Phase 2 retires `gpu_pool.py` into an engine client
at single-session parity and runs the duty-cycle study; multi-session multiplexing lands at the Phase-3 gate) and
the best customer for nearly every §6 feature:

| Dreamverse need (shipping today, hand-rolled) | Engine feature it pulls |
|---|---|
| GPU pool + queue + warmup in the app server | engine scheduler, warmup requests, multi-tenant session multiplexing (§6.3.1) |
| fMP4 segment streaming over WebSocket, `ltx2_segment_start`/chunk events | `OmniEvent` streaming contract; loop `emit` (§6.2.2) |
| frame/audio conditioning chained across 5 s segments; init-image persistence | **session abstraction**: session-scoped latents/caches and continuation conditioning as first-class request inputs — the same machinery as interactive world-model sessions (§12, open question 9, which now has a production owner) |
| prompt-rewrite LLM (external provider today, with timeout/fallback) | AR graph node or external-engine node (§6.3.4); a future omni model collapses rewrite+generation into one resident model |
| LoRA controls per direction (#1420) | LoRA hot-swap on the engine path |
| SP for serving (#1424), NVFP4/FA4/compile | DeployConfig parallelism + PrecisionPolicy + plugins — per-app deploy YAML instead of bespoke wiring |
| safety filtering, prompt memory, session protocol | stays in the app — the workflow layer is a *client* of the engine, never merged into it |

That last row is the architectural rule this section adds: **the workflow layer composes engine requests; it does
not extend the engine.** Dreamverse keeps UX, prompt memory, safety, and session semantics; the engine absorbs
exactly the parts the app had to hand-roll (pool, queue, warmup, streaming transport, placement).

**Where multi-session capacity actually comes from** (a correction the review process forced — note that today's
`gpu_pool.py` deliberately admits one user per GPU, `is_available` false while `connected_users` is non-empty):
step interleaving improves *fairness*, not throughput — two always-on sessions are ~2× service demand, and no
scheduler conjures that away. The capacity claim rests on three real mechanisms: (1) **duty cycle** — directing
sessions are idle-heavy (the user types, watches, decides between segments), so time-averaged demand per session
is well below one GPU; (2) **admission control on the cost model** — sessions admit against measured per-segment
cost and a per-class latency SLO, with simultaneous segment requests queued within the SLO rather than
co-batched; (3) the distillation roadmap lowering per-segment cost over time. Accordingly the multi-session gate
is defined on a **recorded duty-cycle trace** of real session behavior — p95 segment latency within the SLO vs
the single-session baseline — never on synthetic always-on load.

**Transport triggers (from the LiveKit/WebRTC review).** fMP4-over-WebSocket remains the right transport for
segment streaming. WebRTC (LiveKit-class SFU) becomes the transport when any trigger fires: sub-second streaming
latency, one-to-many fanout, or **interactive world models** — the <100 ms motion-to-photon budget
(≈20–40 ms action round-trip + ≤40 ms/frame inference + ~10 ms encode/decode/render) is unreachable over TCP/WS
and needs WebRTC data channels as the action path. The engine contract that keeps both transports thin adapters:
`OmniEvent` optionally carries **raw frames + PTS** (not only encoded segments — the SFU SDK encodes in a native
thread from RGBA/I420 buffers), sessions accept an async action stream, and a flush/interrupt RPC discards stale
frames on a direction change. Until a trigger fires, the SFU/TURN/token operational weight is not paid.

### 9.2 The flywheel: where post-training, inference, and RL meet a product

Dreamverse's intended trajectory — post-training + inference + eventually RL composing one production-ready
workflow — closes the loop of §8:

```
distillation/QAT recipes (§8.3) ──► faster-than-realtime engine ──► vibe directing UX
        ▲                                                                │
        │                                                                ▼
   RL / DPO post-training (§8.7) ◄── preference data: kept vs discarded revisions,
                                      edit chains, per-segment directions
```

Vibe-directing sessions emit *dense, naturally-labeled preference data* — every revision a user keeps over the one
they discarded is a DPO pair; every edit chain is a trajectory with human reward. No video lab has this loop in one
stack: labs with products lack open post-training; open frameworks lack products. The §8 RL layer is what turns
Dreamverse from a demo into a data engine, and the consistency ladder (§8.5) is what guarantees the preferences
collected under the serving profile actually transfer into training.

### 9.3 Positioning

The diffusion community is structurally more diverse than LLM serving — many base models, LoRA ecosystems,
workflow culture (ComfyUI) — and rewards stacks that compose models rather than serve one. "Vibe directing" names a
UX category the way "vibe coding" did, and the moat behind it is not the frontend: it is the only stack where the
app, the engine, and the post-training that makes the app possible are co-designed and parity-gated. Dreamverse
also functions as the runtime's production forcing function (sessions, safety, cost-per-user, consumer-GPU targets
— the Attn-QAT NVFP4-attention work aims at RTX 5090/4090/3090) — keeping the engine honest the way Cosmos3 keeps
the model layer honest (§4.1). Dreamverse's own planned controller/provider split (local-first compute over
Runpod/Modal, per its `AGENTS.md`) stays the single-box story; at fleet scale it delegates to Dynamo (§6.3.5)
rather than growing its own orchestration — provisioning is not our plane.

### 9.4 The ComfyUI question: the funnel, the compiler, and the accelerated cloud

ComfyUI (`~/ComfyUI`, 116K stars, GPL-3.0) is where the commodity diffusion community lives, and its tree tells us
exactly what it is and isn't: a **node-graph orchestrator** — 64 core nodes + 117 `comfy_extras` packs (Wan, Cosmos,
Hunyuan, Mochi video natively in-core), a runtime-patching model manager (`comfy/model_patcher.py`, LoRA hot-patch,
smart VRAM offload, FP8), **39 paid API-node providers in-tree** (`comfy_api_nodes/`: Kling, Runway, Veo, Sora… —
its monetization is orchestrating *other people's* models), 80 in-tree workflow blueprints (more via the registry) plus an App Mode that wraps
graphs in simplified UIs — and **no serving runtime**: a FIFO prompt queue, no cross-prompt batching, no
multi-tenancy, no distillation-awareness (`execution.py`, `server.py`). Structurally: ComfyUI is a frontend and
ecosystem; FastVideo is a (recipe, runtime) factory. Those are complements before they are competitors, which
shapes the play as **three sequenced motions**:

- **Motion A — embed (exists today, underinvested).** `comfyui/` in this repo already ships ComfyUI-FastVideo
  custom nodes (multi-GPU `VideoGenerator` inside their graphs). This is the funnel: meet 116K-stars'-worth of
  users inside their own tool and let the speed delta advertise itself. A "FastVideo Cloud" *API node* — slotting
  into the same `comfy_api_nodes` pattern Kling/Runway use — is the cheapest distribution wedge their own platform
  has standardized.
- **Motion B — compile (the core bet).** The decisive observation, stated at its defensible scope: **the
  tier-1/tier-2 static sublanguage of ComfyUI workflows maps onto `PipelineSpec`** — within that vocabulary, both
  are typed dataflow graphs over model components (the general claim "any workflow is a PipelineSpec" is false:
  arbitrary custom nodes, Python-level patching, and dynamic graphs are exactly what the tiers exclude). The popular surface is small and maps almost 1:1 onto §6:
  `CheckpointLoaderSimple→ModelSpec`, `KSampler→DenoiseLoop` (sampler/scheduler/CFG policies),
  `LoraLoader→LoRA hot-swap`, `CLIPTextEncode→TextEncode node`, `VAEEncode/Decode→codec nodes`,
  `ControlNetApply→ConditioningInjector`, `LatentUpscale→upsample stage`, `WanImageToVideo/EmptyCosmosLatentVideo→`
  task presets. A clean-room **workflow compiler** (ComfyUI JSON → `PipelineSpec` + `OmniRequest`; no GPL exposure —
  translate the format, never embed the code) over a **tiered core vocabulary** — tier 1 ≈ 20 nodes
  (t2i/i2v + LoRA + upscale), tier 2 ≈ 40 (inpaint, ControlNet, video-conditioning variants) — covers the canonical
  workflows, with **capability negotiation**: unsupported nodes produce a coverage report and a clear rejection,
  not silent wrongness. The long tail of arbitrary custom nodes is explicitly out of
  scope for v1 (option later: partition the graph and run unsupported subgraphs in a sandboxed headless ComfyUI
  *sidecar process* — GPL-safe by process isolation — but that is a deliberate, deferred complexity).
- **Motion C — productize (Dreamverse Studio).** Per the thesis that workflows matter less as models improve —
  which ComfyUI itself corroborates by shipping blueprints and App Mode to hide its own graphs — the destination is
  not a node editor clone but *workflows as products*: the top community workflows as one-click, real-time
  primitives inside Dreamverse (5 s 480p in ~2 s end-to-end today; consumer-GPU NVFP4 builds targeting
  5090/4090-class cards), with cloud execution for whatever the local GPU can't hold.

**What this genuinely adds to the architecture** (and it is bounded — no plane is redesigned):

1. **The workflow compiler** — a new request-plane frontend, peer to the OpenAI server: parse, validate against the
   supported vocabulary, emit graphs. The graph plane (§6.2.4) was built for exactly this shape.
2. **Heterogeneous checkpoint serving** — the one real new subsystem. Workflow users bring arbitrary checkpoints,
   stacked LoRAs, ControlNets; the engine has so far assumed one resident model family per pool. Required: a
   **weight/adapter cache** tier in the CacheManager (disk→CPU→GPU, LRU, content-hash keyed `ModelSpec` resolution),
   fast patch/unpatch with ComfyUI-grade LoRA-stacking semantics, and **model-affinity scheduling** (route requests
   to pools where the checkpoint is resident — the diffusion-serving analog of sglang's prefix-affinity routing).
   The LoRA story, honestly sized: **affinity scheduling + patch/unpatch is primary** — group by (checkpoint,
   adapter-set, strengths), with §6.3.2's weight-transition scheduling pricing the swaps. Punica-style batched
   multi-LoRA is a *deferred optimization with a real semantic gap*: vLLM's `LoRARequest` is one integer id with
   scaling baked into `lora_b` at registration, while ComfyUI traffic is N stacked LoRAs with continuous user-set
   `strength_model`/`strength_clip` tweaked per generation — pushing that through Punica means registering every
   (ordered-set × strengths) tuple as a synthetic adapter: near-zero cache hits across strength tweaks,
   registration churn in the stacked weight slots, and concatenated ranks colliding with `max_lora_rank`. Batched
   multi-LoRA earns a place only where adapter sets are small and strengths fixed — named fast-mode presets — not
   as the general workflow-cloud mechanism.
3. **Equivalence modes as product trust** — the consistency machinery (§8.5) reused commercially, with the claims
   scoped precisely. *Exact mode*: the user's base model and sampler math, deterministic seeds, reproducibility —
   **within our runtime**; bit-parity with ComfyUI's own execution is never promised (its model patching,
   sampler/noise details, and dtype/offload behavior differ). *Fast mode*: distilled weights, NVFP4/QAT, cache-dit
   — dramatically faster, *labeled*, with equivalence reports that are **quality-metric comparisons against a
   ComfyUI reference render** (SSIM-class + the §12 eval-system gap, which gates this product claim), not parity
   assertions. This is the moat restated: ComfyUI Cloud and GPU rental can only run stock workflows on
   rented silicon; substituting a faster model **credibly** requires owning the recipes (§8.3) — which is the one
   thing an orchestrator structurally cannot do.

Risks, named: the compatibility treadmill (mitigation: tiered vocabulary + coverage reports, never chase the
custom-node long tail), output-mismatch expectations (mitigation: exact/fast labeling), GPL hygiene (clean-room
compiler; sidecar only ever as a separate process; the in-tree custom-node suite carries ComfyUI-side licensing),
and platform response (ComfyUI controls the registry and frontend; our defense is the recipe flywheel it cannot
replicate, §9.2). Decision recorded in §11.13.

---

## 10. Migration plan

Principles: every phase ships green (SSIM + parity suites); every phase deletes or freezes the code it replaces
(no third stack); models migrate family-by-family behind the legacy adapter. Three additions this plan makes
because the repo's own history demands them:

- **Enforcement, not intent.** The last freeze failed empirically: since `train/` landed (2026-03-09, #1159),
  **19 commits modified the "frozen" `training/` stack** — including a brand-new `cosmos2_5_training_pipeline.py`
  added *nine days after* `training/AGENTS.md` forbade new models there (#1227), plus world-model training
  (#1179) and LongCat finetuning (#1244). This plan's freezes are therefore mechanical: a **CI path gate**
  rejects new files under `fastvideo/training/` and new `DenoisingStage` subclasses once Phase 1 lands;
  **CODEOWNERS** on legacy paths gives the freeze a human veto; every phase has a **named owner**.
- **The inflow rule.** New families land at ~1–2/month (Flux2-Klein and Lucy Edit in a single day; MatrixGame3,
  MagiHuman, Stable Audio, Gen3C this spring). Untreated, the migration tail grows faster than phases retire it.
  Rule: **from Phase-1 completion, new model ports land on the new abstractions** — loop inversion + policies +
  the custom-step hatch suffice for standard diffusion families; until then, new ports land legacy *and* join the
  migration checklist with the porting team owning the later migration. Exceptions are an explicit owner decision.
- **Sizing and dates.** Rough per-phase engineer-month estimates, validated at each kickoff: P−1: 2–3 · P0: 2–3 ·
  P1: 6–9 · P2: 8–12 · P3: 8–12 · P4: 10–15 · P5: 4–6. Owners are assigned at the design review, and Phase 5 gets
  a **calendar target** there — decision deadlines phrased as "by Phase 2" bind to that calendar, not to phase
  labels.

- **Phase −1 — Land the baselines (prerequisite, not optional).**
  Merge the Cosmos3 five-branch chain (`feat/cosmos3-tier-a-port` → `feat/cosmos3-reasoning`) to main, bringing
  its 150-test parity suite into CI: the design's forcing function currently exists only as unmerged branches
  (`pipelines/basic/cosmos3/` on main holds a stray `__pycache__`), and no substrate refactor should proceed
  against an unmerged baseline — the alternative is months of side-branch rebasing across
  ForwardBatch→RequestState and loop inversion, unowned and unsized. Seed SSIM references for the ~7 uncovered
  families (Cosmos 2/2.5, Hunyuan, Hunyuan15(+SR), HYWorld, MagiHuman, Waypoint, MatrixGame-v1) so G5's
  instrument has no vacuous passes. *Gate: cosmos3 suite green in CI on main; SSIM coverage = all shipped
  families.*
- **Phase 0 — Typed I/O (low risk, immediate value).**
  `OmniRequest`/`OmniOutput`/`TaskType` + artifact outputs; `VideoGenerator` shim; ForwardBatch adapter view. The
  in-flight `fastvideo/api/` schema converges here instead of landing as another parallel surface —
  `GenerationRequest` evolves into `OmniRequest`, `GeneratorConfig` into `DeployConfig`, and `compat.py` is
  **frozen** — no new entries, monotonic shrink per phase, zero lines at Phase 5 (§6.6). Cosmos3 audio leaves
  `batch.extra`. *Gate: all SSIM suites unchanged.*
- **Phase 1 — Loop inversion + policies + extension core.**
  `DenoiseLoop`/`ARDecodeLoop` + policy registry; migrate Wan family (incl. causal) and Flux2 first — they jointly
  exercise CFG variants, expert routing, chunk-KV, and image path; `DenoisingStage` subclasses shrink to specs +
  policies. The extension system core lands here because the loop is being rebuilt anyway: observer bus,
  Step/Block interceptors, the `run_blocks` convention, and the ParityAligner — with the **cache-dit adapter**
  (DBCache + TaylorSeer behind `BlockInterceptor`, algorithm-parity with sglang's integration) as the first
  interceptor; the dormant `enable_teacache` flag is retired in favor of plugin config, and
  `fastvideo/forward_context.py` (the vLLM-inherited global, §6.3.3) retires in favor of `StageContext` +
  `AttnMetadataProvider` + state extensions. `train/` distillation
  methods begin adopting the shared step functions, and the landed DiffusionNFT (#1450) migrates off its vendored
  `rl/common/sampling.py` `DiffusionSampler` onto them (§8.1) — a migration of shipped code with users, so it gets
  its own parity evidence: identical seeded rollout latents before/after, plus reward-metric and grad-norm
  regression neutrality. **CacheManager v0 ships here**: per-request chunk-KV slabs behind the
  `KVHandle` seam — the Phase-4 pooled slab allocator is a swap behind the same handle (stated staging, not a
  silent throwaway implementation).
  *Gate: per-model SSIM + a step-level parity harness, built on ParityAligner (old loop vs new loop,
  bit-identical); on the training side, the landed per-method grad-norm regression refs (#1396) extended to cover
  the migrated methods, RL included.*
- **Phase 2 — Engine + StepScheduler.**
  Async engine, queue, concurrency, image batching, streaming events; OpenAI server parity with `multimodal_gen`
  (videos/images endpoints — chat arrives with omni in Phase 4, and **AR continuous batching is deliberately
  deferred to Phase 4 per N5**: the pulling workload is the Cosmos3 reasoner; Phase 2 carries only the
  `ARDecodeLoop` contract from Phase 1). **LTX-2's loops migrate here** as a linear graph (the degenerate case;
  full fan-out waits for Phase 3) — required because Dreamverse runs LTX-2 and the dogfood path needs its loop
  boundaries open. Rollout-service surface lands here too: `update_weights` RPC with the **colocated** transports
  and hand-checked per-family adapters (§8.6 — the general cross-mesh `WeightSyncPlan` follows the spec in
  Phase 3), component-granular sleep/wake, trajectory `weight_version` tagging, and the `RolloutClient` for RL
  engine-client mode (§8.7). The **Dynamo-grade worker surface** ships here (§6.3.5), scoped to stock Dynamo —
  registration wrapping `AsyncEngine` (retiring the locked `examples/diffusers/worker.py` pattern), health/drain,
  cost metrics; streaming/cost integration beyond stock follows asks A2/A3 *or their fallbacks*, and the gate does
  not depend on upstream landing. Offline path still bypasses the queue. **Dreamverse is the dogfooding customer
  at Phase-2 scope**: single-session parity — `gpu_pool.py`/queue/warmup/stream-relay retire into engine-client
  calls — plus a recorded duty-cycle concurrency study that sets the Phase-3 multi-session gate (§9.1). *Gate:
  serving load tests; batch-of-1 video latency regression ≤ 2% (gated on the Phase-1 overhead measurement); an RL
  smoke test runs the landed DiffusionNFT (#1450) in engine-client mode with C1-pinned rollouts — seeded
  final-latent parity against its in-process path on Wan, per-step activation drift measured by ParityAligner and
  within the Phase-1-calibrated ε (per-step *log-prob* drift becomes the gated metric when the first GRPO-class
  method lands); Dreamverse runs single-session on the engine at parity; the engine deploys under
  stock Dynamo as an aggregated `ModelType.Videos` worker and passes health/routing smoke tests.*
- **Phase 3 — Graphs + placement + workflow frontend.**
  PipelineSpec graphs, fan-out models (LTX-2, Hunyuan15+SR), role pools + connectors (port `multimodal_gen`
  disagg), deploy YAML, and the declarative stacked-parallelism spec with pre-flight validation and fake-pool CPU
  tests (§6.3.4). The ComfyUI workflow compiler MVP lands here (it is a client of the graph plane): the tier-1
  (~20-node) vocabulary with capability negotiation, plus the weight/adapter fleet cache and model-affinity
  scheduling (§9.4). Dynamo integration deepens accordingly: affinity events and NIXL-aligned disagg bootstrap
  (§6.3.5; asks A1/A4 in §6.3.6), and the general cross-mesh `WeightSyncPlan` lands with the spec (§8.6). *Gate:
  LTX-2 A/V end-to-end (full fan-out graph); disagg throughput benchmark; topology test suite green on CPU; the
  top community t2i+LoRA / i2v / upscale workflows compile and run with equivalence reports (quality-metric vs a
  ComfyUI reference render, §9.4); **Dreamverse multi-session** — ≥2 concurrent sessions per GPU on the recorded
  duty-cycle trace with p95 segment latency within the SLO set by the Phase-2 study (§9.1).* The study's published
  load profile and targets double as the **§11.6 falsifier**: if step-level scheduling does not beat a
  request-level baseline on those numbers, the simpler scheduler wins and the StepScheduler is descoped — the
  complexity must earn itself on measurements, not on this document.
- **Phase 4 — Omni/MoT native + RL consistency.**
  `PackedSeq` + multi-pathway layers; CacheManager (paged text KV + chunk KV + feature caches); Cosmos3 re-port onto
  the new abstractions (reasoner gets KV cache; t2vs joint denoise; rollout); session-scoped world-model serving.
  **AR continuous batching and paged-KV scheduling arrive here with their pulling workload** (the Cosmos3 reasoner
  and omni chat — N5's condition met), as does `/v1/chat/completions`.
  RL hardening lands alongside: the consistency ladder enforced end-to-end (C1 default, C2 bitwise mode for goldens
  and MoE routing parity), Behavior Record capture/replay (an opt-in instrument, §8.5), TIS/MIS objectives as C0
  fallback, and the omni-RL deliverable scoped per §8.7: **hybrid trajectory capture demonstrated + the objective
  design study** (open question 16) — not a pilot that presumes the objective exists.
  *Gate: Cosmos3 150-test parity suite passes on the new runtime; reasoner pool efficiency — tokens/s/GPU at
  target concurrent denoise throughput (§6.3.1) — alongside the ≥10×-vs-re-prefill sanity floor; C1
  rollout↔training drift ≈ 0 on a Wan RL run, with the drift dashboard live — per-step log-prob drift if a
  GRPO-class method has landed by then, seeded final-latent parity for the likelihood-free NFT otherwise.*
- **Phase 5 — Deprecation, with its real precondition stated.**
  Phase 5 is not just "remaining families migrated": the legacy `training/` stack is a *live consumer* of exactly
  what Phase 5 deletes (`training_pipeline.py:39` imports `ComposedPipelineBase`/`ForwardBatch`/`LoRAPipeline`;
  its validation instantiates real legacy pipelines running the legacy `DenoisingStage`;
  `distillation_pipeline.py:31` likewise). So **legacy `training/` retirement is a Phase-5 precondition**: its
  remaining users migrate to `train/` per the checklist, *then* `ComposedPipelineBase`, the legacy
  `DenoisingStage`, `forward_context.py`, `compat.py`, and `RayDistributedExecutor` are deleted together. This is
  where 4 loop copies → 1 actually completes — not before, and the document claims it nowhere earlier. Docs +
  `.agents/` maps updated. *Gate: the deletion diff itself — a tree with one loop implementation.*

---

## 11. Trade-offs and alternatives considered

### 11.1 Adopt sglang `multimodal_gen` upstream instead of building here

FastVideo's runtime already has a serving-hardened fork living in sglang. Option: make sglang the runtime and
FastVideo a model/stage library.

- **Pros:** serving layer (server, scheduler process, disagg, warmup, RL weight updates) exists and is maintained;
  one community runtime; router/gateway integration free.
- **Cons:** `multimodal_gen` inherited our P1–P3 problems and is diffusion-only — no AR loop, no KV cache, no
  step-level scheduling; the hard 40% of this design (loop inversion, hybrid MoT, CacheManager) doesn't exist there
  either, so we'd build it anyway — in a repo we don't govern, on sglang's release cadence, while FastVideo is the
  research vehicle (Cosmos3, VSA, new ports land here first).
- **Decision:** build natively, but keep **API and concept parity** (request/queue/disagg shapes, server endpoints)
  so the work upstreams cleanly later. The runtime layering (§6.0) is deliberately compatible with theirs.
  *Open strategic question for the team: do we plan a re-convergence (upstream Phase 2's engine into
  `multimodal_gen`), or accept a long-lived friendly fork? This needs an explicit owner-level decision by Phase 2.*

### 11.2 Multi-engine DAG composition only (vllm-omni / sglang-omni model)

- **Pros:** fastest route to separable omni (thinker/talker/vocoder); engine reuse; process isolation.
- **Cons:** expresses single-weights hybrids (Cosmos3 MoT, BAGEL lineage) only as **one opaque monolithic stage**
  — vllm-omni's `bagel`/`lance` *are* real shared-weight MoT, but request-scheduled (`max_num_running_reqs=1`),
  batch-incapable, with the AR+denoise interleaving invisible to the scheduler — so the runtime can't
  schedule/batch/preempt the loops inside; cross-stage KV is a copy, not a shared live cache; weight duplication
  for backbones split across *separate* stages; connector latency on the token path.
- **Decision:** adopt the *composition concepts* (declarative spec, connectors, per-stage schedulers, async_chunk
  streaming) **inside** one runtime, and offer external-engine nodes as an opt-in — rather than making cross-engine
  composition the foundation.

### 11.3 Status quo plus per-model monolithic stages

The Cosmos3 port proved a skilled engineer can ship an omni model inside today's abstractions in days. Why change?

- **Pros:** zero migration risk; maximum per-model freedom; no runtime tax.
- **Cons:** it buys exactly one request at a time with no KV cache — reasoning KV, batching, streaming, and prefix
  reuse remain unbuilt (this document's assessment; the port's `PORT_STATUS.md` records 150/150 parity and makes no
  production-gap claims of its own); every future omni/AR model re-pays the monolith cost; serving never
  materializes.
- **Decision:** rejected as an end state; embraced as a migration mechanism (the legacy adapter *is* this option,
  scoped per-family and temporary).

### 11.4 Big-bang rewrite vs incremental

A clean-room runtime would be simpler to design but violates G5 and repeats the two-training-stacks situation at
larger scale. **Decision:** incremental with phase gates and an explicit Phase-5 deletion milestone. The known risk
of "incremental" — the old path never dies — is mitigated by making each phase delete/freeze something.

### 11.5 Typed dataflow vs blackboard state

Full typed ports everywhere (every tensor an edge) is maximal safety but punishing verbosity for 20+ families;
the pure blackboard is what produced P3. **Decision:** typed artifacts on graph edges + typed per-model state
extensions inside nodes. Cost: two places state can live; mitigated by a lint rule — anything crossing a node
boundary must be an edge artifact.

### 11.6 Scheduler granularity

Request-level scheduling (multimodal_gen) is far simpler; step-level adds queue/dispatch complexity and an
SPMD-consistency obligation (decisions made on rank 0, broadcast — same discipline sglang and multimodal_gen
already use). **Decision:** step-level only where loops exist; one-shot nodes schedule at request granularity.
Without step-level scheduling there is no AR batching, no interleaving, no preemption, no MoT mode multiplexing —
it is the price of G1/G4. **The falsifier:** the Phase-2 duty-cycle study publishes a load profile and targets;
if at the Phase-3 gate step-level scheduling does not beat a request-level baseline on those numbers (≥2 concurrent
sessions per GPU, p95 segment latency within the study's SLO), the StepScheduler is descoped to request granularity
and the loop contract keeps only its streaming/preemption seams — the decision is measured, not assumed.

### 11.7 Cross-request batching for video diffusion

Batched video denoise (shape-bucketed or Cosmos-style token-budget packing) is real throughput but significant
complexity (ragged attention, per-sample steps, CFG alignment). **Decision:** defer; v1 video concurrency =
interleaving + parallelism. Revisit with data after Phase 2 (the StepScheduler interface already permits
multi-request denoise batches, so this is additive).

### 11.8 Build the KV manager vs embed sglang srt

srt's pools are excellent but token-centric and entangled with its scheduler/attention stack; our second cache class
(chunked causal-video KV with sink/local windows, session lifetimes) has no srt analog. **Decision:** native
implementation borrowing srt's allocator design; no radix prefix tree in v1 (content-hash embedding cache covers the
common reuse), reconsider radix when omni chat traffic justifies it.

### 11.9 Fleet layer: Dynamo as first-class partner vs alternatives

- **Build our own router/autoscaler:** rejected outright — it is a separate engineering discipline (Dynamo is a
  Rust codebase with etcd/NATS discovery, radix-indexed KV routing, an SLA planner, KVBM tiering, ModelExpress
  cold-start streaming), and every line of it we wrote would be a line not spent on the §8 substrate.
- **sglang-router only:** lighter, and remains a fine drop-in for simple single-model deployments — but it has no
  planner, no disagg coordination, no multi-tier cache story.
- **Kubernetes-only (HPA + service mesh):** no request-content awareness — no affinity routing, no SLA planner
  with per-request cost models, no KV/disagg coordination.
- **Decision: Dynamo first-class** (contract in §6.3.5). FastVideo already runs as a Dynamo `ModelType.Videos`
  backend (`dynamo/examples/diffusers/worker.py`), and Dynamo already fronts sglang-diffusion and vllm-omni — the
  fleet layer has standardized around exactly our ecosystem. Risks, honestly: Dynamo is NVIDIA-governed and its
  router is token-prefix-centric today (our model/session-affinity routing is a collaboration item, open question
  15). Because the relationship with the Dynamo team is direct, gaps are handled as upstream asks with stated
  fallbacks (§6.3.6) rather than permanent workarounds. Mitigation regardless: the contract is deliberately thin
  and OpenAI-shaped, so the engine stays deployable bare, behind sglang-router, or under any future orchestrator
  without rework.

### 11.10 Explicit extension points vs monkeypatching (the cache-dit question)

cache-dit earns one-line adoption on diffusers (`cache_dit.enable_cache(pipe)`) by patching pipeline/transformer
internals at runtime. We could allow plugins the same freedom.

- **Pros:** zero model-side convention; instant compatibility with third-party techniques as published.
- **Cons:** patches break under `torch.compile`/cudagraph capture and TP-sharded modules; state lands in
  module-level globals (unsafe under concurrent requests — disqualifying for a serving engine); no pre-flight
  conflict detection between plugins; silent breakage whenever model internals shift.
- **Decision:** explicit, versioned hook points (§6.4) plus the `run_blocks` convention for in-tree models.
  Module-level *forward hooks* remain available — but only to read-only observers, where the patching risks don't
  apply. For cache-dit specifically: **depend on the library, not its patching path** — drive cache-dit's
  configs/calibrators/skip math from behind our `BlockInterceptor`. sglang's integration is the precedent and the
  cautionary tale at once: it adopts cache-dit via `BlockAdapter`, then has to monkeypatch
  `CachedContextManager.similarity` for sequence-parallel tensors and hand-call `refresh_context_*` per request —
  exactly the two things (group-aware comparison, request-scoped lifecycle) our interceptor contract provides
  structurally (§6.4).

### 11.11 Adopt `composable_parallel` (RFC #4084) as a dependency vs native implementation

- **Pros of adopting:** shared maintenance; framework-agnostic by design (the RFC names SGLang as a second
  backend); the spec/validation/fake-backend machinery arrives prebuilt.
- **Cons:** the RFC is deliberately Option-A *descriptive* — runtime routing stays with the host engine, and the
  schemes we need most (per-node placement meshes, MoT pathway-sharded TP, chunk-KV-aware CP) are exactly the parts
  deferred to follow-up RFCs; it is an open proposal, not shipped code; we already own a working
  `parallel_state.py` and would be coupling our roadmap to a younger external contract.
- **Decision:** borrow the design — stacked axis specs, per-axis ownership markers, pre-flight validation,
  CPU-testable composition — and implement natively, keeping our spec schema close enough to theirs that
  converging on `composable_parallel` later stays cheap if it matures.

### 11.12 RL: single runtime vs hybrid-engine (verl-omni) vs dual-stack-with-corrections (miles)

- **Hybrid-engine (verl-omni):** reuse best-of-breed engines per role. For FastVideo this would mean
  re-implementing *our own models* inside vLLM-Omni to roll them out — verl-omni literally carries a Wan2.2 rollout
  re-implementation — then importing the correction machinery to patch the numerics. Absurd given we own an engine.
- **Dual-stack with corrections (miles):** the right call at ≥70B MoE scale where Megatron is non-negotiable; its
  mismatch toolkit is world-class — but the toolkit exists to dress a wound we do not have to open at 1–30B scale.
- **Decision:** single runtime, two execution profiles, explicit consistency ladder (§8.5). Crucially, we adopt
  miles' corrections (TIS/MIS) and Behavior Record replay as **built-in C0 degradation modes** rather than
  pretending mismatch cannot occur — profiles will legitimately differ (quantized rollout, cache-dit on), and if a
  Megatron-class trainer ever becomes necessary, the same toolkit is the escape hatch (§8.4 boundary condition).
  And the diffusion-specific clincher (§8.3): inference optimization here *is* post-training — an engine that
  cannot run inside training loops is cut off from its own optimization frontier, so single-runtime is less
  optional for us than it would be for an LLM stack.

### 11.13 ComfyUI: embed, compile, or compete

- **Embed only** (custom nodes inside ComfyUI — exists today in `comfyui/`): cheapest distribution; but FastVideo
  stays trapped under their queue (no batching, no sessions, no multi-tenancy) and the performance story is capped
  by their runtime.
- **Compete head-on** (rebuild the node editor + ecosystem): rejected — 116K stars of network effects and a
  thousand-pack custom-node ecosystem are not assailable frontally, and the graph-editor UX is precisely the layer
  the market (including ComfyUI itself, via App Mode and blueprints) is abstracting away.
- **Compile** (clean-room workflow compiler onto the engine + accelerated cloud): the leverage play — it converts
  their ecosystem into our demand without forking their UX battle, and its differentiation (credibly substituting
  faster recipes, with equivalence reports) is the one move an orchestrator cannot copy (§9.4).
- **Decision: all three, sequenced and bounded.** A = funnel (maintain, add a FastVideo-Cloud API node); B = core
  bet (Phase 3 MVP over the tier-1 vocabulary + weight-fleet cache; never chase the custom-node long tail —
  tiered coverage with explicit rejection, sidecar only if demand proves it); C = Dreamverse Studio as the
  destination (workflows-as-products, real-time as the wedge) — an evolution of §9.1–9.3, not a separate product.
  Scope discipline: B's vocabulary is a *product* decision reviewed per tier, not an open-ended compatibility
  promise.

---

## 12. Risks, concerns, and open questions

1. **Complexity creep.** Three planes, policies, graphs, the cache manager (per-class KV pools + feature + weight-fleet),
   an extension system, a workflow compiler, and an RL layer — and the scope has *grown* with every
   review pass of this document, which is itself the warning. *Mitigations:* phase gates with deletion milestones;
   the additions are sequenced, not front-loaded (fleet cache + compiler in Phase 3, RL hardening in Phase 4, C2
   kernels only where they pay); the offline path stays a ~hundred-line trace (request → linear graph → loops) that
   a new contributor can read top-to-bottom; policies are plain objects, not a plugin framework.
2. **Hot-loop overhead / compile friendliness.** Per-step dispatch, policy indirection, observer/interceptor
   chains, and emit handling must not cost milliseconds; cudagraph/compile capture needs stable step bodies.
   *Mitigations:* policy *bindings* resolved at build time (policy *state* is per-request in `LoopState`, §6.2.3);
   hook chains assembled at build and literally
   absent when unused (§6.4), with `needs_eager`/`graph_safe` declarations scoping any capture loss; `step()`
   bodies are the same tensor programs as today's loop bodies; capture per (node, shape-bucket, branch-profile),
   with breakable CUDA graphs as the optimization tier (experimental upstream — the eager path is the correctness
   guarantee, §6.3.3); Phase-2 gate includes a ≤2% batch-of-1 latency budget, itself gated on the Phase-1
   loop-inversion overhead measurement.
3. **Always-on loop inversion has only narrow precedent — not the hybrid slice.** The one prior art is
   vllm-omni's `SupportsStepExecution` (runtime-owned diffusion iteration at step granularity, but opt-in,
   Qwen-Image-only, off in every deploy — §6.2.2); no surveyed system makes it the always-on default (vLLM/sglang
   own AR token loops; diffusers' loop blocks own their own iteration — §6.2.3), and the hybrid AR+denoise
   multiplexing on shared weights is the deepest end of a bet that is novel from its
   first step. *Mitigations:* the Phase-1 parity gates are the load-bearing control (old loop vs new loop,
   bit-identical, per family — this is where the novelty risk is actually retired); the Phase-1 overhead
   measurement gates the latency story; the Cosmos3 port's 150-test bit-exact suite is the hybrid-slice prototype
   harness, exercised early in Phase 4 before generalizing.
4. **SPMD-consistent scheduling.** All ranks of a pool must agree on every step decision; divergence deadlocks NCCL.
   *Mitigations:* rank-0-decides + broadcast (existing pattern in multiproc executor and multimodal_gen); decision
   inputs restricted to broadcast state; a debug mode asserts cross-rank decision hashes. The same channel carries
   the **abort broadcast** (§6.3.1) — scheduling and failure isolation share one consistency mechanism by design.
5. **Two stacks during migration** (the very smell we're fixing). *Mitigations:* legacy adapter means zero
   dual-maintained models — a family is either unmigrated (old code, untouched) or migrated (old code deleted);
   per-family checklist tracked in `.agents/`; Phase 5 is a scheduled deletion, not an aspiration. The substrate
   split (§8) raises the stakes: once `train/` methods consume the shared loops, a loop change breaks both worlds —
   so training rollouts join the parity gates (trajectory-identical before/after) and substrate PRs run training
   smoke tests, not just SSIM. The training-side gate layer already exists in CI to extend, not invent: #1396's
   per-method, device-keyed layer-0 grad-norm regression refs (`fastvideo/tests/train/methods/`, Modal-wired) —
   coarse 10%-rtol tripwires today, the natural carrier for the per-step parity assertions as methods migrate.
6. **CI/GPU cost.** Step-parity harnesses, serving load tests, SSIM suites, Cosmos3 parity on B200s. *Mitigations:*
   step-level parity tests run on CPU/tiny configs (the Cosmos3 oracle methodology generalizes); GPU suites stay
   per-family and gated to touched families — except substrate-wide PRs in Phases 0–2, where "touched" means all
   of them and the full matrix runs as a budgeted cost (G5).
7. **Relationship with sglang `multimodal_gen`** (§11.1) — needs an explicit decision by Phase 2: upstream,
   friendly fork, or shared core package. Drift here is a strategic cost regardless of architecture.
8. **Single runtime does not equal automatic numerical parity.** Two execution profiles remain
   (grad/checkpointed/FSDP-gathered training forward vs no-grad/graphed/sharded/cached rollout forward), and the
   failure mode is precisely characterized: kernels are deterministic per fixed batch but **not batch-invariant**
   (Thinking Machines' analysis — reduction strategies change with batch/shape/sharding), so profile differences
   alone break bit-equality; miles needed FA3 + DeepGEMM batch-invariant kernels to reach bitwise parity even
   *within* matched stacks. *Mitigations:* the consistency ladder (§8.5) makes the level explicit per run instead
   of implied; C1 is gated by ParityAligner drift CI (log-prob diff for likelihood-based methods, seeded
   final-latent/activation parity for NFT-class); TIS/MIS and the Behavior Record ship as built-in C0
   degradation; a batch-invariant `deterministic` kernel mode (op-swap, §8.5 C2) is scoped for goldens rather than
   promised everywhere; drift metrics (`logprob_diff` and IS clip fraction where the objective has them,
   prediction-space deviations like #1450's `old_deviate`/`old_kl_div` otherwise) are always-on training
   telemetry, not post-mortem tools.
9. **Open: session semantics.** KV/cache lifetimes beyond a single request — interactive world models (game
   sessions), multi-turn omni chat, and now Dreamverse scene continuation (chained-segment frame/audio conditioning,
   §9.1), which makes this a production requirement with a concrete customer rather than a research nicety —
   owner: Phase 4 design review, with Dreamverse's protocol as the reference workload.
10. **Open: speculative decoding for AR stages** (sglang-omni lacks it; vllm-omni inherits vLLM's). The
    `ARDecodeLoop.step` contract should permit multi-token bundles from day one even if drafting lands later —
    miles already ships speculative rollout with online-SFT'd draft models, evidence this matters for RL
    throughput too. The contract shape to copy is vLLM v1's: draft tokens ride the scheduler output and are
    rejection-sampled in the same step (§6.3.1).
11. **Open: quantization/FP8/FP4 interaction** with multi-pathway layers and the Attn-QAT work (linear/MLP FP4 paths
    recently landed) — verify policy/layer factoring doesn't block per-pathway quantization configs. The
    ParityAligner observer (§6.4) should become the standard harness for validating precision changes, and miles'
    unified-FP8 lesson applies directly: rollout and trainer share one `PrecisionPolicy`/quant recipe —
    "train BF16, roll out FP8" is only legal at consistency level C0 (§8.5).
12. **Open: hook-point surface stability.** Which hook points are public/versioned vs internal, and whether the
    cache-dit adapter lives in-tree or upstream (vipshop/cache-dit already ships adapters for other frameworks) —
    decide when the first external plugin lands (Phase 1–2).
13. **Open (narrowed): heavyweight reward-model packaging.** The original concern — the #1438 draft vendored the
    full HPSv3/VideoAlign repos, trainers and deepspeed configs included — was resolved before landing: #1450 ships
    only thin PickScore/CLIP scorers loading weights from HF, and the `add-reward-model` skill now governs
    additions. What remains open is the heavyweight case: when HPSv3/VideoAlign/VLM-judge-class rewards land, do
    they arrive as pip dependencies, pruned runtime-only ports, or engine components scored off-rank (§8.7)? The
    landed pattern (GPU-resident scorer on every training rank) does not survive reward models with their own
    multi-GB footprints.
14. **Open: workflow-compatibility tiering.** Which ComfyUI node vocabulary each tier covers, what the coverage
    SLA is, whether/when the headless-ComfyUI sidecar for unsupported subgraphs is worth its operational cost, and
    how exact-vs-fast equivalence is reported to users (§9.4) — decide with Phase 3 data from the top community
    workflows.
15. **Open: which Dynamo asks land upstream, and when.** §6.3.6 enumerates seven (A1 affinity key spaces, A2 cost
    interface, A3 media streaming, A4 role-graph disagg, A5 RL weight plane, A6 KVBM generalization, A7 sessions),
    each with a fallback. Sequence them with the Dynamo team against our phases (A2/A3 → Phase 2, A1/A4 → Phase 3,
    A5–A7 → Phase 4); commit fallbacks into the plan only for asks that miss their phase. Ask-sequencing needs an
    owner and decision dates at the design review — and the A3 fallback (direct-WebSocket bypass) must be treated
    as temporary by policy, or it hardens into exactly the permanent workaround §11.9 claims the relationship
    avoids.
16. **Open: the omni-RL objective** (§8.7). Three named problems before any hybrid pilot: per-segment
    normalization across O(1–10)-nat token log-probs vs 10⁶–10⁷-dim Gaussian step densities; credit assignment
    across the non-differentiable sampled-token boundary into shared trunk weights; reasoning-regression control
    (reference-KL anchoring for the und pathway). Owner: the RL lead, before Phase 4 scopes its deliverable.

### Known gaps — acknowledged, deliberately not yet designed

Named here so their absence reads as a decision, not an oversight. Each needs an owner and a design note before
the phase it gates:

- **Generation-quality evaluation system** (gates Phase 2/3 product claims). SSIM gates and ParityAligner cover
  regression and numerical parity; they do not measure *quality*. "Fast mode is visually equivalent" (§9.4), RL
  reward validation (§8.7), and distillation-recipe comparisons all need a real eval harness — VBench-class
  benchmarks, reward-model scoring, human-preference sampling — with tracked baselines per model family.
- **The flywheel's data plane** (gates §9.2 → §8 RL). Preference events from Dreamverse need schema, storage,
  lineage, filtering, and a consent/licensing policy before any RL method may train on them. The design covers
  trajectory capture, not the data engineering or governance around it.
- **Tenancy, quotas, metering, and input hardening for the cloud API** (gates the §9.4 business). The engine emits
  per-request cost; who meters, bills, and rate-limits is undesigned. Workflow JSON is untrusted input — resource
  ceilings (resolution/steps/duration), schema validation, and abuse controls are required at the boundary.
- **Artifact/model registry for (recipe, runtime) outputs.** Distilled/QAT checkpoints are products; their
  versioning, distribution (HF/S3/ModelExpress), provenance back to recipes, and license posture as derivatives of
  differently-licensed bases are unspecified — interacts with the weight-fleet cache (§9.4) and A5 (§6.3.6).
- **Long-job resilience beyond the pool.** In-pool failure isolation, cancellation, partial artifacts, and
  preemption are now designed (§6.3.1); what remains undesigned is durability *across* engine restarts (where
  serialized `LoopState` persists, retention policy) and cross-pool job migration.
- **Behavior Record at training scale.** Full MoE-routing capture for Cosmos3-class requests is GB-per-sample
  (§8.5); if it is ever used beyond goldens/debugging, the streaming/storage/transport design for GB-scale
  trajectories from disaggregated rollout fleets does not exist.
- **One observability schema.** Drift telemetry (§8.5), profiler output (§6.4), cost metrics (§6.3.5), and engine
  stats are each named where they arise; Phase 2 should consolidate them into a single telemetry spec.

---

## Appendix A — Source material

- FastVideo main: `fastvideo/pipelines/` (`composed_pipeline_base.py`, `stages/base.py:29`,
  `stages/denoising.py:237`, `pipeline_batch_info.py:238`), `fastvideo/worker/`, `fastvideo/entrypoints/`,
  `fastvideo/models/dits/causal_wanvideo.py:80-143`.
- Cosmos3 port: the full-omni branch is
  https://github.com/hao-ai-lab/FastVideo/tree/feat/cosmos3-reasoning — top of the stacked chain
  (`feat/cosmos3-tier-a-port` → `-i2v` → `-audio` → `-action` → `-reasoning`), covering T2V/I2V/T2I, joint
  video+stereo-audio (t2vs), **action** (domain-aware pathway, joint vision+action+sound denoise), and greedy
  text reasoning — 150/150 parity tests vs the official framework. Local worktree:
  `/home/william5lin/FastVideo_cosmos3_port`; status doc `tests/local_tests/cosmos3/PORT_STATUS.md`.
- Official Cosmos3: `cosmos-framework/cosmos_framework/model/vfm/` (`mot/unified_mot.py`,
  `mot/cosmos3_vfm_network.py`, `omni_mot_model.py`, `data/vfm/sequence_packing.py`, `inference/inference.py`).
- sglang: `~/sglang/python/sglang/multimodal_gen/` (runtime, disaggregation, pipelines_core) and
  `~/sglang/python/sglang/srt/` (managers, mem_cache, model_executor, disaggregation).
- vllm-omni: `~/vllm-omni/vllm_omni/` (`config/stage_config.py`, `engine/orchestrator.py`,
  `distributed/omni_connectors/`, `diffusion/diffusion_engine.py`).
- sglang-omni: `~/sglang-omni/sglang_omni/` (`config/schema.py`, `pipeline/`, `relay/`, `scheduling/`).
- NVIDIA Dynamo: `dynamo/` (`examples/diffusers/worker.py` — the existing FastVideo video worker;
  `lib/bindings/python/` (`register_llm`, `@dynamo_endpoint`, `DistributedRuntime`);
  `lib/kv-router/src/protocols.rs` (KV/affinity event format); `components/src/dynamo/sglang/init_diffusion.py`
  and `components/src/dynamo/vllm/omni/` (sibling-stack backends); `components/src/dynamo/planner/` (SLA
  autoscaler); `lib/kvbm-*` (multi-tier KV offload); `docs/design-docs/disagg-serving.md`). Contract analysis in
  §6.3.5.
- vLLM-Omni RFC #4084 "Composable Parallel Strategies": https://github.com/vllm-project/vllm-omni/issues/4084
  (open proposal by @kushanam, 2026-06-03; evaluated in §5, §6.3.4, §11.11).
- cache-dit (training-free DiT cache acceleration: DBCache/FBCache/TaylorSeer): https://github.com/vipshop/cache-dit
  (integration model evaluated in §11.10).
- Existing FastVideo hook seeds: `fastvideo/hooks/` (`hooks.py` — `ModuleHookManager`/`ForwardHook`;
  `activation_trace.py` — env-gated parity tracing with `trace_step`; `layerwise_offload.py`), and the dormant
  `enable_teacache` flag (`fastvideo/pipelines/pipeline_batch_info.py:182`).
- sglang's cache-dit integration: `~/sglang/python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`
  (`enable_cache_on_transformer`/`_dual_transformer`, `_patch_cache_dit_similarity`, `refresh_context_*`;
  `cache_dit_config` in `runtime/server_args.py:116`).
- Training stacks: `fastvideo/train/` (`trainer.py`, `methods/base.py`,
  `methods/distribution_matching/{dmd2,self_forcing}.py`, `methods/rl/` (see below), `models/base.py`,
  `callbacks/validation.py`, `utils/training_config.py`), legacy `fastvideo/training/*_pipeline.py`, preprocessing
  `fastvideo/pipelines/preprocess/` (incl. `preprocess_pipeline_ode_trajectory.py`).
- RL (landed): PR #1450 "[feat] Add Wan RL DiffusionNFT training" (commit `5854aec2c`, 2026-06-11):
  https://github.com/hao-ai-lab/FastVideo/pull/1450 — `fastvideo/train/methods/rl/{__init__.py, diffusion_nft.py,
  common/{__init__,prompt_sampling,sampling,validation}.py, rewards/{__init__ (the`build_multi_reward_scorer`
  factory), frame_rewards, media}.py}`, trainer hooks
  (`manages_optimization`/`managed_train_step`/`on_validation_begin`), `ModelBase.decode_latents`,
  `examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml`, and
  `.agents/skills/{add-rl-method,add-reward-model,rlhf-training-abstractions}/SKILL.md`. (Supersedes the #1438
  "Adam/dnft" draft cited in earlier revisions of this doc.)
- Training-side regression gates (landed): PR #1396 (commit `633d39356`) — per-method, device-keyed layer-0
  grad-norm regression refs: `fastvideo/tests/train/methods/{grad_norm_regression.py, grad_norm_refs.json}`,
  Modal wiring in `fastvideo/tests/modal/pr_test.py::run_train_framework_tests`.
- Typed API in flight: `fastvideo/api/` (`schema.py`, `sampling_param.py`, `compat.py` (651 lines), `presets.py`,
  `results.py`, `request_metadata.py`, per-model `matrixgame2.py`/`matrixgame3.py`).
- verl-omni: `~/verl-omni/verl_omni/` (`trainer/diffusion/{ray_diffusion_trainer,rollout_correction,
  diffusion_algos}.py`, `pipelines/wan22_dance_grpo/vllm_omni_rollout_adapter.py`,
  `pipelines/schedulers/flow_match_sde.py`, `workers/rollout/vllm_rollout/vllm_omni_async_server.py`,
  `workers/engine/{fsdp,veomni}/`, `reward_loop/reward_manager/multi.py`).
- miles: `~/miles/` (`train.py`, `train_async.py`, `miles/backends/training_utils/loss_hub/{losses,corrections}.py`
  (TIS/MIS), `miles/backends/megatron_utils/update_weight/` (IPC/broadcast/P2P transports),
  `miles/backends/megatron_utils/replay_utils.py` (R3), `miles/utils/types.py` (`Sample` with
  `rollout_routed_experts`, `weight_versions`), `miles/rollout/base_types.py` (rollout-fn contract),
  `docs/en/advanced/{miles-router,fp8-low-precision}.md`).
- ComfyUI: `~/ComfyUI` (GPL-3.0; `execution.py` + `comfy_execution/caching.py` (graph execution, output caching),
  `nodes.py` (64 core nodes), `comfy_extras/nodes_{wan,cosmos,hunyuan,mochi,video_model}.py` (in-core video),
  `comfy/{model_management,model_patcher,lora,multigpu}.py`, `comfy_api_nodes/` (39 paid API providers),
  `blueprints/` (80 in-tree workflow templates; more via the registry), `QUANTIZATION.md`); FastVideo's existing custom-node suite: `comfyui/`
  (this repo).
- Dreamverse: `apps/dreamverse/` (`arch.md`, `design.md`, `AGENTS.md`; runtime `dreamverse/{main,gpu_pool,
  prompt_enhancer,session_init_image}.py`; `web/` Next.js frontend; `serve_configs/streaming_demo.yaml`); blogs:
  https://haoailab.com/blogs/dreamverse/ (vibe directing) and https://haoailab.com/blogs/fastvideo-dreamverse-release/
  (release; NVFP4/FA4/compile on B200, fMP4-over-WebSocket, frame/audio segment conditioning).
- xDiT: `~/xDiT/xfuser/` (`config/config.py` — ParallelConfig axes; `core/distributed/{parallel_state,
  group_coordinator,runtime_state}.py`; `docs/methods/{pipefusion,usp,hybrid}.md`; DistVAE adapters in
  `model_executor/models/runner_models/wan.py`).
- cosmos-rl: `~/cosmos-rl/cosmos_rl/` (`dispatcher/{controller,command}.py`, `policy/trainer/llm_trainer/
  grpo_trainer.py` (AIPO/dual-clip/off-policy masking), `rollout/{rollout_base,worker/weight_sync}.py`
  (RolloutBase registry; buffer-model async sync), `rollout/wfm_rollout/`, `utils/{distributed,parallelism_map}.py`).
- diffusers Modular: `~/diffusers/src/diffusers/modular_pipelines/` (`modular_pipeline.py` —
  `LoopSequentialPipelineBlocks` line ~1294, `PipelineState`/`BlockState`; `modular_pipeline_utils.py` —
  `ComponentSpec`; `components_manager.py`) and `src/diffusers/guiders/` (`guider_utils.py` — `BaseGuidance`).
- torchtitan: `~/torchtitan/torchtitan/` (`distributed/parallel_dims.py` — DeviceMesh builder + validation;
  `distributed/{fsdp,activation_checkpoint}.py`; `components/checkpoint.py` — async DCP;
  `experiments/rl/actors/{trainer,generator}.py` — TorchStore weight sync; batch-invariance utils in
  `distributed/utils.py`).
- LiveKit agents: `~/agents/livekit-agents/livekit/agents/` (`voice/avatar/_runner.py` — `AvatarRunner`,
  `AVSynchronizer`; `voice/avatar/_datastream_io.py` — data-stream/RPC control;
  `examples/primitives/video-publisher.py` — `rtc.VideoSource.capture_frame`).
- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (He et al., 2025):
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ — batch-invariance analysis and the
  true-on-policy (KL ≡ 0) RL result underpinning consistency level C2 (§8.5); companion library
  `thinking-machines-lab/batch_invariant_ops` (torch.Library op-swap, FlexAttention-based deterministic vLLM mode).
- vLLM core: `~/vllm` (pulled 2026-06-10) + RFC #42770 "Changes in vLLM Model Development"
  (https://github.com/vllm-project/vllm/issues/42770, Woosuk Kwon, 2026-05-15; discussion incl. the compile lead's
  dissent and the two-tier head/tail resolution of 2026-05-28); linked: PR #42304 (breakable CUDA graph,
  `VLLM_USE_BREAKABLE_CUDAGRAPH`), #43224 (porting compiler fusions to manual call sites), #24384 (HF-decoupled
  config), `vllm/v1/worker/gpu/model_states/interface.py` (`ModelState`). FastVideo's inherited
  `fastvideo/forward_context.py` is the in-repo instance of the hack this RFC deletes (staged retirement — 194
  references across ~50 files; new-path StageContext in Phase 1, global frozen for legacy consumers, deletion at
  Phase 5; §6.3.3).
  v1 code studied in depth: `vllm/v1/engine/{async_llm,core,core_client,coordinator}.py` (process topology, DP
  waves), `vllm/v1/core/sched/{scheduler,output}.py` (step-budget scheduling, in-step encoder budget, delta
  outputs), `vllm/v1/core/{kv_cache_manager,kv_cache_coordinator,block_pool,encoder_cache_manager}.py`
  (hybrid KV groups over one BlockPool; refcounted encoder cache), `vllm/v1/worker/gpu/{input_batch,states}.py`
  (persistent batch, staged writes), `vllm/v1/cudagraph_dispatcher.py`,
  `vllm/distributed/kv_transfer/kv_connector/v1/base.py` (`KVConnectorBase_V1`), `vllm/lora/` +
  `punica_wrapper/` (batched multi-LoRA), `vllm/device_allocator/cumem.py` (tag-based sleep).
