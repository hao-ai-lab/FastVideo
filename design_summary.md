# FastVideo Next-Gen Runtime — Two-Page Summary

**Companion to** `design.md` (v19, 2026-06-12) · **Status:** draft for discussion · **Ask:** read this, then dive into the sections you own.

---

## The problem

FastVideo's pipeline abstraction has been outgrown by its own model zoo. Four facts, all on `main` today:

- **The denoise/sampling loop exists in four copies** — inference stages (`pipelines/stages/denoising.py`, a 1,381-line file), `train/` distillation methods, legacy `training/` monoliths, and the just-landed RL work. The fourth copy documents the cause in its own docstring: `DiffusionSampler` (PR #1450) *"intentionally does not call FastVideo's full inference pipelines"* because the only consumable units are family-bound pipeline classes. Every new post-training method must pick between a wrong dependency and a private loop.
- **There is no serving runtime.** One request at a time, no queue, no cross-request batching. Dreamverse — our shipping product — hand-rolls its own GPU pool, queue, warmup, and streaming relay at a cost of **one B200 per user session**.
- **Cosmos3 outgrows both the stage abstraction and the alternatives.** Its AR text reasoner and multimodal diffusion denoiser are *the same resident weights* driven by two loop types within one request — our full port (incl. action modality, on `feat/cosmos3-reasoning`) runs it only by bypassing stages with one monolithic block. Multi-engine DAG stacks (vllm-omni, sglang-omni) compose separable stages with disjoint weights; none can express a mixture-of-transformers model. This is the forcing function.
- **RL has arrived and pays the tax already** — likelihood-free DiffusionNFT for Wan (#1450): in-process rollouts with zero serving-grade optimizations (no CFG, dense attention, full 25-step ODE), a vendored sampler, and a parallel validation path built *because* inference pipelines aren't consumable as a library.

## The design

```
Request plane     OpenAI-compatible server (videos/images/audio/chat) · AsyncEngine
                  OmniRequest ─► admission ─► queue ─► OmniOutput (typed modality parts)
Pipeline plane    PipelineSpec: declarative graph per family
                  nodes: Stage | LoopStage (DenoiseLoop, ARDecodeLoop) · typed Artifacts
                  policies: CFG, ExpertRouting, AttnMetadata, Precision, FlowShift
Execution plane   StepScheduler (multiplexes denoise + AR steps) · worker pools (TP/SP + CFG-parallel)
                  CacheManager (paged text-KV ▪ chunked causal-video KV ▪ feature caches) · connectors
```

Default deployment is exactly today's: one SPMD pool, co-located nodes, synchronous call. Serving is additive configuration, not a different code path. Five load-bearing decisions:

1. **Loop inversion.** Loops become `LoopStage` nodes exposing `init / step / finalize`; the runtime owns iteration, families own step bodies (with a custom-step escape hatch — the runtime never dictates step factoring). This is the enabler for step-level scheduling, streaming, MoT interleaving, and one shared loop across inference, distillation, and RL. Stated honestly: **no surveyed system does this at scheduler granularity** — the risk is retired by Phase-1 bit-identical parity gates and a measured falsifier, not borrowed validation.
2. **Cost-model scheduling.** Denoise steps and AR tokens are incommensurable (bidirectional attention is O(L²) per step with zero KV amortization; steps differ ~1000×) — the budget currency is **predicted GPU-time** from a per-(model, phase, shape) cost model, calibrated by the profiler and published to Dynamo's router/Planner as the same artifact.
3. **One substrate for inference, training, and RL** — models, loaders, configs, schedulers, parallel state, loop step bodies — under a strict `engine never imports train` rule. Trainers keep their internals; their embedded sampling paths migrate onto the shared loops.
4. **Consistency is a declared, measured contract**, not a hope: **C0** corrected (profiles differ, TIS/MIS fixes it) / **C1** kernel-pinned (RL default; drift gated in CI) / **C2** bitwise (batch-invariant kernels + Behavior Record, for goldens and MoE parity). One repo ≠ automatic parity — the ladder is what makes the single-runtime bet honest.
5. **Extensions, never monkeypatching**: read-only observers (ParityAligner, ActivationTrace, Profiler, NaNWatch) and compute-altering interceptors at declared points — **cache-dit** is the first interceptor. **Dynamo is the fleet layer** (first-class partner, not a dependency we rebuild): registration/health/cost contract in-engine, seven concrete upstream asks (affinity key spaces, cost interface, media streaming, role-graph disagg, RL weight plane, KVBM generalization, sessions) — each with a fallback.

## Why this is the moat

Unlike LLMs — where inference optimization is post-hoc on frozen weights — **a usable video model is itself a post-training artifact**. Every inference capability we ship is a *(recipe, runtime)* pair: step distillation ↔ few-step samplers; self-forcing ↔ causal KV streaming; QAT-NVFP4 ↔ FP4 kernels; VSA ↔ sparse attention backend; RL ↔ samplers + capture. The training loop *embeds* the inference loop, so whoever owns both sides of the pair owns the optimization frontier. The industry's RL pain proves the converse: verl-omni re-implements Wan2.2 inside vLLM-Omni and corrects the numerics afterward; miles' headline features are all mismatch patches for two runtimes with different kernels. We answer with one model definition, one kernel set, one measured ladder — at FastVideo's 1–30B FSDP2 scale, where the bet is viable.

## What's pulling on it

| Customer | Pull | Proof point |
|---|---|---|
| **Cosmos3 / omni** | MoT loops, packed sequences, reasoner KV, world-model rollout | 150-test parity suite on `feat/cosmos3-reasoning` |
| **Dreamverse** | Engine-client replaces hand-rolled pool; capacity = duty cycle + cost-model admission + distillation | today 1 B200/session; Phase-3 gate: ≥2 sessions/GPU on a recorded duty-cycle trace, p95 within SLO |
| **RL (landed)** | #1450 migrates onto shared loops (Phase 1), engine-client rollouts (Phase 2+); GRPO-class next | the vendored-sampler docstring; C1-by-construction discipline |
| **ComfyUI funnel** | embed (nodes) → **compile** (workflow→PipelineSpec, accelerated cloud) → productize (Studio) | tier-1 ~20-node static sublanguage maps onto PipelineSpec |

## Migration — seven phases, each independently shippable

| Phase | Ships | Gate |
|---|---|---|
| **−1** | Merge cosmos3 chain; seed SSIM for uncovered families | baselines exist |
| **0** | Typed omni I/O; config freeze (`compat.py` shrinks monotonically to zero) | all SSIM suites unchanged |
| **1** | **Loop inversion** + policies + extension core (cache-dit, ParityAligner); RL migrates off its vendored sampler | old vs new loop **bit-identical**; per-method grad-norm refs (#1396) extended |
| **2** | AsyncEngine + StepScheduler; LTX-2 linear graph; Dynamo worker (stock); colocated weight sync | ≤2% batch-1 latency regression; Dreamverse single-session parity; RL engine-client parity |
| **3** | PipelineSpec graphs, role pools, declarative parallelism, ComfyUI compiler MVP, general WeightSyncPlan | ≥2 Dreamverse sessions/GPU on recorded duty-cycle trace |
| **4** | Cosmos3 native; AR continuous batching + paged KV (arriving *with* their workload, per N5); RL hardening (C1/C2, Behavior Record) | Cosmos3 parity suite on new runtime; drift ≈ 0 on a Wan RL run |
| **5** | Deletion: legacy `training/` retires, then `ComposedPipelineBase`, legacy loop, `forward_context.py`, `compat.py`, `RayDistributedExecutor` | the deletion diff — **4 loop copies → 1** |

Enforcement the last freeze lacked (it was broken 19×): `compat.py` frozen from Phase 0; CI path gates + CODEOWNERS once Phase 1 lands; new families land on new abstractions from Phase-1 completion.

## What we are deliberately not doing

Datacenter orchestration (Dynamo's job) · trainer internals (frozen `training/`; `train/` is a consumer) · replacing the bit-exact porting methodology · migrating 20+ families at once · **standalone LLM-serving excellence** — AR machinery arrives only at the sophistication omni workloads pull (N5).

## Decisions we need from this review

1. **sglang `multimodal_gen` relationship** — upstream, friendly fork, or shared core (decide by Phase 2; drift is a strategic cost either way).
2. **Dynamo asks** — green-light proposing the Phase-2 asks (A2 cost interface, A3 media streaming) to the team first, with A5 (RL weight plane) queued behind them?
3. **Phase −1 start** — merge the cosmos3 chain and seed SSIM baselines now; it blocks everything else.
