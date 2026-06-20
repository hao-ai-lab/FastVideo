# FastVideo v2 — A Model-Native Runtime for the (Recipe, Runtime) Era

**Status:** source of truth. This README is the single design document for v2 — **what is actually built and
running in `v2/`** plus the forward roadmap.

**What v2 is.** A model-native serving **and** training substrate for composite multimodal models — video,
image, audio, omni/MoT, world models, VLAs — where the atomic unit is a typed **(recipe, runtime) pair**
owned by a `ModelCard` and every iterative computation is a **driven loop**. Serving is **pooled
run-to-completion**: a request takes a pool slot and runs its program (its loops) start-to-finish. The
control flow, contracts, scheduling, caching, parity gates, and training math are real and CPU-testable; the
heavy neural forwards run either on a numpy **toy backend** (laptop, no GPU) or the real **torch backend**
(`platform/backends/torch_backend.py`), selected by the `platform/` dispatch substrate with **no change to
loops/caches/parity/training**. The model + layer code is **vendored into `v2/` (zero `fastvideo` imports)**.
The kept model set is 8 families — Wan2.1, Wan-causal (self-forcing), LTX-2, Flux2, MatrixGame2, and the omni
cards Cosmos3 / BAGEL / Qwen-Omni.

---

## Table of contents
1. [The thesis](#1-the-thesis) · 2. [Three signature ideas](#2-three-signature-ideas) ·
3. [Planes & dependency order](#3-planes--dependency-order) · 4. [Model Plane](#4-model-plane--the-center) ·
5. [The driven-loop contract](#5-the-driven-loop-contract) · 6. [Runtime & scheduler](#6-runtime--scheduler) ·
7. [Memory, cache, transport, compile](#7-memory-cache-transport-compile) · 8. [Parallelism](#8-parallelism-as-a-model-contract) ·
9. [Correctness — parity as a typed gate](#9-correctness--parity-as-a-typed-gate) · 10. [Training & RL](#10-training--rl-on-the-same-loops) ·
11. [Weight-sharing topologies & stress tests](#11-weight-sharing-topologies--the-stress-test-catalog) ·
12. [Request/session/artifact + programs/workflows](#12-request-session-artifact--programsworkflows) ·
13. [Serving & fleet](#13-serving--fleet) · 14. [Extensions](#14-extensions) ·
15. [Model taxonomy](#15-the-model-taxonomy-what-the-runtime-must-serve) ·
16. [Package layout (actual)](#16-package-layout-actual) · 17. [Current status & GPU bring-up](#17-current-status--gpu-bring-up) ·
18. [Roadmap](#18-roadmap) · 19. [Reference synthesis](#19-reference-synthesis) ·
20. [Honest unknowns & falsifiers](#20-honest-unknowns--falsifiers)

---

## 1. The thesis

Three facts about video/omni generation dictate the architecture:

- **A deployable model is a post-training artifact.** Unlike an LLM (where inference optimizes frozen weights
  post-hoc), a *usable* video model is *created* by training: step distillation is mandatory for latency, low
  precision needs QAT, causal/world models are made by distillation + self-forcing. Every inference capability
  is therefore a **(recipe, runtime) pair** — the weights and the loop that produced-and-assumes them are one
  versioned object, a `ModelCard`.
- **Video/omni systems are loop systems.** The work is iteration — denoise timesteps, AR decode, chunked
  rollout, VAE tiles, encoder chunks, audio tokens, reward batches, optimizer steps, media chunks — not a
  single `forward()`. A runtime that reduces everything to `forward()` cannot schedule, batch, cancel, stream,
  reserve memory for, or capture the behavior of what actually runs.
- **Omni models share weights across loop types within one request.** Cosmos3's text reasoner and multimodal
  denoiser are the *same resident weights*, driven by an AR loop then a diffusion loop in one request. This
  cannot be a DAG of separate engines (that doubles 30B+ of weights and severs the shared KV/denoise state); it
  must be one resident instance running many loops. The hard part — the differentiation — is making those loops
  **runtime-visible and step-scheduled** (vllm-omni's `bagel_single_stage`/`lance` prove the sharing is
  *expressible*, but bury it in one opaque `DIFFUSION` stage the scheduler never sees inside).

**The one invariant, stated once:**

```
Model cards own components, loops, recipes, and parity.
Programs compose loops into tasks.   Workflows compose models into pipelines.
The scheduler executes the steps of loops as WorkUnits under one budget.
Caches are correct by key, not by hope.
Training records behavior on the same loops it serves.
Deployment places and routes; products stream artifacts; neither defines the model.
```

Everything below is the elaboration of that invariant.

---

## 2. Three signature ideas

### 2.1 The (recipe, runtime) pair is a first-class, versioned, typed object
A model is not a checkpoint. It is a `ModelCard` owning, as one versioned unit: the **components** (weights,
loaders, layouts), the **loops** it can run, the **recipe** that produced the weights (distillation/QAT/RL
method, parents, `assumes_loop`, `assumes_precision`), and the **parity contract** binding the train-forward
to the serve-forward at a declared consistency level. You cannot ship the weights without the loop they assume
(`assumes_loop`), or change the loop without re-proving parity. This turns "we do training and inference in one
repo" from an org chart into a typed guarantee.

### 2.2 Driven loops — the model owns control flow, the runtime owns execution
A loop is a serializable state machine `init → (next → advance)* → finalize`: the model describes *the next
step it needs* (`next` is kernel-free — it returns a typed `WorkPlan` thunk), the runtime decides *when and
with whom that step runs* (`await ctx.execute(plan)` — the inversion point: admission, batching, placement,
streaming, behavior capture), and the model folds the result back (`advance`) and decides what to do next.
Content-adaptive decisions (cache-dit skips, EOS, VSA tile selection) are ordinary control flow in the model;
per-request state lives in a typed `LoopState`, **never in module globals**, so concurrent requests sharing
one resident instance through the pool cannot smear state — the failure mode that makes naive loop-inversion
dangerous is *structurally* excluded.

### 2.3 One vocabulary spans every weight-sharing topology
The Card/Loop/Program split is not specialized to one sharing pattern. The **same primitives** express the
whole spread — and every one is just `shared_weight_components` bindings on `LoopSpec`s plus hand-off nodes in
a `Program`, **no new primitive**:

| Topology | Components | Loops | Sharing | Recipe |
|---|---|---|---|---|
| Single diffusion | `transformer` | `diffusion_denoise` | — | `wan21` |
| **MoT omni** | **one** `transformer` | `ar_decode` + `diffusion_denoise` | both loops → **same** component | `cosmos3`, `bagel` |
| **Joint LM+gen RL** | `llm` + `transformer` | `ar_decode`→`llm`, `diffusion_denoise`→`transformer` | **disjoint** experts, one request, jointly RL'd | `unified` |
| **Cascade omni-speech** | `thinker`+`talker`+`vocoder` | `ar_decode`→`ar_decode`→`audio_decode` | **disjoint**, **chained** | `qwen_omni` |
| **N-way joint RL** | N `refiner_i` + `transformer` | N×`ar_decode` + `diffusion_denoise` | **disjoint**, N-way jointly RL'd | `multi_expert` |

The signature claim, validated by the §11 stress tests: *the weight-sharing graph is data on the card, not
structure in the runtime.* (BAGEL's real MoT is *partial* sharing — co-resident experts sharing attention,
separate FFNs — expressed via the expert-routing policy over one resident instance; a fifth point on the same
axis, still no new primitive.)

---

## 3. Planes & dependency order

```
   Products: Python · CLI · OpenAI server · ComfyUI · Dreamverse · RTC · Trainer   (thin: validate, request, subscribe)
   Request / Session / Artifact / Stream                                           v2/core/request/, v2/runtime/session.py
   Program Plane  (typed loop programs; cross-model Workflows)                      v2/core/program/
   ┌──────────────── Model Plane (CENTER) ────────────────┐                         v2/core/card/, v2/recipes/*/card.py
   │  ModelCard: components · loops · recipe · parity ·    │
   │  capabilities · caches · parallelism · precision      │
   └───────────────────────┬───────────────────────────────┘
   ┌────────────────────────┼────────────────────────┐     same loops, different capture
   │ Runtime / Scheduler      │ Training / RL          │     v2/runtime/  ·  v2/training/
   │ WorkUnits · GPU-time budget │ rollout·reward·sync │
   └────────────────────────┼────────────────────────┘
   Memory · Cache · Transport · Compile                                             v2/runtime/{memory,cache,transport}, v2/runtime/cudagraph.py
   Parallelism (named axes → DeviceMesh, validated, part of the cache key)          v2/core/parallel/
   Platform dispatch (COMPONENTS/KERNELS registries → toy | torch backend)          v2/platform/
   Deployment / Fleet (DeploymentCard → LocalFleet | Dynamo; never the core)        v2/serving/deploy/, v2/serving/
```

**Enforced boundaries:** `card/` imports no product/runtime; `runtime/` executes `card/` loops but defines no
semantics; `training/` requires behavior records but forks no loop and **the engine never imports `training`**
(verified — the only `training` mention in `runtime/` is the comment documenting this rule); cross-model
`Workflow` orchestration sits *above* the engine (no change to the single-instance hot path). `parity/` is a
first-class package, not a test folder.

---

## 4. Model Plane — the center

```python
class ModelCard:
    model_id: str                       # "fastwan-1.3b-nvfp4-4step"
    family: str
    components: dict[str, ComponentSpec]
    loops: dict[str, LoopSpec]
    capabilities: CapabilityMatrix      # text_to_video, image_to_video, reasoning_text, vae_decode, ...
    recipe: RecipeSpec                  # what produced these weights (§2.1)
    parity: ParitySpec                  # train-forward ≡ serve-forward, to a declared level (§9)
    caches: dict[str, CacheContract]
    parallelism: ParallelismContract
    precision: PrecisionContract
```

The card is both a **declarative contract** (validatable before any GPU touches it) and a **runtime factory**
(it instantiates components, binds loops, resolves caches). `card.validate()` checks every loop's components +
cache policies exist and that `recipe.assumes_loop` is a declared loop.

- **`RecipeSpec`** — `method` (`dmd2`/`self_forcing`/`diffusion_nft`/`unified_rl`/`base`/…), `parents`
  (teacher/base ids), `assumes_loop`, `assumes_precision`, `consistency_required`. `assumes_loop` is the teeth:
  a 4-step distilled model cannot be served under a 50-step sampler without a typed mismatch error.
- **`ComponentSpec`** — `component_id`, `kind` (`dit`/`vae`/`text_encoder`/`reasoner_tower`/…), `load_id` (the
  real module to load on the torch backend), `factory` (the toy stand-in), `checkpoint` (weights path/HF id),
  `required_for`/`optional_for`/`resident_for` task sets, precision/placement policies. (Note: `required_for`
  is currently declared on every card but not yet consumed by the executor — see §18 roadmap P0.)
- **`LoopSpec`** — `loop_id`, `kind` (`LoopKind.DIFFUSION_DENOISE`/`AR_DECODE`/`CHUNK_ROLLOUT`/`AUDIO_DECODE`/…),
  `work_unit_kind`, `shared_weight_components`,
  `cache_policy`, `loop_factory`.

A `ModelInstance` is a resident, loaded card: component instances, model state, caches, compiled graphs, a
parallel plan. **A request may run several of the card's loops against one `ModelInstance`** — that single
sentence is what makes omni native, and two loops binding the same `shared_weight_components` get the *same
live object* via `instance.component()` (no weight duplication, no DAG split).

---

## 5. The driven-loop contract

```python
class Loop(Protocol):                 # v2/core/loop/contracts.py — protocols, not ABCs
    def init(self, req, model, ctx) -> LoopState         # per-request state (seeded rng, latents…)
    def next(self, st)              -> WorkPlan | Done    # describe the next step (KERNEL-FREE: a run() thunk)
    def advance(self, st, result)   -> LoopState          # fold result; capture behavior under ROLLOUT
    def finalize(self, st)          -> LoopResult          # outputs + metrics + behavior
```

The runtime's `LoopRunner` is the only place iteration lives: `next` → `await ctx.execute(plan)` → `advance`,
emitting `plan.emits` as stream chunks. Why this contract is right, against the failure modes:

- **Content-adaptive steps are natural** — `next()` reads `state` (and `advance` already folded in the last
  `StepResult`), so cache-dit's skip, AR's EOS, and VSA's tile selection are ordinary control flow; `next()`
  is still kernel-free (it *describes* work), which is all the scheduler needs.
- **Cross-request state safety is structural** — all per-request state is in `LoopState`; there are no
  module-level residual/KV globals, so concurrent pooled requests cannot smear state.
- **Serializable ⇒ resumable/migratable** — a `LoopState` is a resume point (preempt, migrate, crash-recover).

**Policies decompose the step body** (CFG/flow-shift/precision/expert-routing) — composed *into* a loop, not
branched *inside* it. **CFG is a policy over one shared denoise body** (proven by vllm-omni's `CFGParallelMixin`
unifying 2-forward / batched / cfg-parallel under one predict/combine pair), expressed as **three layers**:
(1) `CFGPolicy` is *in-loop* — branch vocabulary, combine formula, per-request mutable state (the adaptive-gate
cached delta is the canonical state case; batched-vs-2-forward is a dispatch detail inside one policy); (2)
`cfgp` is a *parallelism axis* that shards branches across ranks and runs the same rank-invariant combine;
(3) companions are an *orchestrator pattern* upstream of diffusion. Two caveats: `combine` runs in the step
body's numeric space (Cosmos combines in x0-space post-EDM, not noise-space), and embedded-guidance (Flux) is a
degenerate single-branch identity-combine policy, not "no CFG". A family whose math is genuinely braided ships
a **custom `next`/`advance`** using samplers/CFG as a *library* — the runtime requires only the four methods.

---

## 6. Runtime & scheduler

**One WorkUnit, one currency.** Every `ctx.execute(plan)` is a `WorkUnit` — the smallest schedulable action
with a resource reservation and a loop boundary. Kinds: `AR_TOKEN`, `AR_PREFILL`, `DIFFUSION_STEP`,
`DIFFUSION_WINDOW`, `CHUNK_STEP`, `ENCODER_CHUNK`, `VAE_TILE`, `AUDIO_CHUNK`, `REWARD_BATCH`, `LOGPROB_BATCH`,
`TRANSFER`, `CACHE_IO`, `GRAPH_CAPTURE` (a 13-kind taxonomy). Tokens are *one kind*, not the scheduler — the
generalization of vLLM's token scheduler that diffusion forces.

**Serving is pooled run-to-completion.** A request takes a slot in the serving pool (`AsyncEngine`, bounded by
`max_concurrent`) and runs its program to completion — its loops stepped one unit at a time by a
`ProgramRunner`. There is **no GPU-time cost model, no cross-request step-interleaving, and no batching**:
concurrency is the pool size, and the only admission gate is **memory**.

**Admission rule (the soundness condition):** before a loop step runs, reserve its memory — resident (held for
the whole loop) + worst-case peak activation. An infeasible reservation (need > pool capacity) fails fast
(`AdmissionInfeasible`); reservations are **refundable** (released on completion), so memory is a concurrency
gate, not a lifetime cap. **Cancellation is common-path** (abandoning in-flight work is normal): it takes
effect at the next step boundary, drops the request, and releases `LoopState` + cache handles + the pool slot.

---

## 7. Memory, cache, transport, compile

**Cache correctness is a contract.** `CacheKey` carries every output-semantic field — `model_id`,
`component_id`, `loop_id`, per-component `weights_version`, `adapter_versions`, `precision`,
`parallel_plan_hash`, `shape_sig`, `layout_sig`, `scheduler_sig`, `guidance_sig`, `seed`, `input_hashes`,
`step_index`, `contract_version`. **If a field can change output semantics, it is in the key.** Incorrect reuse
is worse than no reuse: the key is *partitioned* by `adapter_versions` (a te-LoRA-differing request doesn't
serve stale embeddings), and a weight-sync bumps only the affected component's `weights_version` — so a
transformer sync **does not flush the frozen text-encoder's feature cache** (a K-sample RL group encodes its
shared prompt once).

**Per-class pools (the granularity reality).** No single unified block pool — cache classes differ by 150–500×
in natural granularity (text-KV page ≈ 64 KB/layer; causal-video latent-chunk slab ≈ 9.6–32 MB/layer). Each
class gets a statically budgeted pool: paged text-KV (`ar_decode`), slab chunk-KV (`chunk_rollout`, with a
training mode that disables mid-rollout recycling), feature caches (content-hash keyed, ref-counted),
residual caches (cache-dit, scoped per `LoopState`), weight/adapter cache. **KV is the minority case** — a pure
bidirectional deployment allocates none of it.

**Memory / transport / compile.** Tagged pools with sleep/wake by tag (CuMem-style, component-granular for RL).
Transport is manifest-based and pluggable: in-proc reference → SHM → CUDA IPC → NCCL/NIXL → object-store;
KV-bearing edges speak a `KVConnector`-shaped protocol (`chunk_ready` readiness + credit-based flow control,
sglang-omni's model). Compile: CUDA graphs + `torch.compile` keyed on `(model, component, loop, work_kind,
shape_sig, precision, parallel_plan, backend)` — **never full-graph across the engine**; per-step piecewise
capture is wired (`runtime/cudagraph.py`, declared on 19 cards) with a static-buffer discipline and
version-eviction on weight sync.

---

## 8. Parallelism as a model contract

Parallelism is not a launch flag — it affects cache keys, scheduling, transport, capture, and parity, so it
lives on the card. `ParallelPlan` axes (`v2/core/parallel/plan.py`):
`("dp","tp","sp","cp","cfgp","pp_patch","vae","ep","fsdp","role","replica")`. Declarative, validated
(`validation.py`: `cfgp ≤ 2`; `pp_patch` is **invalid for causal/AR** because stale KV breaks causality;
ownership conflicts like a `BatchedCFG` policy *and* a `cfgp` group are build errors), compiled to a PyTorch
`DeviceMesh` via a `ParallelDims`-style builder. **Pre-flight or it fails at load, never halfway.** Degree-one
axes exist as trivial groups so component code needs no special cases. **Pools are single-node**; multi-node
scale is *multiple pools* fronted by the fleet (§13). Note: Wan/LTX shipped weights parallelize via **sequence
parallelism (`sp`)**, not TP (they use `ReplicatedLinear`); real `ColumnParallelLinear` lives in `flux2`.

---

## 9. Correctness — parity as a typed gate

Every card carries a `ParitySpec`. Parity is **measured, never assumed**, by a `ParityAligner` observer:
record named taps per step/block from a reference, replay with fixed seeds, report the first divergence beyond
per-tap tolerance.

**The consistency ladder:**
```
C0  component parity   — VAE/encoder/transformer-block/scheduler-step in isolation
C1  loop parity        — full denoise trajectory / AR logits, fixed seed
C2  behavioral identity — the train-forward and serve-forward agree on the quantity the RL objective uses:
       · likelihood-based (GRPO/UniRL): per-step log-prob identity ⇒ PPO ratio == 1
       · likelihood-free (DiffusionNFT): seeded final-sample + prediction-space identity — NO log-probs to match
C3  distribution parity — rollout distribution under allowed nondeterminism (defined; not yet consumed — §18)
C4  artifact quality   — SSIM/reward/human-preference (gates product claims; needs the eval system)
```
The **C2 split** is load-bearing and the lesson of the landed RL stack: the shipped DiffusionNFT is
likelihood-free (no log-probs; "log-prob identity" is undefined for it), while UniRL is likelihood-based — both
are demonstrated, on opposite halves of the rung.

**Execution-path parity.** Parity is measured, not assumed: the typed consistency ladder (C0–C3) bounds the
per-tap gap, and `compare_outputs` proves two execution paths agree **bit-for-bit** (e.g. disaggregated
serving == inline). Per-request state lives only in `LoopState` (never module globals), so it cannot smear
across requests sharing a pool. **Three execution profiles, one loop definition:** serve (no-grad, graphed,
cached), rollout (serve + behavior capture), train (grad, checkpointed) — they differ only in grad mode and
capture; the ladder measures the gap.

---

## 10. Training & RL on the same loops

```
serve   : request      → program → loop → WorkUnits → artifacts
rollout : prompt batch → program → loop → WorkUnits → BehaviorRecords → rewards → update
```

The loop kernel is shared; the only difference is capture and training policy. **This is the moat — the one
place a serving-only runtime structurally cannot follow.** The rollout forward *is* the serve forward plus
capture, so every serving optimization (distilled samplers, cache-dit skips, CFG-parallel, paged/feature
caches, step batching) is automatically a rollout optimization, and there is one numerics surface (the ladder
*measures* the gap rather than a correction layer *papering over* it). The industry's two-runtime tax
(verl-omni re-implements Wan inside vLLM-Omni + a correction layer; miles' TIS/MIS/bitwise-logprobs/R3 are
mismatch patches) is exactly what collocation deletes — viable at FastVideo's 1–30B FSDP2 scale.

- **`BehaviorRecord`** — captured at generation time: seeds, scheduler trajectory, timesteps, latents-or-refs,
  log-probs *where applicable*, sampled/action tokens, guidance, reward in/out, precision, parallel plan,
  `weights_version`. Sized honestly — an opt-in instrument for goldens, not always-on.
- **`WeightSyncPlan`** ships a **role**, not "the weights" (student / EMA / decay-blended old-policy /
  reference / teacher / critic), with a **per-component scope** so a sync versions and cache-invalidates one
  expert in isolation. Lifecycle (the RL flywheel's hardest correctness): freeze admission → drain/boundary-stop
  in-flight loops → transfer → bump version + invalidate that component's caches → resume.

Methods (a faithful CPU port — NFT is line-for-line vs the source — carrying none of the GPU/FSDP/checkpoint
infra):

| Method | Consistency | Roles | Notes |
|---|---|---|---|
| `finetune` | C1 | student | plain flow-match regression |
| `dmd2` | C2 (free) | student + fake-score critic + teacher | distribution-matching distillation |
| `diffusion_nft` | C2 (free) | student + **old** (decay-blended) + reference | samples from *old*, not student |
| `self_forcing` | C2 (free) | student + teacher | causal/chunked student |
| `unified_rl` | C2 (based) | student (llm+transformer) + reference | §11 — joint LM+gen RL |
| `joint_multi_rl` | C2 (based) | N refiners + generator | N-way joint RL |
| `workflow_rl` | C2 (based) | two instances | end-to-end RL across a cross-model workflow |

---

## 11. Weight-sharing topologies & the stress-test catalog

The central question — *does the Card/Loop/Program split generalize beyond serving + MoT, or will joint
multi-expert RL / cross-model pipelines / interactive sessions / joint A/V / content-adaptive compute / hot
weight-sync force a redesign?* — was answered by a battery of stress tests. **Every frontier capability landed
as a new card / method / loop / workflow / session-driver / controller, with NO new runtime primitive** (the
only real bug any test surfaced — a no-op generator gradient — was a fix in the sampler *library*). Condensed:

- **Joint LM+generator RL** (UniRL/PromptRL) — one reward → token policy-gradient on the LM *and* FlowGRPO PPO
  on the DiT; dual log-prob capture (categorical + Gaussian SDE); likelihood-based C2 (per-step identity ⇒
  ratio == 1); two independently-versioned weight-sync plans; SDE rollout sampler gated behind `sde_rollout` so
  the serve path is byte-for-byte unchanged.
- **Qwen-Omni cascade** — three disjoint experts, three loop types (`ar_decode→ar_decode→audio_decode`),
  chained cross-stage conditioning, streaming codec→waveform.
- **Cross-model Workflow** (T2I→I2V) — composition across *distinct* model instances; a `workflow_id` is a
  first-class servable in the same namespace as a `model_id`,
  registered via `WorkflowRegistry`. Plus **nested workflows** (recursive) and **non-linear shapes** (fan-out,
  best-of-N feedback).
- **N-way joint RL** — generalizes joint RL to arbitrary N (per-component sync, dict grad-targets); surfaced a
  *credit-assignment* finding (per-expert reward clean; shared reward noisy) — a reward-shaping choice, not a
  substrate change.
- **Interactive world-model session** — persistent cross-request state, transactional step-boundary
  cancellation, no cross-session smearing.
- **End-to-end RL over a workflow** — one *final-video* reward trains an *earlier* model; proven causal by a
  control (constant reward ⇒ nothing moves).
- **Joint A/V** (LTX-2 T2VS, per-modality guidance), **content-adaptive compute** (cache-dit skip + early-exit,
  ragged step counts), **hot weight-sync under in-flight serving**
  (drain-correct), **served reward model** (`REWARD_BATCH`), **speculative decoding** (exact, lower-latency
  AR), the **RL→distill flywheel** (RL-improve → distill from the RL'd teacher → faster card with provenance),
  and the **adapter plane** (per-request LoRA/ControlNet over one base).

---

## 12. Request / session / artifact + programs/workflows

Typed runtime objects, not IDs in a batch: **`Request`** (task is *declared*, never inferred; `inputs:
list[ModalPart]`; AR `sampling` vs `diffusion` params; `OutputSpec`), **`Session`** (long-lived interactive
context: prompt memory, media streams, persistent cross-request chunk-KV), **`Artifact`** (named, typed, with
provenance — `VideoArtifact`, `AudioArtifact(sample_rate)`, `TextArtifact`, … — killing the `extra["audio"]`
pattern), **`Stream`** (one ordered event channel), **`CancelScope`**. A **`Program`** composes one card's
loops into a task DAG (`ComponentNode` for kernel-free seams, `ModelLoopNode` to drive a loop to completion);
a **`Workflow`** composes *across* model instances (each stage a full `engine.run`, artifacts threaded
stage→stage) — the crossing is a Workflow boundary, not a program loop step, so each model keeps its parity
guarantee. Workflows **compile, they are not the runtime** (a ComfyUI graph maps onto a `Program`; unknown
nodes become `ExternalNode`s or a coverage rejection — never silent wrongness).

---

## 13. Serving & fleet

Per the standing instruction "don't completely rely on Dynamo; we still need our own version," v2 ships a
complete stack and treats Dynamo as one optional backend: **`AsyncEngine`** (queue, lifecycle, live SSE
streaming, step-boundary cancellation) + an OpenAI-compatible server on stdlib asyncio (`serving/http.py`:
`/v1/chat` SSE, `/v1/images`, `/v1/videos` job+poll, `/v1/models`, `/health`, `/metrics`); a
**`DisaggregatedRunner`** proven **bit-identical to inline** + role/stage pools; connectors with
`chunk_ready` + credit-based flow control; **our own `LocalFleet`** (cost/affinity/least-loaded routing,
health/drain) and a **`DynamoWorkerAdapter`** exporting a `DeploymentCard` so Dynamo *can* front us but never
*defines* the core. The clean line: the fleet owns global routing / cold start / role-pool scaling / SLO
placement / failover; the engine owns model load / loop execution / local memory+cache / parity.

---

## 14. Extensions

Versioned hook points assembled at loop build (an unused hook is *literally absent* from the hot path), wrapping
`ctx.execute(plan)`. **Observers (read-only):** `ParityAligner`, `NaNWatch`. **Interceptors
(compute-altering):** `StepInterceptor` (step-skip / cached
prediction) and `BlockInterceptor` (cache-dit DBCache/FBCache/TaylorSeer). State lives in
`LoopState.plugin_state[id]`, keyed **per request and per CFG branch** — the structural fix for module-global
residual state that silently corrupts cache-dit/TeaCache forks under concurrency. **Capability negotiation:** a
4-step distilled card *rejects* a residual-skip interceptor rather than producing garbage. **Trust boundary:**
plugins are enabled at deploy scope only; requests only *parameterize* pre-enabled plugins through validated
schemas. This is the seam M\* (§18) calls "extensible — integrate FastVideo-STA / xDiT / Inferix / FlashDrive";
v2 already has the mechanism.

---

## 15. The model taxonomy (what the runtime must serve)

| Paradigm | Examples | Loop shape | State | Output |
|---|---|---|---|---|
| Bidirectional video diffusion | Wan2.1/2.2, Hunyuan(15), LongCat, Cosmos2/2.5, LTX-2 | N denoise steps over full clip | latents, CFG branches, block caches | video |
| Few-step distilled | DMD/FastWan, TurboWan, rCM | 1–4 denoise steps | latents | video |
| Causal/AR video | Wan-Causal(-DMD), MatrixGame2/3, LongCat-VC, SF-Wan | outer chunk loop × inner denoise | DiT KV (slab chunk-KV) | video (streamable) |
| Interactive world models | MatrixGame, GameCraft, HYWorld, Gen3C, LingBotWorld | chunk loop driven by live actions | KV + action/camera cond | video stream |
| Image | Flux2, SD3.5, Qwen-Image, Kandinsky5 | N denoise steps | latents | image (batchable) |
| Audio / Joint A/V | Stable Audio; LTX-2 (video+audio), Cosmos3 t2vs | denoise over (joint) latents | per-modality CFG | audio / video+audio |
| Multi-stage refinement | LTX-2 (base→upsample→refine), Hunyuan15-SR | pipeline of loops | inter-stage latents | video |
| AR token decode | Cosmos3 reasoner; omni thinkers/talkers | token loop until EOS | paged text-KV | text / codec tokens |
| Vocoder / one-shot | LTX-2 vocoder, audio codec → wav | single forward / chunked | none | audio |
| Hybrid MoT (omni) | Cosmos3, BAGEL | AR loop then/while denoise, shared weights | text-KV + denoise state + packed seq | text+video+audio+action |

A runtime that serves the MoT row serves everything above it; a DAG-of-engines cannot (the reasoner and
denoiser share weights).

---

## 16. Package layout (actual)

```
v2/
  core/        the model-native vocabulary — contracts everything depends on (no kernels):
    enums.py / types.py   Capability/LoopKind/WorkUnitKind/ConsistencyLevel/ExecutionProfile + shared types
    card/        ModelCard, ComponentSpec, LoopSpec, RecipeSpec, ParitySpec, instance, load_card
    loop/        contracts (LoopState/WorkPlan/StepResult/Done), driver (LoopRunner), policies (cfg/flowshift/
                 precision/routing), sampler (flow-match + FlowGRPO SDE)
    program/     ComponentNode/ModelLoopNode/Program; Workflow + WorkflowRegistry (cross-model; §12)
    request/     requests, params (DiffusionParams + sde_rollout), tasks (TaskType), streams, artifacts, cancel
    parity/      aligner, ladder, compare (compare_outputs — bit-parity between execution paths)
    parallel/    plan (axis vocab), mesh, validation
  runtime/     engine (run/run_serial, workflow-aware), async_engine (pooled run-to-completion), scheduler
               (memory admission), cudagraph (piecewise capture), disaggregated (DisaggregatedRunner), pools, session, context
    cache/       keys (CacheKey, content_hash), classes (feature/residual/slab_kv/paged_kv), manager
    memory/      allocator, reservations, refundable budget
    transport/   manifests + connectors (in-proc; chunk_ready + credit flow)
    extend/      observers (NaNWatch), interceptors (cache-dit), registry, base
  platform/    Platform.detect(); COMPONENTS(kind,device,variant) + KERNELS(op,device,arch,variant) registries;
               backends/{toy.py (numpy reference + parity oracle), torch_backend.py (real GPU)}
  recipes/     the concrete cards/programs/loops — the kept families: wan21 (+ i2v / 2.2 variants),
               wan_causal (self-forcing), ltx2 (distilled / base / 2.3 joint A/V), flux2, matrixgame2,
               and the omni cards cosmos3 / bagel / qwen_omni (+ omni, their shared AR/vocoder loops)
  training/    rollout, behavior, rewards (+ServedRewardScorer), weight_sync (+WeightSyncController),
               flywheel, methods/{finetune,dmd2,diffusion_nft,self_forcing,unified_rl,joint_multi_rl,workflow_rl}
  serving/     AsyncEngine glue, OpenAI server (http.py); deploy/ — DeploymentCard, LocalFleet, DynamoWorkerAdapter
  tests/       22 files, 143 tests — `pytest v2/tests/` OR `python3 v2/run_tests.py` (zero deps)
  (root)       __init__.py · version.py · run_tests.py · registry.py · video_generator.py · examples.py
  --- _vendor/ : copied from fastvideo (FULL cutover; v2 imports zero `fastvideo.*`; internal layout mirrors upstream for diffing) ---
  _vendor/
    models/      the real nn.Module architectures for the kept models (dits/ vaes/ encoders/ audio/
                 upsamplers/) + the component loader/ + the lazy class registry — vendored, not stubs
    layers/      tensor-parallel linear/attention/norm/rotary/etc (copied verbatim)
    attention/   attention backend registry + SDPA (sparse/MoBA backends use the optional fastvideo_kernel)
    configs/     arch + pipeline config dataclasses + pipeline_registry (config-class resolution)
    distributed/ parallel_state + communication_op (single-process 1×1 device mesh dist-init)
    platforms/   device/CUDA platform detection (distinct from the v2 runtime `platform/`)
    api/         slim inference-config dataclasses (schema + results) the VideoGenerator consumes
    hooks/ · logging_utils/ · third_party/ + fastvideo_args.py · utils.py · logger.py · envs.py · forward_context.py
```

---

## 17. Current status & GPU bring-up

**The kernels are no longer toys-only.** The `platform/` substrate selects backends through two tuple-keyed
registries (`COMPONENTS(kind, device, variant)` + `KERNELS(op, device, arch, variant)`) that a detected
`Platform` resolves, with the numpy **toy backend** as the terminal fallback rung *and* parity oracle. On a GPU
box, `platform/backends/torch_backend.py` provides real `TorchComponent` adapters wrapping the real model code
(resolved from each card's `load_id`, weights from `ComponentSpec.checkpoint`) — and the loops, scheduler,
caches, parity, training, and workflows are **unchanged**, exactly as the (recipe, runtime) separation promised.

**Verified on H100 this session:**
- **20+ real model families GPU-verified** end-to-end via the torch backend (`VideoGenerator.from_pretrained` +
  `generate_video`), including SF-Wan (self-forcing causal, CFG-free 4-step DMD) and LTX-2 two-stage SR
  (frame-verified) — see [`../v2_debug_videos/vlm.md`](../v2_debug_videos/vlm.md).
- **The omni trio runs real via vllm-omni** (in an isolated venv; FastVideo's env untouched): **BAGEL-7B-MoT**
  (two-stage MoT, on-prompt 1024² image), **Qwen2.5-Omni-7B** (thinker→talker→code2wav, coherent text + 24kHz
  speech, 2-GPU), **Cosmos3-Nano** (`Cosmos3OmniDiffusersPipeline` T2V, 720p, frame-verified). Recipe +
  box-specific flags (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`, guardrails-off) are saved in the
  `vllm-omni-bringup` memory; outputs in [`../v2_debug_videos/omni/`](../v2_debug_videos/omni/).

**What is genuinely not built yet** (named, not hidden): real *distributed* parallelism inside one pool
(collectives are stubbed; multi-node is *multiple* pools fronted by the fleet); the ComfyUI workflow compiler;
WebRTC realtime *wire* (the interactive *session* logic is built); full LLM-grade AR serving (radix trees,
chunked-prefill sophistication); C3 (batch-invariance) and C4 (quality/preference + the eval system); a
torch-native loop surface (loops still marshal numpy↔torch at the boundary — a perf follow-up).

```bash
cd /home/scratch.willlin_ent/FastVideo
python3 -m pytest v2/tests/ -q      # 143 tests
python3 v2/run_tests.py             # same suite, zero deps
```

---

## 18. Roadmap

The forward plan has two tracks: (A) **adopt the valuable ideas from the M\* paper** (the closest external work
to v2's thesis), and (B) **finish the GPU port**. The full M\* gap-analysis (28-agent workflow, adversarially
verified against the code) is in [`.agents/exploration/mstar-v2-roadmap.md`](../.agents/exploration/mstar-v2-roadmap.md).

**M\* in one line.** *M\*: A Modular, Extensible, Serving System for Multimodal Models* (arXiv 2606.12688) is
essentially v2's thesis one step more mature: "every composite model is a dataflow graph; every request is a
*Walk* over it." It beats vLLM-Omni/SGLang-Omni on exactly the models v2 now runs (BAGEL, Qwen3-Omni) and
explicitly names **FastVideo** as an integratable technique. **v2 already implements the harder half** (its
`Program` *is* M\*'s graph; `shared_weight_components` *is* cross-Walk node sharing) and **exceeds** M\* on
execution-path **bit-parity**, integrated training, and the `extend/` plugin seam.
The insight: v2's substrate is ~80% built but parts are **inert** (authored metadata never wired to an
executor).

**Prioritized (P0 = highest leverage / lowest risk; parity-safe, CPU-toy-testable):**

| Pri | Item | Action |
|---|---|---|
| **P0** | Min-components per request | Consume `required_for` in `Program.active_nodes`; **fix the real bug**: `runtime/engine.py:88` uses `self.program.nodes` while `runtime/disaggregated.py:96` uses `active_nodes(request)` — the two runners disagree. Deliver via the registry/card builder so all cards inherit it. → M\*'s "execute the minimum components per request," cutting wasted reasoner/AR steps on single-modality BAGEL/Cosmos3 requests. |
| **P0** | Real EOS + declarative `DynamicLoop` | `recipes/omni/ar_loop.py` docstring claims EOS-stop but `next()` only checks `max_tokens`. Honor `eos_id` + `req.sampling.stop`; add `LoopSpec.dynamic_stop` + `register_loop_stop`. Also **training-enabling** (world-model rollout horizon). |
| **P1** | CFG/branch as a label over one paged KV pool | Make `PagedKVCache` a real `(namespace,label)` store over one budget; reuse `CacheKey.guidance_sig` for the hash (NOT `partition_field`). → the measured BAGEL win (AR path only; diffusion has no KV). |
| **P1** | `extend/` plugin: FastVideo-STA / Inferix | Expose this repo's sparse/sliding-tile attention + block-diffusion as `Interceptor`/`EngineKind` plugins — the paper's named integration, highest paper-alignment, low risk (seam exists). |
| **P1** | `ParitySpec.output_determinism` | Close the dormant C3 rung so SDE/stochastic RL can declare its parity contract honestly. |
| **P2** | Pluggable data plane; per-(node,Walk) placement; declarative TP/SP degrees; loop-spanning CUDA graphs | Gated on the multi-GPU runtime; the live 2-GPU Qwen-Omni bring-up is the natural first test. Keep the cheap declarative halves now (`EngineKind` tag, placement key, populate `parallel_plan_hash` on the serving cache path). |

**Do NOT regress** (where v2 already meets or beats M\*): execution-path bit-parity, the C0–C4 consistency
ladder, the integrated training plane (RL→distill flywheel, drain-correct hot weight-sync), CPU-toy parity for
the whole stack, the `extend/` plugin seam, Dynamo citizenship.

**GPU-port track:** wire real distributed collectives (one-pool TP/SP); torch-native loop surface (drop the
numpy↔torch boundary marshalling); admission budgeting of capture cost; per-stream workspace pool for a
concurrent executor; the AR/vocoder GPU op families; the GPU training surface (`mse_grad_step`).

---

## 19. Reference synthesis

What v2 takes and what it constrains, per surveyed system (the design is a synthesis, never a copy):

| Source | Take | Constrain / reject |
|---|---|---|
| Cosmos3 (official + port) | Shared instance across reason/diffusion/action/sound; packed multimodal sequences; component+scheduler parity matrices | A strong `ModelCard`, not the framework; no Cosmos branching in the global runtime |
| vLLM core | Running-first scheduling, reservation-before-admission, model-owned state, KV/encoder cache managers, CuMem sleep/wake, CUDA-graph dispatch, KV-connector split | Token scheduling is one WorkUnit kind; **never** full-graph compile |
| sglang `multimodal_gen` | Role pools, request lifecycle, capacity dispatch, transfer manifests, disagg state machine, cache-dit integration | No giant mutable `Req`/`ForwardBatch` as the API; not single-item diffusion scheduling |
| vLLM-Omni | Frozen pipeline-spec ⟂ deploy-YAML split; `OmniConnectorBase` + `chunk_ready`; `SupportsStepExecution` as loop-inversion prior art (we generalize to always-on); `CFGParallelMixin` proves CFG-as-policy; 3 cache subsystems confirm per-class pools | Expresses MoT only as one **opaque** request-scheduled stage (no step visibility); cross-stage KV is a *copy* |
| sglang-omni | The `next/wait_for/merge_fn/stream_to` edge vocabulary; Relay + **credit-based flow control** | Stages own disjoint weights; hybrid only as AR-stage→DiT-stage |
| Dynamo | Fleet routing, disagg role pools, KV-aware routing, KVBM, SLA planner, cold-start weight streaming | Orchestrates engines; never the core. Export a `DeploymentCard` to it |
| diffusers Modular | `ComponentSpec`/`modular_model_index.json` interchange; Guiders ≈ CFG policies | A Python pipeline interpreter is not the perf boundary; loop blocks own their iteration (not inversion) |
| xDiT / PipeFusion / USP | DiT parallelism catalog (USP, ring/ulysses, PipeFusion, CFG-parallel, DistVAE) + world-size validation | Parallelism lives in the runtime + card, not a wrapper-per-model; `pp_patch` invalid for causal |
| TorchTitan | Named mesh axes, `ParallelDims` validation, ModelSpec discipline, batch-invariance utils | Adopt the discipline, not the stack; `WeightSyncPlan` owns layout (DCP/TorchStore don't reshard) |
| verl-omni / miles / cosmos-rl | Rollout adapters, per-step capture, async rewards, group-relative advantage, TIS/MIS, deterministic modes, per-payload weight-version, AIPO off-policy masking | The **two-runtime tax is the thing to delete**; capture behavior *in* the serving loop |
| UniRL-Zero / PromptRL | Joint LM-refiner + flow-generator RL under one reward; FlowGRPO SDE/ODE with per-step log-probs; group advantage → token-PG + PPO; prompt-only vs joint ablation | A card + a method, not a bespoke trainer; SDE sampler gated behind `sde_rollout`; PPO ratio rests on the likelihood-based C2 gate |
| ComfyUI | Workflow graph, node-signature cache, model memory management | Compile to `Program`; dynamic node execution is not the core; GPL hygiene |
| Dreamverse / LiveKit | Sessions, prompt memory, typed media IPC, cancellation, duty-cycle capacity, preference-data flywheel; realtime frame/PTS streaming | Product/session behavior is first-class in the request plane, never merged into the model core; RTC only when triggers fire (<100ms interactive) |
| **M\* (2606.12688)** | The Walk-Graph framing (named Walks + state machine), per-(node,Walk) placement, CFG-as-cache-label over one paged pool, the "extensible: integrate FastVideo/xDiT/Inferix" call-out | v2 already has the harder half + exceeds on cost/parity/training; adopt the declarative authoring layer where it's inert (§18) |

---

## 20. Honest unknowns & falsifiers

An ambitious design is not an unfalsifiable one. The bets, with the experiment that kills each:

- **Step-level scheduling must *pay* for video.** Runtime-owned diffusion iteration has narrow precedent
  (vllm-omni's opt-in `SupportsStepExecution`); v2 makes it the always-on universal contract. **Falsifier:** on
  a real duty-cycle trace, if step-level scheduling does not beat a request-level baseline (≥2 concurrent
  sessions/GPU, p95 within SLO), degrade to request-level dispatch and keep only the loop contract's
  streaming/cancellation/behavior seams (which still justify it). The contract is safe even if the scheduling
  bet loses — that is the insurance.
- **The general WorkUnit scheduler may be over-general.** The *mechanism* half is validated (`VAE_TILE` and
  `REWARD_BATCH` schedule through one pool); the *economic* half (does it pay vs an in-loop call)
  is a GPU-port measurement.
- **Cost-model admission is a modeling bet** — argued on its narrow window (many small concurrent jobs),
  measured on the port.
- **Quality is unmeasured.** C4 (artifact quality / preference) and the eval system it needs do not exist yet,
  and they gate every product claim ("fast mode is equivalent", RL reward validity, distillation comparisons).

**Final position.** A model card is a (recipe, runtime) pair with a parity obligation; the model owns loop
semantics, the runtime owns loop lifecycle; one resident instance runs many loops, served pooled
run-to-completion; caches are correct by key, parity by test; training records behavior on the same loops it
serves. The weight-sharing topology, the composition graph, the
training recipe, the reward, and the session/sync lifecycle are all **data** over cards, loops, workflows, and
controllers — so a new frontier capability is a card or a driver, not a rewrite.
