# FastVideo v3 — A Model-Native Runtime for the (Recipe, Runtime) Era

**Status:** unconstrained north-star. This document assumes we are free to build a brand-new architecture with no
backward-compatibility, no migration tax, and no obligation to the current code. It exists to define the *ceiling*:
the system FastVideo should be if nothing held it back. Migration is a separate, later question — deliberately out of
scope here.

**Lineage:** this is the synthesis of `design.md` (the strategic thesis, product pull, and hard-won serving realism)
and `designv2.md` (the model-native center and typed contracts), with the open tensions of both resolved rather than
hedged.

---

## Table of contents

1. [The thesis](#1-the-thesis)
2. [The two signature ideas](#2-the-two-signature-ideas)
3. [Planes and their dependency order](#3-planes-and-their-dependency-order)
4. [Model Plane — the center](#4-model-plane--the-center)
5. [The loop contract — driven loops](#5-the-loop-contract--driven-loops)
6. [Runtime and scheduler — one currency, one WorkUnit](#6-runtime-and-scheduler)
7. [Memory, cache, transport, compile](#7-memory-cache-transport-compile)
8. [Parallelism as a model contract](#8-parallelism)
9. [Correctness — parity as a typed gate](#9-correctness)
10. [Training and RL on the same loops](#10-training-and-rl)
11. [Extensions — observers and interceptors](#11-extensions)
12. [Request, session, artifact, stream](#12-request-session-artifact-stream)
13. [Programs and workflows](#13-programs-and-workflows)
14. [Deployment and fleet](#14-deployment-and-fleet)
15. [Worked examples](#15-worked-examples)
16. [What this unlocks](#16-what-this-unlocks)
17. [Honest unknowns and falsifiers](#17-honest-unknowns-and-falsifiers)
18. [Package layout](#18-package-layout)
19. [Reference synthesis](#19-reference-synthesis)

---

## 1. The thesis

Three facts about video generation, taken together, dictate the architecture.

**A deployable video model is a post-training artifact.** Unlike an LLM — where inference optimizes frozen weights
post-hoc — a *usable* video model is *created* by training: step distillation is mandatory for usable latency, low
precision needs QAT, and causal/world models are made by distillation plus self-forcing. Every inference capability is
therefore a **(recipe, runtime) pair**: the weights and the loop that produced-and-assumes them are inseparable. A
"4-step NVFP4 FastWan" is not weights plus a flag; it is a distillation recipe, a sampler, a precision path, and a
parity contract that are one object.

**Video systems are loop systems.** Denoise timesteps, AR decode, chunked world-model rollout, VAE tiles, encoder
chunks, audio tokens, reward batches, optimizer steps, media chunks — the work is iteration, not a single `forward`. A
runtime that reduces everything to `forward()` cannot schedule, batch, cancel, stream, reserve memory for, or capture
the behavior of the thing that actually runs.

**Omni models share weights across loop types within one request.** Cosmos3's text reasoner and multimodal denoiser
are the *same resident weights*, driven by an AR loop and then a diffusion loop in one request. This cannot be a
DAG of separate engines (that doubles 30B+ of weights and severs the shared KV/denoise state); it must be one resident
model instance running many loop types. This is achievable — vllm-omni's `bagel_single_stage`/`lance` already run one
resident MoT instance doing AR `generate_text` and diffusion `generate_image` on co-resident experts in a single
request — *but* they bury that interleaving inside one opaque `DIFFUSION` stage their scheduler never sees inside,
request-scheduled with `max_num_running_reqs` forced to 1. The hard part, and the differentiation, is not *expressing*
the shared-weight loops — it is making them **runtime-visible, step-scheduled, batchable, and cost-priced.**

The architecture that falls out:

> **The atomic unit is the (recipe, runtime) pair, owned by a typed `ModelCard`.** Everything — serving, training, RL,
> products, deployment — is a *view* over that card. The **runtime owns loop *lifecycle*** (admission, scheduling,
> batching, caching, cancellation, streaming, behavior capture); the **model owns loop *semantics*** (typed state
> transitions and kernel execution). One resident model instance runs many loops; one scheduler schedules the
> *steps* of all of them in a single currency; one parity contract binds the train-forward to the serve-forward so
> the recipe and the runtime never silently drift apart.

The single invariant, stated once:

```text
Model cards own components, loops, recipes, and parity.
Programs compose loops into tasks.
The scheduler executes the steps of loops as WorkUnits under one budget.
Caches are correct by key, not by hope.
Training records behavior on the same loops it serves.
Deployment places and routes; it does not define semantics.
Products stream artifacts; they do not reach into the model.
```

Everything below is the elaboration of that invariant.

---

## 2. The two signature ideas

Two ideas do most of the work and are what an unconstrained design can reach that an incremental one cannot.

### 2.1 The (recipe, runtime) pair is a first-class, versioned, typed object

A model in v3 is not a checkpoint. It is a `ModelCard` that owns, as one versioned unit:

- the **components** (weights, loaders, layouts),
- the **loops** it can run (the runtime semantics),
- the **recipe** that produced the weights (distillation/QAT/RL config, teacher, data contract, the sampler the
  recipe assumes), and
- the **parity contract** asserting that the train-forward and the serve-forward agree to a declared level.

You cannot ship the weights without the loop they assume, and you cannot change the loop without re-proving parity.
This makes design.md's "(recipe, runtime) pair" *literal*: the deployable artifact carries its own provenance and its
own correctness obligation. It is the thing that turns "we do training and inference in one repo" from an org chart
into a guarantee — and it is essentially un-retrofittable, which is exactly why it belongs in a clean design.

### 2.2 Driven loops: the model owns control flow, the runtime owns execution

Loop inversion, done right, is not a heavy `plan_step`/`run_step` contract and not a hidden `for t in timesteps`. It
is a **driven loop**: the model describes the *next step it needs*, the runtime *decides when and with whom that step
runs*, and the model folds the result back into its own state and decides what to do next. The model keeps its control
flow (so content-adaptive decisions — cache-dit skips, EOS, VSA tile selection — are natural); the runtime keeps the
`await` (so admission, batching, cancellation, streaming, and behavior capture are universal). Per-request state lives
in the loop's own typed `LoopState`, never in module globals, so interleaving requests through one model instance
cannot smear state — the failure mode that makes naive loop-inversion dangerous is *structurally* excluded.

These two ideas are developed in §4–§5. The rest of the system is their consequence.

---

## 3. Planes and their dependency order

```text
        Products: Python · CLI · OpenAI API · ComfyUI · Dreamverse · RTC · Trainer
                                   │  (thin: validate intent, make requests/sessions, subscribe)
        Request / Session / Artifact / Stream
                                   │  (typed runtime objects, cancellation, streaming)
        Program Plane                       ← typed loop programs + compiled workflows
                                   │
        ┌──────────────── Model Plane (CENTER) ────────────────┐
        │  ModelCard: components · loops · recipe · parity      │
        │  capabilities · caches · parallelism · precision      │
        └───────────────────────┬──────────────────────────────┘
                                 │
   ┌─────────────────────────────┼─────────────────────────────┐
   │ Runtime / Scheduler          │ Training / RL                │  (same loops, different capture)
   │ WorkUnits · GPU-time budget  │ rollout · reward · weight-sync│
   └─────────────────────────────┼─────────────────────────────┘
                                 │
        Memory · Cache · Transport · Compile  (typed CacheKey, per-class pools, CuMem sleep/wake)
                                 │
        Parallelism (named axes → DeviceMesh, validated, part of the cache key)
                                 │
        Deployment / Fleet  (DeploymentCard → Dynamo; never the core)
```

Dependency rules (enforced at the package boundary, §18):

- Products do not define model semantics. Workflows do not define model semantics. Deployment does not define model
  semantics. Training does not redefine model semantics. **All of them reference the Model Plane.**
- The runtime *executes* model loops but does not *own* their math. Training *captures* behavior on serving loops but
  does not *fork* them. Cross-cutting concerns — extensions (§11), parallelism (§8), and parity (§9) — are contracts
  declared on the card, not features bolted onto the runtime.

---

## 4. Model Plane — the center

### 4.1 ModelCard

```python
class ModelCard:
    model_id: str                       # "fastwan-1.3b-nvfp4-4step"
    family: str                         # "wan"
    components: dict[str, ComponentSpec]
    loops: dict[str, LoopSpec]
    capabilities: CapabilityMatrix      # text_to_video, image_to_video, reasoning_text, vae_decode, ...
    recipe: RecipeSpec                  # ← what produced these weights (signature idea §2.1)
    parity: ParitySpec                  # ← train-forward ≡ serve-forward, to a declared level (§9)
    caches: dict[str, CacheContract]
    parallelism: ParallelismContract
    precision: PrecisionContract
    checkpoint: CheckpointManifest      # explicit components, layouts, key maps — no name-detector guessing
```

The card is both a **declarative contract** (strict enough to validate before any GPU touches it) and a **runtime
factory** (it knows how to instantiate components, bind loops, and resolve caches). It is hub-interchange compatible
with diffusers' `modular_model_index.json` / `ComponentSpec` so models published either way load both ways.

`CheckpointManifest` replaces today's implicit `model_index.json` + name-detector resolution with explicit declared
components and `required_for` / `optional_for` task sets (the Cosmos3 lazy-sound-VAE problem becomes a declaration, not
an `if env_var` inside `forward`).

### 4.2 RecipeSpec — the provenance half of the pair

```python
class RecipeSpec:
    method: str                 # "dmd2" | "self_forcing" | "attn_qat_nvfp4" | "diffusion_nft" | "base"
    parents: list[str]          # teacher / base model_ids this was distilled or RL'd from
    data_contract: DataRef      # what the recipe trained on (for governance and reproduction)
    assumes_loop: str           # the loop_id this recipe's weights require at serve time
    assumes_precision: str      # the precision the QAT recipe baked in
    consistency_required: str   # the minimum parity level this recipe's outputs are valid under (§9)
```

`assumes_loop` and `assumes_precision` are the teeth: a 4-step distilled model whose `assumes_loop = "ddim_4step"`
cannot be served under a 50-step sampler without a typed mismatch error. The recipe and the runtime are bound.

### 4.3 ComponentSpec and LoopSpec

```python
class ComponentSpec:
    component_id: str
    kind: str                          # dit | vae | text_encoder | reasoner_tower | reward_head | ...
    load_id: str
    config_schema: type
    io_schema: tuple[type, type]
    precision_policy: PrecisionPolicy
    placement_policy: PlacementPolicy
    parallel_constraints: ParallelConstraint
    parity_tests: list[ParityTestSpec]

class LoopSpec:
    loop_id: str                       # diffusion_denoise | ar_decode | chunk_rollout | vae_tile | ...
    state_schema: type                 # the typed LoopState (no dicts)
    step_schema: type                  # the typed WorkPlan a step emits
    result_schema: type                # the typed StepResult a step returns
    behavior_schema: type | None       # what to capture for RL (None if not training-relevant)
    step_cost_model: CostModel         # predicted GPU-time per step at (shape, precision, policy) — §6
    valid_parallel_plans: list[ParallelPlanPattern]
    graph_capture: GraphCapturePolicy
    cache_policy: CachePolicy
```

A `ModelInstance` is a resident, loaded card: component instances, model state, caches, compiled graphs, and a
parallel plan. **A request may run several of the card's loops against one `ModelInstance`.** That single sentence is
the difference between this design and a stage-only design, and it is what makes omni native:

```text
one Cosmos3 ModelInstance, one request:
  ar_decode(reasoner)  →  pack  →  diffusion_denoise(vision[+action][+sound])  →  vae_tile_decode  →  audio_decode
  └────────────── same resident weights, shared packed state, scheduled as steps ──────────────┘
```

---

## 5. The loop contract — driven loops

### 5.1 The contract

A loop is a **serializable state machine** the runtime drives:

```python
class Loop(Protocol):
    def init(self, req: Request, model: ModelState, ctx: LoopContext) -> LoopState: ...
    def next(self, state: LoopState) -> WorkPlan | Done: ...      # describe the next step; NO GPU kernels here
    def advance(self, state: LoopState, result: StepResult) -> LoopState: ...   # fold result in; decide what's next
    def finalize(self, state: LoopState) -> LoopResult: ...
```

The runtime's driver — the only place iteration lives:

```python
state = loop.init(req, model_state, ctx)
while True:
    plan = loop.next(state)                  # typed WorkPlan: resources, cache reads/writes, shape, sinks, cancel-scope
    if isinstance(plan, Done):
        break
    result = await ctx.execute(plan)         # ← THE INVERSION POINT: runtime admits, batches, places, runs, returns
    state = loop.advance(state, result)      # content-adaptive: next() can branch on everything in state, incl. result
    for chunk in plan.emits:
        ctx.emit(chunk)                       # streaming falls out
return loop.finalize(state)
```

Why this is the right contract, point by point against the failure modes:

- **Content-adaptive steps are natural.** `next()` reads `state`, and `advance()` has already folded in the last
  `StepResult` — so cache-dit's skip decision (a residual comparison from the prior step), AR's EOS, and VSA's
  content-dependent tile selection are ordinary control flow in the model. This is the tension `designv2.md`'s
  "pure plan_step that pre-declares shape" could not resolve; here it dissolves, because planning the *next* step is
  allowed to depend on the *previous* result. `next()` is still kernel-free (it *describes* work; it does not run it),
  which is all the scheduler needs.
- **Cross-request state safety is structural.** All per-request state is in the loop's typed `LoopState`. There are no
  module-level residual/KV globals (the bug that silently corrupts cache-dit and TeaCache forks under concurrency).
  Interleaving requests through one `ModelInstance` cannot smear state because there is no shared mutable state to
  smear. This is the safety property naive loop inversion lacks, made impossible-to-get-wrong by construction.
- **The runtime owns everything it needs and nothing it doesn't.** `execute(plan)` is the single seam for admission,
  memory reservation, cross-request batching, placement, graph dispatch, cancellation, and behavior capture. The model
  never sees the scheduler; the scheduler never sees the model's math.
- **Serializable, therefore migratable and resumable.** `LoopState` is typed and serializable, so a half-finished
  1000-step job is a resume point: preempt by stopping the driver, migrate by shipping `LoopState` to another worker,
  recover from a crash by replaying from the last serialized state. (A coroutine that keeps state in a suspended Python
  frame — the tempting sugar — cannot do this; the explicit state machine is the price of resumability, and it is
  worth paying.)

Custom step bodies are first-class, not an escape hatch with an asterisk: a family whose math is genuinely braided
(Cosmos's EDM coefficients consumed inside the CFG branch with an x0-space combine; LTX-2's 1–4 runtime-decided
guidance passes) writes `next()`/`advance()` by hand using samplers and CFG utilities as a *library*. The runtime
requires only the four methods; *how* a step body is factored is the model's business. Policies (CFG, expert routing,
precision, flow-shift, conditioning) are the *default* decomposition that deletes duplication for the families that
fit — never an admission requirement.

### 5.2 Loop granularity

Chosen by runtime value, not purity:

- too coarse → cannot cancel/batch/reserve/stream/record at useful points;
- too fine → scheduler overhead dominates, graph capture fragments;
- good default → one denoise step (or window), one AR decode batch, one encoder chunk, one VAE-tile batch, one
  reward/logprob batch.

The runtime may **fuse** adjacent compatible WorkPlans after planning (an optimization); the unfused boundary remains
the semantic model, so parity and behavior capture are defined on the unfused loop.

### 5.3 CFG is a policy over *one* shared denoise body (verified)

A natural worry: CFG changes the *shape* of the step (one forward vs two vs a batched pair vs a data-parallel split),
so can one shared denoise loop really host all of it by swapping a policy? **Yes — and it is proven by existing code,
not aspiration.** vllm-omni's `CFGParallelMixin.predict_noise_maybe_with_cfg` + `combine_cfg_noise`
(`diffusion/.../cfg_parallel.py:76-212`) already runs sequential-2-forward, batched-1-forward, *and* cfg-parallel
through **one** pair, with the loop body unaware of which. The clean cut is three layers:

- **In-loop `CFGPolicy`** — branch vocabulary (`[cond]`, `[cond, uncond]`, per-modality, STG-perturbed),
  the combine formula (standard `uncond + s·(cond−uncond)`, CFG-zero `st_star`, `cfg_normalize`/`guidance_rescale`),
  and **per-request mutable state** (the adaptive-gate cached delta with model-id self-invalidation is the canonical
  state case — and exactly why state lives in `LoopState`, §5.1). **Batched-vs-two-forward is a *dispatch detail
  inside one policy*, not a separate mechanism.** This covers classic / batched / adaptive-gate / per-modality.
- **`cfg`-parallel is a *parallelism axis*, not a policy** — it shards the policy's branches across ranks and runs the
  *same rank-invariant `combine`* on every rank. It composes *under* any `CFGPolicy`; you own a `BatchedCFG` policy
  **or** a `cfg` group, never both (the §9 build-guard).
- **Companions are an *orchestrator pattern*, not in the loop** — splitting a request into companion sub-requests
  upstream of diffusion (the conditioning is precomputed and bundled in; the loop is unchanged).

Two caveats keep the first pass honest: the `combine` runs in *the step body's* numeric space (Cosmos combines in
x0-space after EDM preconditioning, not noise-space — the body fixes the space, the policy fixes the algebra), and
embedded-guidance (Flux) is a **degenerate single-branch identity-combine policy** (guidance rides inside the forward
kwarg), kept *inside* the same abstraction rather than special-cased as "no CFG." This is the same shared denoise body
that RL rollout reuses (§10) — one CFG taxonomy serves both serving and rollout.

---

## 6. Runtime and scheduler

### 6.1 One WorkUnit, one currency

Every `await ctx.execute(plan)` produces a **WorkUnit**: the smallest schedulable action with a resource reservation
and a loop boundary. Kinds: `ar_prefill`, `ar_token`, `diffusion_step`, `diffusion_window`, `chunk_step`,
`encoder_chunk`, `vae_tile`, `audio_chunk`, `reward_batch`, `logprob_batch`, `transfer`, `cache_io`, `graph_capture`.
Tokens are *one kind*, not the scheduler — this is the generalization of vLLM's token scheduler that diffusion forces.

**The budget currency is predicted GPU-time, not counts.** A bidirectional denoise step re-attends the full latent at
O(L²) with zero KV amortization (every step pays full price); an AR decode step is ~O(context) against a cache; a
chunked-causal step sits between. Counting "steps" or "tokens" puts items three orders of magnitude apart in one
bucket. So each WorkUnit converts to GPU-seconds via its `LoopSpec.step_cost_model`, calibrated online by the Profiler
(§11). **The same cost model is the interface published to the fleet** (§14): the scheduler's internal budget and
Dynamo's routing/autoscaling input are one object, built once.

Two honesty caveats, kept from design.md's contact with reality:

- **Admission uses the conservative baseline.** The design's own flagship features make realized cost unknowable in
  advance — cache-dit skips are residual comparisons, VSA tiles are content-dependent, AR length is unbounded
  (budgeted at the `max_tokens` cap, refunded on early EOS). Telemetry refines calibration; it never licenses
  admission optimism.
- **A denoise step is indivisible.** A 30s-1080p step bounds iteration latency no matter the budget. Mitigations are
  first-class, not afterthoughts: cost-class pools (jumbo steps don't co-schedule with latency-class work), SP within
  a node to shrink jumbo wall-time, and admission-time SLO classes so the fleet planner scales pools per class. This
  is why the scheduler is *cost-aware*, not just *count-aware*.

### 6.2 WorkPlan and admission

```python
class WorkPlan:
    loop_id: str
    instance_id: str
    kind: str
    shape_sig: ShapeSignature        # for batch compatibility + graph capture key
    resources: ResourceRequest       # compute (GPU-s), resident bytes, peak-activation bytes, cache blocks, xfer bw, sinks
    cache: CachePlan                 # typed reads/writes (§7)
    placement: PlacementHint
    cancel_scope: CancelScope
    emits: list[StreamChunk]
class Done: result: LoopResult
```

**Admission rule (the soundness condition of multiplexing):** *do not admit a waiting WorkUnit unless every resource
it requests can be reserved* — compute budget **and** memory (resident + worst-case peak) **and** cache blocks **and**
transfer bandwidth **and** graph-capture shape **and** output sinks. Two requests that fit individually but jointly OOM
are rejected at admission, not discovered at step 37. This is vLLM's "token budget is half the story, `allocate_slots`
is the other half," generalized.

### 6.3 The scheduler, in layers (each testable on a fake pool, no GPU)

1. **RequestScheduler** — accepts requests/sessions, selects programs, starts loop drivers.
2. **LoopScheduler** — drives `next()`, collects pending WorkPlans.
3. **BatchScheduler** — groups compatible WorkPlans by `(instance, loop_kind, shape_sig, precision, parallel_plan,
   graph_key)`; image diffusion and AR decode batch across requests, jumbo video stays batch-of-1, Cosmos-style
   token-budget packing is an opt-in.
4. **PlacementScheduler** — worker, role pool, instance, device mesh.
5. **TransferScheduler** — tensor / cache / artifact movement as scheduled WorkUnits.
6. **AdmissionController** — the reservation gate of §6.2.

Policies: running loops first (vLLM); preempt only at loop step boundaries; cancel only at declared scopes; prefer
cache hits when latency/fairness allow; never starve a long denoise loop behind short AR requests.

### 6.4 SPMD consistency and failure isolation

All ranks of a pool must make identical scheduling decisions or NCCL deadlocks. **Rank-0 decides, broadcasts** — the
existing discipline, now also the channel for the **abort broadcast** (failure isolation and scheduling share one
consistency mechanism). Failure classes: *request-fatal* (NaN flagged by NaNWatch, one request's step error) →
SPMD-consistent abort of that request, deliver partial artifacts with a structured error; *pool-fatal* (illegal
access, NCCL desync) → pool re-init, invalidate pool caches, resume requests from serialized `LoopState` where one
exists. **Cancellation is common-path, not exceptional** — vibe-directing makes abandoning in-flight work the *normal*
user action; cancel takes effect at the next step boundary, drops queued WorkUnits, releases `LoopState` and cache
handles, reports `cancelled`.

---

## 7. Memory, cache, transport, compile

Video and omni inference are memory systems as much as compute systems. This plane is explicit and typed.

### 7.1 Cache correctness is a contract

```python
class CacheKey:
    model_id: str; component_id: str; loop_id: str | None
    weights_version: str; adapter_versions: dict[str, str]
    precision: str; parallel_plan_hash: str
    shape_sig: str; layout_sig: str
    scheduler_sig: str | None; guidance_sig: str | None; seed: int | None
    input_hashes: dict[str, str]; step_index: int | None
    contract_version: str
```

**If a field can change output semantics, it is in the key.** Incorrect reuse is worse than no reuse. The serving
hazard this kills: a workflow-cloud request that shares a prompt but differs in te-LoRA stack must not serve stale
embeddings — so the key is *partitioned* by `adapter_versions`, not flushed. An RL `update_weights` bumps
`weights_version` and invalidates wholesale.

### 7.2 Per-class pools (the granularity reality)

There is **no single unified block pool**, because a unified pool requires uniform bytes-per-block and our cache
classes differ by 150–500× in natural granularity (a text-KV page ≈ 64 KB/layer; a causal-video latent-chunk slab is
9.6–32 MB/layer) and their demand is workload-decoupled. Each class gets a statically budgeted pool behind one
`CacheHandle`: paged text-KV (`ar_decode`), slab chunk-KV (`chunk_rollout`, with a declared training mode that
disables mid-rollout recycling and keeps grad-aware index snapshots), feature caches (text/vision-encoder, content-hash
keyed, reference-counted FIFO), residual caches (cache-dit, scoped per `LoopState`), weight/adapter cache
(disk→CPU→GPU LRU for the workflow cloud). MoT falls out: the und pathway draws paged KV, the gen pathway draws slab or
nothing — independent budgets, no interference. **KV is the minority case** — a pure bidirectional deployment allocates
none of it; the machinery materializes only when a card declares KV-bearing loops.

### 7.3 Memory, transport, compile

- **Memory** — tagged pools, sleep/wake by tag (CuMem-style; tags are component names), reservation before admission,
  per-role budgets, host-pinned staging. Sleep/wake is component-granular for RL (drop DiT + caches, keep
  VAE/text-encoder resident).
- **Transport** — manifest-based and pluggable: in-proc reference → SHM → CUDA IPC → NCCL/UCXX/NIXL/RDMA →
  object-store. KV/cache-bearing edges speak a `KVConnector`-shaped protocol (scheduler-side query/alloc/finish +
  worker-side async load/save) so NIXL/LMCache/Mooncake/KVBM implement it directly. Transfers are scheduled WorkUnits,
  not side effects.
- **Compile** — CUDA graphs and `torch.compile` managed by a `CompileCache` keyed on `(model, component, loop,
  work_kind, shape_sig, precision, parallel_plan, backend)`. **Never full-graph across the engine** (per vLLM's own
  reversal): per-block compile where it pays, manual fused ops permitted in model code, breakable CUDA graphs as an
  *optimization tier* over an always-correct eager baseline. Graph capture is planned by the scheduler (padding,
  bucketing, capture sizes affect admission and batching).

---

## 8. Parallelism as a model contract

Parallelism is not a launch flag; it affects cache keys, scheduling, transport, capture, and parity, so it lives on
the card.

```python
class ParallelPlan:
    axes: dict[str, int]   # dp, tp, sp(=ulysses×ring), cp, cfgp(≤2), pp_patch, vae, ep, fsdp, role, replica
    mesh_order: list[str]
    placement: PlacementSpec
    communication: CommunicationSpec
```

Declarative, validated, compiled to a PyTorch `DeviceMesh` via a `ParallelDims`-style builder
(product-of-degrees validation, cached submeshes). **Pre-flight or it fails at load, never halfway.** Ownership
conflicts are build errors (CFG owned by a `BatchedCFG` *policy* or a `cfgp` *group*, never both). Applicability
conditions travel with axes: `pp_patch` (PipeFusion displaced-patch pipelining) is **invalid for causal/AR** (stale KV
breaks causality) and the validator enforces it per card. Degree-one axes exist as trivial groups so component code
needs no special cases. **Pools are single-node**; multi-node scale is *multiple pools* fronted by the fleet (§14) —
the engine never owns a cross-node NCCL mesh inside one pool.

---

## 9. Correctness — parity as a typed gate

This is the section both prior documents needed and neither fully had.

### 9.1 The parity contract

Every card carries a `ParitySpec`. Parity is **measured, never assumed**, by a `ParityAligner` observer (§11): record
named taps per step/block from a reference (the official framework, or a pre-change build); compare-mode replays with
fixed seeds and reports the first divergence beyond per-tap tolerance. This is the engine behind the "old loop vs new
loop, bit-identical" gate and the standing instrument for every port, precision change, and kernel swap.

### 9.2 The consistency ladder (with the rung both prior docs missed)

```text
C0  component parity      — VAE, encoder, transformer block, scheduler step in isolation
C1  loop parity           — full denoise trajectory / AR logits, fixed seed
C2  behavioral identity   — the train-forward and serve-forward agree on the quantity the RL objective uses:
                              · likelihood-based methods (GRPO-class): per-step log-prob identity
                              · likelihood-free methods (DiffusionNFT-class): seeded final-sample +
                                prediction-space identity (old_deviate / ref-MSE) — there are NO log-probs to match
C3  distribution parity   — rollout distribution under allowed nondeterminism
C4  artifact quality      — SSIM-class, reward agreement, human-preference (gates product claims; needs the eval system)
```

The C2 split is load-bearing and is the lesson of the landed RL stack: the shipped Wan DiffusionNFT is
**likelihood-free** — it captures only final clean latents and contrasts the student against an implicit negative
policy in prediction space, so "log-prob identity" is *undefined* for it. A ladder that assumes log-probs (as both
`design.md`'s and `designv2.md`'s early framings did) cannot describe the only RL method actually in the tree. RL
methods declare their required level on the `RecipeSpec`.

### 9.3 The gate that catches what batch-of-1 cannot

Loop inversion's real hazard is **cross-request state smearing under interleaving** — and a batch-of-1 parity gate is
*structurally blind* to it, because the corruption only manifests when two requests share a loop. §5.1 excludes the
hazard by construction (state in `LoopState`, never globals), but construction-arguments need a test. So v3 makes a
**batch-of-N interleave parity test** a *required* gate: two (or more) concurrent requests, interleaved at step
granularity, must be bit-identical to the same requests run serially. This is the test the whole loop-inversion bet
lives or dies on, and it is named here as a first-class obligation, not left implicit.

### 9.4 Three execution profiles, one definition

Even in one runtime there are three forwards: the **serve** forward (no-grad, graphed, cached, possibly quantized), the
**rollout** forward (serve profile + behavior capture), and the **train** forward (grad, checkpointed, FSDP-gathered).
They share *one* loop definition; they differ only in grad mode and capture. The ladder measures the gap; the recipe
declares the level it needs. "Train BF16, serve FP8" is legal only at C2-corrected with importance-sampling, and the
card says so. This is how the (recipe, runtime) pair stays honest: the contract is typed and tested, not trusted.

---

## 10. Training and RL on the same loops

```text
serve   : request      → program → loop → WorkUnits → artifacts
rollout : prompt batch → program → loop → WorkUnits → BehaviorRecords → rewards → update
```

The loop kernel is shared; the only difference is output capture and training policy — not a second interpretation of
the model. This is design.md's §8 thesis and v2's training plane, with the dependency rule kept absolute: **`training`
may require behavior records but must not fork serving loop logic; the engine never imports `training`.** The engine
*is* the rollout engine (it already runs the loops); the trainer is a client.

**This is the moat — and it is the one place a serving-only runtime structurally cannot follow.** vllm-omni proves
omni serving can be production-grade, but it has *no* training/RL plane at all; verl-omni and miles prove the
alternative — a standalone trainer-side sampler on a *different* runtime than serving — costs the two-runtime tax
forever. The whole point of collocation is that **the rollout forward *is* the serve forward plus capture**: same loop,
same caches, same batcher, same numerics. Three consequences nothing else gets:

- **Every serving optimization is automatically a rollout optimization.** Distilled few-step samplers, cache-dit
  skips, CFG-parallel, paged/feature caches, step batching — the recipe team builds them once for serving and the RL
  rollout inherits them for free. FastVideo's *own* landed DiffusionNFT is the negative example that proves the
  point: it vendors a bare-model `for`-loop (`rl/common/sampling.py`, whose docstring says it "intentionally does not
  call FastVideo's full inference pipelines"), and DMD2 vendors a *second* one (`dmd2.py::_student_rollout`) — so
  today's rollout runs with **zero serving-grade optimizations** (no CFG, dense attention, full 25-step ODE,
  one-sample-at-a-time). Collocation deletes both private loops.
- **RL rollout is a *better* batching case than open-world serving — not a worse one.** A GRPO/NFT group is K
  *identical-config* samples of one prompt: same shape, same schedule, same CFG branch. The landed config is K=24
  (`num_video_per_prompt: 24`), 6 prompts/batch × 48 batches = **288 prompt-slots/GPU/epoch, each a 24-wide homogeneous
  denoise batch** — zero bucketing required (serving must bucket heterogeneous resolutions/steps/CFG across users; a
  GRPO group is homogeneous *by construction*). And all K samples share one prompt embedding, so the content-hash
  feature cache computes the text encoder **once per group instead of 24×**. The vendored sampler captures none of
  this; it carries the embedding per sample and runs one shape at a time.
- **One numerics surface.** Serve-forward and rollout-forward differ only in grad mode and capture (§9.4), so there is
  no rollout-vs-train kernel gap to patch — the consistency ladder *measures* the gap rather than a correction layer
  *papering over* it. For the landed likelihood-free NFT, "reuse holds" means it holds at the **C2 behavioral rung**
  (seeded sample + prediction-space identity), under a `CFGPolicy` that is conditional-only and a `WeightSyncPlan`
  whose role is the decay-blended old policy — all of which the card already declares.

- **BehaviorRecord** — captured at generation time (reconstructing later is fragile): seeds, scheduler trajectory,
  timesteps, latents-or-refs, logprobs *where applicable*, sampled/action tokens, guidance, reward in/out, cache
  assumptions, precision, parallel plan, attention backend, deterministic flags, `weights_version`. Sized honestly:
  full MoE-routing capture is GB/sample for Cosmos3-class requests, so it is an **opt-in instrument** for goldens and
  debugging, not always-on.
- **Weight-sync lifecycle** — freeze admission for the affected role/version → drain or boundary-stop in-flight loops →
  transfer weights/deltas → bump `weights_version` → invalidate incompatible caches and graphs → publish version →
  resume. A `WeightSyncPlan` is three inputs (mesh specs + per-model layout adapters + transport), validated
  pre-flight, CPU-testable on fake pools. RL ships a *role*, not "the weights": student / EMA / decay-blended old
  policy is declared (the landed NFT behavior policy is the *old* copy, not the student — the plan must carry that).
- **Roles** (policy, rollout, reference, reward, critic, evaluator, data, coordinator) reference the same cards and
  loops; they are deployment concerns, scaled by the fleet.
- **The industry tax we delete:** verl-omni re-implements Wan inside vLLM-Omni and corrects numerics afterward; miles'
  headline features (TIS/MIS, bitwise logprobs, R3 routing replay, unified FP8) are all mismatch patches for *two
  runtimes with different kernels*. One model definition, one kernel set, one measured ladder is the answer — viable
  at FastVideo's 1–30B FSDP2 scale (the boundary condition: a Megatron-class trainer at 100B+ re-enters the
  two-runtime world, and the ladder is the fallback there).

---

## 11. Extensions — observers and interceptors

The optimization, debugging, and parity surface, as versioned hook points assembled at loop build (an unused hook is
*literally absent* from the hot path). It composes with §5 cleanly: the hooks wrap `ctx.execute(plan)`.

- **Observers (read-only):** `ParityAligner` (§9), `Profiler` (per-step wall+CUDA, calibrates the cost model),
  `NaNWatch` (first-NaN localization), `ActivationTrace`. They cannot mutate state.
- **Interceptors (compute-altering):** `StepInterceptor` (step-skip / cached-prediction) and `BlockInterceptor`
  (cache-dit's DBCache/FBCache/TaylorSeer). State lives in `LoopState.plugin_state[id]`, keyed **per request and per
  CFG branch** — the structural fix for the module-global residual state that silently corrupts cache-dit/TeaCache
  forks under concurrency. cache-dit is the reference integration (the library sglang's serving already uses);
  conflicting interceptors are rejected pre-flight; a 4-step distilled card *rejects* step-skip caches rather than
  producing garbage.

**Trust boundary:** plugins are enabled at deploy scope only (never a per-request `plugins=[...]` field that would wire
third-party code selection into the public API); requests only *parameterize* pre-enabled plugins through validated
schemas, and exact-mode requests reject `distribution_altering` parameterization outright.

---

## 12. Request, session, artifact, stream

Typed runtime objects, not IDs in a batch (the Dreamverse/LiveKit lesson):

- `Request` — one generation, scoring, encoding, training-sample, or conversion job.
- `Session` — a long-lived interactive context: prompt memory, media streams, cancellation, partial updates,
  cross-request chunk-KV that persists for a game/scene session.
- `Artifact` — a *named, typed* output with provenance (which node produced it): `VideoArtifact`, `AudioArtifact(
  sample_rate)`, `TextArtifact(token_ids, text)`, `TensorArtifact`, `LatentArtifact`. This kills the `extra["audio"]`
  pattern — audio carries its sample rate as a first-class artifact, not a dict passenger.
- `Stream` — one ordered event channel for previews, media chunks, progress, logs, finals.
- `CancelScope` — structured cancellation target (request / loop / stream / session).

Typed event taxonomy (`request.*`, `session.*`, `artifact.*`, `media.{init,chunk,complete}`, `trace.*`). A
`media.chunk` must know its stream, byte-range or shared-buffer ref, codec/container, timestamp range, and
preview-vs-final — invalid combinations are unrepresentable.

The **request is the only currency crossing the product boundary.** A typed `Request` carries `task: TaskType`
(declared, never inferred), `inputs: list[ModalPart]` (Text/Image/Video/Audio/Action/Latent), AR `sampling` vs
`diffusion` params, an `OutputSpec` (requested modalities + streaming + capture flags), and per-node overrides. Task is
declared; heuristics may only *suggest* a default at the boundary.

---

## 13. Programs and workflows

A **Program** composes a card's loops into a task; the card says what loops *exist*, the program says how to *run* them
for this request. Kinds: `InlineProgram` (many loops, one resident instance — the omni default), `DisaggregatedProgram`
(encoder→denoiser→decoder role pools), `WorkflowProgram` (compiled from ComfyUI), `TrainingProgram`,
`RealtimeProgram`. Nodes: `ModelLoopNode`, `ComponentNode`, `ExternalNode`, `ArtifactNode`, `ControlNode`,
`StreamNode`, `TransferNode`. Edges are typed (`TensorEdge`, `ArtifactEdge`, `StreamEdge`, `ControlEdge`, `CacheEdge`,
`BehaviorEdge`). Linear pipelines are the degenerate case; branches/fan-out/fan-in are real (video and audio decode in
parallel after a joint denoise). A separate deploy config maps nodes → pools/devices/parallelism, defaulting to "one
pool, everything colocated."

**Workflows compile, they are not the runtime.** A ComfyUI workflow's tier-1/tier-2 static sublanguage maps onto a
`Program` (`CheckpointLoaderSimple→card`, `KSampler→diffusion_denoise` with sampler/CFG policies,
`LoraLoader→adapter hot-swap`, `ControlNetApply→ConditioningInjector`); unknown nodes become `ExternalNode`s or a
coverage rejection — never silent wrongness. The moat: an orchestrator can run stock workflows on rented silicon;
substituting a *credibly faster* model requires owning the recipe (§2.1) — which an orchestrator structurally cannot
do. Equivalence is a quality-metric vs a reference render (C4), never a bit-parity claim.

---

## 14. Deployment and fleet

The engine exports a `DeploymentCard` and lets a fleet orchestrator (Dynamo) route — Dynamo orchestrates engines, it
is never the engine core.

```python
class DeploymentCard:
    engine_id: str; model_cards: list[str]
    capabilities: CapabilityMatrix; role_pools: list[RolePoolSpec]
    supported_programs: list[str]; supported_parallel_plans: list[ParallelPlan]
    cache_events: list[CacheEventSpec]; transfer_endpoints: list[TransferEndpoint]
    cost_model: CostModel             # the SAME §6 cost model — one object, two consumers
    health: HealthSchema; slo: SLOSchema
```

Clean line: the **fleet** owns global routing, tenant policy, cold start, role-pool scaling, cross-node transfer,
placement-by-SLO, health/failover, multi-engine upgrades, global cache routing. The **engine** owns model load, loop
execution, local scheduling, local memory/cache, model-specific behavior, parity, WorkUnit batching. The asks of the
fleet are concrete and each has a fallback: generic affinity key-spaces (checkpoint/session/lora/weight_version beyond
token prefixes), a heterogeneous request-cost interface (the §6 cost model), chunked media streaming through the
frontend, role-graph disagg (N roles, not two), an RL weight plane (versioned broadcast + staleness-aware routing),
cache-object tiering (KVBM generalized to latent/session caches), and session lifecycle as a routing primitive.

---

## 15. Worked examples

**(a) Text → video, one instance.** `Request(T2V)` → `InlineProgram` → `diffusion_denoise` loop. Driver: `init`
builds sigmas/latents; `next` emits a `diffusion_step` WorkPlan (batch-of-1, SP+CFG-parallel); `advance` folds the
model output and the CFG combine; cache-dit's `BlockInterceptor` may skip blocks based on the prior residual; at
`Done`, a `vae_tile_decode` loop runs; output is a named `VideoArtifact`. Compiles/captures exactly like today's inner
loop — nothing tensor-level changes for batch-of-1.

**(b) Cosmos3 omni, one request, shared weights.** `InlineProgram` over one `ModelInstance`: `ar_decode(reasoner)`
yields `ar_token` WorkUnits that join the AR continuous-batching group → `pack` → `diffusion_denoise(vision+action+
sound)` yields `diffusion_step` WorkUnits → fan-out `vae_tile_decode` + `audio_decode`. The reasoner's tokens and the
denoiser's steps hit the *same resident weights*; the scheduler is the mode multiplexer. AR decode runs data-parallel
across the cfg×sp weight-replica axes (decode is sequence-length-1; SP has nothing to shard). This is the workload no
DAG-of-engines can express.

**(c) Image serving at scale.** Many `Request(T2I)` → the `BatchScheduler` groups `diffusion_step` WorkUnits by
resolution bucket and batches across requests every step — the case where cross-request batching pays most. The
*same* scheduler that runs (a) and (b).

**(d) RL rollout.** A `TrainingProgram` drives the *same* `diffusion_denoise` loop with `OutputSpec(capture=behavior)`;
each step emits a `BehaviorRecord` slice; rollouts run C2 by construction (in-process, trainer kernels, pinned
attention). For likelihood-free NFT the behavior is seeded final latents + prediction-space deviations; for a future
GRPO-class method it is per-step log-probs — the loop is identical, the capture differs, the ladder rung is declared.

**(e) Dreamverse session.** A `Session(realtime_video_continue)` holds chunk-KV across 5s segments; `push_text`
updates prompt memory; `stream` yields `media.chunk` previews from loop `emit`s; a direction change throws
`Cancelled` at the next step boundary and starts a new segment. Capacity comes from duty cycle + cost-model admission +
distillation — interleaving is fairness, not throughput.

**(f) ComfyUI compile.** `workflow.compile(json)` → a `WorkflowProgram` of `ModelLoopNode`/`ComponentNode` over a
weight-fleet-cached card, with stacked-LoRA patch/unpatch priced by the §6 weight-transition cost — same runtime, new
frontend.

---

## 16. What this unlocks (the unconstrained payoff)

Things no incremental design — and neither prior document — could actually claim:

- **True omni/MoT serving.** One resident model, many loop types, scheduled at step granularity, in one request. Not a
  monolith bypassing the abstraction (the Cosmos3 port's necessary hack), not a DAG doubling weights — native.
- **Train ≡ serve by construction.** Because rollout and serve are the *same loop*, the (recipe, runtime) flywheel is
  real and measured, not aspirational: Dreamverse's directing sessions emit preference data → the RL plane → faster
  distilled cards → a better product, with the ladder guaranteeing the preferences collected under the serving profile
  transfer into training.
- **Real-time interactive omni.** The driven-loop contract + sessions + WebRTC frame/PTS streaming + step-boundary
  cancellation make the <100ms motion-to-photon interactive world-model loop expressible in the same runtime that
  serves batch T2V.
- **One substrate, three personas.** A research vehicle (new ports land as cards), a product engine (Dreamverse, the
  workflow cloud), and an RL rollout engine — without three codebases. The dependency rules keep them from fusing into
  mud.
- **Correctness you can sign.** A deployable card is a *(recipe, runtime)* pair with a typed parity obligation; "this
  fast model is equivalent" is a claim with a test behind it, which is the one thing an orchestrator-without-recipes
  can never say.

---

## 17. Honest unknowns and falsifiers

An unconstrained design is not an unfalsifiable one. The bets, stated with the experiment that kills each:

- **The novelty is concentrated and real.** Runtime-owned diffusion iteration has a **narrow, opt-in precedent** —
  vllm-omni's `SupportsStepExecution` (`prepare_encode/denoise_step/step_scheduler/post_decode`,
  `diffusion/models/interface.py:44-67`) is exactly runtime-owned diffusion iteration at step granularity, and maps
  almost 1:1 onto our `init/next/advance/finalize` — but it is Qwen-Image-only and off in every shipped deploy. What
  is unprecedented is making it the **always-on universal contract** *and* a fully general WorkUnit scheduler over
  heterogeneous units. The risk is not the loop contract (a state machine is well-understood, and now demonstrably
  shippable); it is whether step-level cross-request scheduling *pays* for video. **Falsifier:** publish a load profile and targets from a real duty-cycle trace; if step-level scheduling does
  not beat a request-level baseline (≥2 concurrent sessions/GPU, p95 within SLO), the scheduler degrades to
  request-level dispatch and the loop contract keeps only its streaming/cancellation/behavior seams — which still
  justify it. The contract is safe even if the scheduling bet loses; that is the design's insurance.
- **The general WorkUnit scheduler may be over-general.** Scheduling VAE tiles, transfers, and graph-captures through
  the *same* admission machinery as denoise steps is elegant and unproven. **Falsifier:** if, after Phase 2, the
  non-diffusion/non-AR WorkUnit kinds (tile, transfer, cache_io) gain nothing from unified scheduling over a simple
  in-loop call, collapse them back to in-loop operations and keep WorkUnits for the step-bearing kinds only.
- **Cost-model admission is a modeling bet.** It converges toward cost-class pool routing once the indivisible-step
  reality is respected — which is close to what request-level pooling + a fleet planner already do. The fine-grained
  interleave win has a *narrow* window (many small concurrent jobs); it should be argued on that window, measured.
- **The clean-slate premise is the elephant.** This document deliberately ignores migration. The org that would build
  it broke its own freeze 19 times and ships 20+ families, a live product, and a landed RL stack. A clean-slate
  rebuild is the highest-risk path that exists for *this* org; the responsible realization is to build v3 as a
  *parallel* engine around one forcing-function card (Cosmos3), prove it on the parity ladder, then migrate families
  onto it behind an adapter while everything keeps shipping — i.e., reach this architecture incrementally. That plan is
  out of scope here by request; it is non-optional in reality.
- **Quality is unmeasured.** C4 (artifact quality / human preference) and the eval system it needs do not exist yet,
  and they gate every product claim ("fast mode is equivalent", RL reward validity, distillation comparisons). Named
  as a required, currently-absent subsystem, not assumed.

---

## 18. Package layout

```text
fastvideo/
  card/         specs, components, loops, recipes, parity, checkpoints, capabilities   # the Model Plane
  loop/         driver, loopstate, workplan, policies (cfg, expert, precision, flowshift, conditioning)
  runtime/      engine, scheduler/{request,loop,batch,placement,transfer,admission}, workers, events
  cache/        keys, classes/{paged_kv, slab_kv, feature, residual, weight_fleet}, policies
  memory/       allocator, sleep_wake, reservations
  transport/    manifests, backends/{shm, cuda_ipc, nccl, nixl, kvbm}, relay
  parallel/     plans, mesh, process_groups, validation
  parity/       aligner, ladder, interleave_gate                                       # §9 is its own home
  extend/       observers, interceptors, cache_dit, registry, trust
  program/      specs, compiler, workflows
  request/      requests, sessions, artifacts, streams, cancel
  training/     rollout, behavior, rewards, weight_sync, methods                       # imports card/loop/runtime; never imported by them
  deploy/       cards, role_pools, dynamo_adapter
  integrations/ comfyui, dreamverse, livekit, diffusers
```

Enforced boundaries: `card/` imports no product/runtime; `runtime/` executes `card/` loops but defines no semantics;
`training/` may require behavior records but forks no loop; `integrations/` adapt external systems into core specs and
events, never bypass them. **`parity/` is a first-class package**, not a test folder — it is how the (recipe, runtime)
pair is kept honest.

---

## 19. Reference synthesis

| Source | Take | Constrain / reject |
|---|---|---|
| Cosmos3 (official + port) | Shared model instance across reasoning/diffusion/action/sound; packed multimodal sequences; component+scheduler parity matrices | A strong `ModelCard`, not the framework; no Cosmos-specific branching in global runtime |
| vLLM core | Running-first scheduling, reservation-before-admission, model-owned state, encoder/KV cache managers, CuMem sleep/wake, CUDA-graph dispatch, KV-connector split | Token scheduling is one WorkUnit kind; never full-graph compile |
| sglang `multimodal_gen` | Role pools, request lifecycle, capacity dispatch, transfer manifests, disagg state machine, cache-dit integration | No large mutable `Req`/`ForwardBatch` as the stable API; not single-item diffusion scheduling |
| vLLM-Omni | Frozen pipeline spec separate from deploy YAML (verified, adopt); `OmniConnectorBase` + `chunk_ready` readiness; **`SupportsStepExecution` as loop-inversion prior art** (opt-in, Qwen-Image-only — we generalize to always-on); TP-rank- and CFG-branch-aware KV-copy transfer; **`CFGParallelMixin` proves CFG-as-policy over one shared denoise body** (§5.3); 3 separate cache subsystems confirm per-class pools | Expresses shared-weight MoT (`bagel`/`lance`) only as **one opaque request-scheduled stage** the scheduler never sees inside — no step visibility, no cross-request batching by default; cross-stage KV is a *copy*, not a shared live cache; **no cost model** (per-stage count budgets); readiness-parking, not credit flow; RDMA = Mooncake/Mori/Yuanrong, not NIXL/NCCL |
| sglang-omni | The `next/wait_for/merge_fn/stream_to` edge vocabulary; Relay transport + **credit-based flow control** (this is sglang-omni's, not vllm-omni's) | Stages own disjoint weights; hybrid AR+diffusion only as AR-stage → DiT-stage; per-model bootstrap duplication |
| Dynamo | Fleet routing, disagg role pools, KV-aware routing, KVBM, SLA planner, ModelExpress cold-start/weight streaming | Orchestrates engines; never the engine core. Export a `DeploymentCard` + cost model to it |
| diffusers Modular | `ComponentSpec`/`modular_model_index.json` interchange; Guiders ≈ CFG policies | A Python pipeline interpreter is not the performance boundary; import is lossy |
| xDiT | DiT parallelism catalog (USP, ring/ulysses, PipeFusion, CFG-parallel, DistVAE) + world-size validation | Parallelism lives in the runtime + card, not a wrapper-per-model library; `pp_patch` invalid for causal |
| TorchTitan | Named mesh axes, `ParallelDims` validation, ModelSpec discipline, TorchStore weight-sync, batch-invariance utils | Adopt the discipline, not the stack; DCP/TorchStore don't reshard — `WeightSyncPlan` owns layout |
| verl-omni / miles / cosmos-rl | Rollout adapters, per-step capture, async rewards, group-relative advantage, TIS/MIS, deterministic/batch-invariant modes, per-payload `weight_version`, AIPO/off-policy masking | The two-runtime tax is the thing to delete; capture behavior *in* the serving loop, not after the fact |
| ComfyUI | Workflow graph, node-signature cache, model memory management, App-Mode (workflows-as-products) | Compile to `Program`; dynamic node execution is not the serving/training core; GPL hygiene |
| Dreamverse | Sessions, prompt memory, typed media IPC, cancellation, the duty-cycle capacity reality, the preference-data flywheel | Product/session behavior is first-class in the request plane, never merged into the model core |
| LiveKit | Realtime sessions, push audio/video, interruptions, turn/activity state, frame+PTS streaming | Realtime triggers only when they fire (<100ms interactive); don't force RTC onto offline jobs |
| Thinking Machines (batch-invariance) | The C2/C3 mechanism: batch-invariant kernels for bitwise rollout↔train identity | Scoped to goldens; the conservative baseline governs admission |

---

## Final position

```text
A model card is a (recipe, runtime) pair with a parity obligation.
The model owns loop semantics; the runtime owns loop lifecycle.
One resident instance runs many loops; one scheduler runs their steps in one currency.
Caches are correct by key; parity is correct by test; the interleave gate is non-negotiable.
Training records behavior on the same loops it serves.
Deployment places and routes; products stream artifacts; neither defines the model.
```

This is the ceiling: a model-native runtime where omni is native, train and serve are the same loops by construction,
correctness is a typed contract you can sign, and the (recipe, runtime) flywheel is real. The constraint we removed to
see it was migration. Putting that constraint back is the next document, not this one.
