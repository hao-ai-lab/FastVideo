# FastVideo v4 — The Unified Design (as built in `v2/`, validated by joint LM+generator RL)

> **What this document is.** `design_v3.md` is the north star — the argument for a model-native runtime
> for the (recipe, runtime) era. This document, v4, is the *unified* design: it folds the v3 thesis
> together with **what was actually built and tested** in the `v2/` package, and with the one workload
> that most stressed the design — **UniRL/PromptRL-style joint LM + generator reinforcement learning**
> (arXiv 2510.17937; PromptRL, arXiv 2602.01382). v4 is not aspirational. Every structural claim below
> is backed by code in `v2/` and a passing test (97 tests, 15 files, two independent runners — `pytest`
> and a zero-dependency `v2/run_tests.py`). The neural forwards are toy numpy stand-ins (no GPU/torch in
> this environment, §12); the **control flow, contracts, scheduling, caching, parity gates, and training
> math are real**, which is the whole point of the (recipe, runtime) separation: swap the component
> factory for a torch adapter and the loops/policies/scheduler/caches/training are unchanged.

---

## Table of contents

1. The thesis (carried, with validation status)
2. The **three** signature ideas (v3 had two; the third is what the stress test proved)
3. Planes and dependency order (as enforced in `v2/`)
4. The Model Plane — `ModelCard` · `Loop` · `Program`
5. The driven-loop contract — `init/next/advance/finalize`
6. Runtime & scheduler — one WorkUnit, one budget, the interleave gate
7. Caches & the consistency ladder (the C2 split)
8. Training & RL on the same loops
9. **The UniRL/PromptRL joint-RL stress test (the centerpiece)**
10. Serving & fleet — our own stack, Dynamo optional
11. What v2 is and is not (honesty)
12. Package layout (actual)
13. Falsifiers, open questions, reference-synthesis delta

---

## 1. The thesis (carried, with validation status)

Three facts about video/omni generation dictate the architecture, unchanged from v3:

- **A deployable model is a post-training artifact.** A usable model is *created* by training (distillation
  for latency, QAT for precision, distillation+self-forcing for causal/world models). Every inference
  capability is therefore a **(recipe, runtime) pair**: the weights and the loop that produced-and-assumes
  them are one versioned object — a `ModelCard`. ✅ *Built:* `v2/card/` (`RecipeSpec` + `ParitySpec` on every
  card; 5 phase-1/2 cards + the unified card).
- **Video/omni systems are loop systems.** The work is iteration — denoise steps, AR tokens, chunked
  rollout, VAE tiles, optimizer steps, media chunks — not a single `forward()`. A runtime that reduces
  everything to `forward()` cannot schedule, batch, cancel, stream, reserve memory for, or capture the
  behavior of what actually runs. ✅ *Built:* `v2/loop/` driven-loop contract; `v2/runtime/` WorkUnit
  scheduler; every denoise step and AR token is a runtime-visible WorkUnit.
- **Omni models share weights across loop types within one request.** ✅ *Built and tested:* the Cosmos3 and
  BAGEL cards run **one resident MoT instance** driving an `ar_decode` loop then a `diffusion_denoise` loop
  on shared weights (`v2/tests/test_omni.py`).

The single invariant, restated and now demonstrable in code:

```text
Model cards own components, loops, recipes, and parity.   # v2/card/, v2/models/*/card.py
Programs compose loops into tasks.                         # v2/program/, v2/models/*/program.py
The scheduler executes loop steps as WorkUnits under one budget.   # v2/runtime/
Caches are correct by key, not by hope.                    # v2/cache/ (CacheKey)
Training records behavior on the same loops it serves.     # v2/training/ (engine never imports training)
Deployment places and routes; it does not define semantics.# v2/deploy/, v2/serving/
Products stream artifacts; they do not reach into the model.# v2/request/, v2/serving/server.py
```

## 2. The three signature ideas

v3 named two. The joint-RL stress test (§9) forced articulation of a third — the one that turns the design
from "an elegant serving runtime" into "the substrate for content-creation RL."

### 2.1 The (recipe, runtime) pair is a first-class, versioned, typed object

A model is not a checkpoint; it is a `ModelCard` owning components, loops, the recipe that produced the
weights, and the parity contract binding train-forward to serve-forward. You cannot ship the weights
without the loop they assume, or change the loop without re-proving parity. ✅ `v2/card/card.py`.

### 2.2 Driven loops: the model owns control flow, the runtime owns execution

A loop is a state machine — `init → (next → advance)* → finalize` — where the model describes *the next
step it needs* and the runtime decides *when and with whom that step runs*. Per-request state lives in a
typed `LoopState`, never in module globals, so interleaving requests through one instance cannot smear
state. ✅ `v2/loop/contracts.py`; the **interleave parity gate** (`v2/parity/interleave_gate.py`) proves
serial == interleaved bit-for-bit — the bet loop-inversion lives or dies on.

### 2.3 (new) One vocabulary spans every weight-sharing topology — including joint multi-expert RL

The Card/Loop/Program split is not specialized to one sharing pattern. The **same primitives** express:

| Topology | Components | Loops | Sharing | Card |
|---|---|---|---|---|
| Single diffusion | `transformer` | `diffusion_denoise` | — | `wan21` |
| **MoT omni** | **one** `transformer` | `ar_decode` **+** `diffusion_denoise` | both loops → **same** component | `cosmos3`, `bagel` |
| **Joint LM+gen RL** | `llm` **+** `transformer` (separate) | `ar_decode`→`llm`, `diffusion_denoise`→`transformer` | **disjoint** experts, one request | `unified` |
| **Cascade omni-speech** | `thinker` **+** `talker` **+** `vocoder` (separate) | `ar_decode`→`ar_decode`→`audio_decode` | **disjoint**, **chained** (talker reads thinker hidden state; vocoder reads talker tokens) | `qwen_omni` |

These four are the *full* spread of weight-sharing topologies — one module, one shared module across two
loops, two disjoint experts trained jointly, three disjoint experts chained — and every one of them is just
`shared_weight_components` bindings on `LoopSpec`s plus hand-off nodes in a `Program`: **no new primitive**.
MoT (shared) and UniRL (disjoint, jointly trained) are topological *opposites*; Qwen-Omni adds a third axis,
*depth* (a three-stage cascade with cross-stage conditioning). The signature idea: *the weight-sharing graph
is data on the card, not structure in the runtime.* This is what lets one engine serve a 1.3B single-DiT, a
30B MoT, a Qwen+FLUX joint-RL recipe, and a thinker→talker→vocoder speech model without four runtimes.
✅ Proven by `test_unified_rl.py::test_two_separate_experts_not_a_shared_mot` and
`test_thinker_talker.py::test_three_disjoint_experts_three_loop_types`.

## 3. Planes and dependency order

```text
   Products: Python · CLI · OpenAI API (v2/serving/server.py) · fleet (v2/serving, v2/deploy)
                          │  (thin: validate intent, make requests, subscribe to streams)
   Request / Session / Artifact / Stream                          v2/request/
                          │
   Program Plane  (typed loop programs)                           v2/program/, v2/models/*/program.py
                          │
   ┌──────────── Model Plane (CENTER) ────────────┐               v2/card/, v2/models/*/card.py
   │  ModelCard: components · loops · recipe ·     │
   │  parity · capabilities · caches · parallelism │
   └───────────────────────┬───────────────────────┘
                           │
   ┌────────────────────────┼────────────────────────┐
   │ Runtime / Scheduler      │ Training / RL          │  same loops, different capture
   │ v2/runtime/  WorkUnits   │ v2/training/  rollout · │
   │ + GPU-time budget        │ reward · weight-sync   │
   └────────────────────────┼────────────────────────┘
                           │
   Memory · Cache · Transport · Parallelism      v2/memory, v2/cache, v2/transport, v2/parallel
                           │
   Deployment / Fleet  (DeploymentCard → LocalFleet | Dynamo)     v2/deploy/, v2/serving/
```

**Enforced boundary (the load-bearing rule):** `training/` imports `card`/`loop`/`runtime`/`models`; **the
engine never imports `training`.** Rollout is the serve forward *plus capture* — a client of the engine,
not a fork of it. ✅ enforced by import structure; `v2/training/__init__.py` documents it.

## 4. The Model Plane — `ModelCard` · `Loop` · `Program`

The three-way split (the question the user asked v2 to explain and then stress-test):

- **`ModelCard`** — the *(recipe, runtime)* pair. Owns `components` (`ComponentSpec`: a lazy factory + a
  `load_id` pointing at the real torch module + `required_for`/`optional_for`/`resident_for`), `loops`
  (`LoopSpec`: kind, work-unit kind, `shared_weight_components`, `cache_policy`, `loop_factory`), `recipe`
  (`RecipeSpec`: method, `assumes_loop`, `assumes_precision`, `consistency_required`), `parity`
  (`ParitySpec`: consistency levels, interleave-required), `capabilities`, `caches`, `precision`,
  `parallelism`. `card.validate()` checks every loop's components and cache policies exist. *The card is the
  center; everything else is a view over it.* ✅ `v2/card/card.py`, `v2/models/*/card.py`.
- **`Loop`** — the runtime semantics: a driven state machine (§5). The model owns the transitions; the
  runtime owns the lifecycle.
- **`Program`** — composes loops + component-nodes into a task DAG (`ComponentNode` for kernel-free seams
  like text-encode/pack/vae-decode; `ModelLoopNode` to drive a loop to completion and bind its result into
  a slot). ✅ `v2/program/`, `v2/models/*/program.py`.

**Naming & rationale** (carried from the explanation v2 gave): "Card" because it is the spec sheet for a
deployable artifact — recipe on one side, runtime on the other. "Loop" because the atomic *runtime*
semantic in this domain is iteration, not `forward`. "Program" because a request is a *composition* of
loops, and composition deserves its own typed object distinct from any single loop. Versus FastVideo's
`ComposedPipeline` (stages own both semantics and orchestration, fused) and vllm-omni's pipeline-spec +
deploy-YAML (good split, but the diffusion loop is one opaque stage) — v2 separates *all three*: semantics
(Loop), composition (Program), and the deployable unit (Card).

## 5. The driven-loop contract

```python
class Loop:                                   # v2/loop/contracts.py (duck-typed; protocols, not ABCs)
    def init(self, req, model, ctx)  -> LoopState           # build per-request state (seeded rng, latents…)
    def next(self, st)               -> WorkPlan | Done     # describe the next step (kernel-free: a thunk)
    def advance(self, st, result)    -> LoopState           # fold the result; capture behavior if ROLLOUT
    def finalize(self, st)           -> LoopResult          # produce outputs + metrics + behavior
```

`next` is **kernel-free** — it builds a `WorkPlan` carrying a `run(model, override=None)` thunk, a
`ResourceRequest` (compute-seconds, resident bytes, peak activation), a `ShapeSignature` (for batch
coalescing), and optional `emits` (stream chunks). The runtime decides *when* to call `run`. `advance`
folds the returned `StepResult` back into `LoopState` and, under the `ROLLOUT` profile only, appends a
behavior slice to `st.trajectory`. This is how **train ≡ serve**: identical `next`/`advance`, the profile
flips capture on. ✅ `WanDenoiseLoop` (`v2/models/wan21/loop.py`), `ARDecodeLoop`
(`v2/models/omni/ar_loop.py`).

**Policies decompose the step body.** CFG, flow-shift, precision, and expert-routing are *policies*
composed into a loop, not branches inside it — `ClassicCFG.combine`, `FlowShiftPolicy.build_schedule`,
`PrecisionPolicy.cast`, `NoRouting/MoT routing`. CFG is a policy over *one* shared denoise body, so the
same body serves cond-only (guidance=1, the RL case) and full CFG (serving). ✅ `v2/loop/policies.py`,
`v2/tests/test_policies.py`.

## 6. Runtime & scheduler — one WorkUnit, one budget

Every step of every loop — a denoise step, an AR token, (later) a VAE tile or a transfer — is a `WorkUnit`
priced in **one currency: GPU-seconds**. Admission reserves a refundable `StepTicket` against a compute
budget and resident/peak memory against typed pools *before* a step runs; infeasible requests fail fast
(`AdmissionInfeasible`) rather than busy-spin; budget is refunded on completion. ✅ `v2/runtime/`,
`v2/memory/`, `v2/tests/test_admission.py`.

The **interleave gate** is the correctness spine: `run_serial(reqs)` must equal `run_interleaved(reqs)`
bit-for-bit, where interleaved round-robins one unit per request per tick across *all* loops (including
the two-loop unified program). A batch-of-1 gate is structurally blind to cross-request state smearing;
this one is not. ✅ `v2/parity/interleave_gate.py`; passes for all phase-1 models, the omni MoT cards, and
the unified two-loop program (`test_unified_serve_interleave_parity`).

## 7. Caches & the consistency ladder

**Caches are correct by key.** `CacheKey` carries every output-semantic field: `model_id`, `component_id`,
**per-component** `weights_version`, `adapter_versions`, `precision`, and content-hashed inputs. Per-class
pools (`feature`, `paged_kv`, …) match the granularity reality. The per-component version is load-bearing:
a transformer weight sync bumps only the transformer's version and invalidates only its cache entries — the
**frozen text-encoder's feature cache survives**, so a K-sample RL group encodes its shared prompt *once*.
✅ `v2/cache/keys.py`, `v2/models/common.py::cached_text_encode`, `v2/tests/test_cache.py`.

**The consistency ladder** (parity as a typed gate, declared on the card):

- **C0** structural · **C1** deterministic-seed reproducibility · **C2** rollout↔train identity ·
  **C3** batch-invariance · **C4** artifact-quality/preference (named, not yet built).
- **The C2 split** (the rung v3 added, now demonstrated by *two* methods that sit on opposite halves):
  - **C2 likelihood-free** — DiffusionNFT: no log-probs; identity is *behavioral* (seeded final sample +
    prediction-space MSE). ✅ `v2/training/methods/diffusion_nft.py`.
  - **C2 likelihood-based** — UniRL: a *per-step log-prob identity*. Recomputing each SDE step's Gaussian
    log-prob under the unchanged rollout weights reproduces the captured log-prob (⇒ PPO ratio == 1) to
    float32 trajectory-storage precision. ✅ `test_c2_likelihood_identity_at_rollout_weights` (max
    `|Δlogp| ≈ 3.3e-7`). *This is the parity gate the PPO importance ratio rests on* — if it failed, the
    ratio would be measuring a kernel gap instead of a policy change.

## 8. Training & RL on the same loops

```text
serve   : request      → program → loop → WorkUnits → artifacts
rollout : prompt batch → program → loop → WorkUnits → BehaviorRecords → rewards → update
```

The loop kernel is shared; the only difference is capture and training policy. The moat — and the place a
serving-only runtime structurally cannot follow — is that **the rollout forward *is* the serve forward plus
capture**: every serving optimization (distilled samplers, caches, step batching) is automatically a
rollout optimization, and there is one numerics surface (the ladder *measures* the gap rather than a
correction layer *papering over* it).

`v2/training/` is ~16× smaller than `fastvideo/train/` not because it does less *method math* — it is a
faithful CPU port (NFT is line-for-line vs the source `diffusion_nft.py`) — but because it carries none of
the GPU/FSDP/checkpoint/distributed infrastructure. The five+one methods:

| Method | Consistency | Roles (WeightSyncPlan) | Notes |
|---|---|---|---|
| `finetune` | C1 | student | plain flow-match regression |
| `dmd2` | C2 (free) | student + fake-score critic + teacher | distribution-matching distillation |
| `diffusion_nft` | C2 (free) | student + **old** (decay-blended behavior) + reference | samples from *old*, not student |
| `self_forcing` | C2 (free) | student + teacher | causal/chunked student |
| **`unified_rl`** | **C2 (based)** | **student (llm+transformer) + reference (both)** | **§9 — joint LM+gen RL** |

`WeightSyncPlan` ships a **role**, not "the weights" (student / EMA / old_policy / reference / teacher /
critic), and now a **component scope** so a sync versions and cache-invalidates *one* expert in isolation.
✅ `v2/training/weight_sync.py`.

## 9. The UniRL/PromptRL joint-RL stress test (the centerpiece)

> *The user's framing: "the LLM and the generator may be updated simultaneously by training recipes —
> in this case, reinforcement learning. This is critical to the future of FastVideo and content creation."*

UniRL-Zero / PromptRL train a **prompt-refiner LM** (Qwen) and a **flow generator** (FLUX) *together* under
one RL reward: the LM rewrites the prompt, the rewritten prompt conditions the generator, a single reward
on the generated image produces a **group-relative advantage**, and that one advantage drives **two
updates** — a token policy gradient on the LM and a FlowGRPO PPO update on the diffusion transformer. They
are *separate experts* (not weight-shared), updated *simultaneously*. This is the hardest thing v2 had been
asked to express: a training rollout that spans **two different loop types**, captures **two kinds of
log-prob**, and lands **two component updates** from **one reward**.

### 9.1 What was built (`v2/models/unified/`, `v2/training/methods/unified_rl.py`)

- **`unified` card** — two disjoint experts: `llm` (`ToyPromptRefiner`, a categorical policy over
  refinement actions — the Qwen role) bound to `ar_decode`, and `transformer` (`ToyDiT`, the FLUX role)
  bound to `diffusion_denoise`. Serve program: `text_encode → refine(ar_decode) → apply_refinement →
  denoise(diffusion_denoise) → vae_decode`.
- **`flow_sde_step_with_logprob`** (`v2/loop/sampler.py`) — the FlowGRPO **SDE rollout sampler** (stochastic
  step + per-step Gaussian log-prob), distinct from the deterministic ODE `flow_match_euler_step` used at
  serve time. Gated on `DiffusionParams.sde_rollout` (default `False`) so **the serve path is byte-for-byte
  unchanged** — the 82 pre-existing tests never see it.
- **`UnifiedRLMethod`** — the joint GRPO loop: per prompt, the LM *samples* a refinement action (capturing
  its categorical log-prob); K diffusion samples roll out under SDE (capturing per-step log-probs), with
  `num_skip_refinement` samples using the *original* prompt (partial refinement) so the advantage contrasts
  refined vs unrefined; one reward → group-relative advantage A = clip((r−mean)/(std+ε), ±5); the advantage
  drives **(a)** the LM via REINFORCE (`−A·logπ(a) + β·KL`) and **(b)** the DiT via FlowGRPO PPO
  (`max(−A·ρ, −A·clip(ρ,1±ε)) + β·KL`, ρ = exp(logp_cur − logp_rollout)); two learning rates
  (LLM_LR ≫ DIT_LR), two `WeightSyncPlan`s (`("llm",)` and `("transformer",)`).

### 9.2 What it exercises that nothing else in v2 did

1. **A rollout that spans two loop types** — `ar_decode` (refine) *and* `diffusion_denoise` (generate), not
   one. The training rollout drives the *same* shared loops the engine serves.
2. **Dual log-prob capture** — a categorical action log-prob (LM) *and* a Gaussian per-step log-prob (SDE
   diffusion), both feeding one advantage.
3. **Likelihood-based C2** (§7) — the per-step PPO identity, the counterpart to NFT's likelihood-free C2.
4. **A multi-component joint update** — two weight-sync plans, versioned and cache-scoped *independently*
   (the LM sync must not flush the frozen text-encoder's feature cache, and does not).
5. **Rollout sampler ≠ serve sampler** — a controlled §9.4 divergence (SDE for exploration at train time,
   ODE for determinism at serve time), gated so it cannot leak into serving.
6. **A flag for prompt-only vs joint** — `joint=False` freezes the generator (ODE rollout, no DiT update);
   only the LM learns. This is PromptRL's prompt-only ablation, as a one-line mode.

### 9.3 What it proved (the test results)

- **The LM actually learns** to pick the reward-favored refinement: `P(target action)` 0.125 → 0.95 (joint)
  and 0.90 (prompt-only) over 40 iterations.
- **The generator actually moves** in joint mode (finite FlowGRPO grads, PPO ratio ≈ 1.0, KL finite) and is
  **byte-for-byte frozen** in prompt-only mode.
- **The two experts version independently** (distinct weight versions; `text_encoder` stays at `v0`).
- **The two-loop serve program passes the interleave parity gate** unchanged.
- ✅ All in `v2/tests/test_unified_rl.py` (9 tests); full suite **91 passed**, zero regressions.

### 9.4 The verdict: the design *held*

The stress test required **no new primitive** — only (a) a second sampler in the sampler *library*, (b) two
boolean fields on `DiffusionParams`, (c) a gated capture branch in the existing `WanDenoiseLoop`'s
`advance`, (d) a new toy component, a new card, and a new method, and (e) one generalization:
`WeightSyncPlan` gained a `components` scope (a strict improvement that also makes the §7.1 cache-scoping
guarantee explicit). The loop contract, the runtime, the scheduler, the cache keys, the parity gate, the
program model, and the training-plane boundary were **untouched**. *A joint multi-expert RL recipe is a
new card + a new method, not a new engine* — which is exactly the claim §2.3 makes, now with a test behind
it. The refinements the test forced (the component-scoped weight sync; the explicit likelihood-based C2
rung; `guidance_scale=1` at RL time to keep the PPO recompute faithful) are improvements, not patches.

### 9.5 Second stress test — Qwen-Omni thinker→talker→vocoder (the cascade topology)

A second canonical vllm-omni model was ported to confirm the §2.3 claim from a different direction:
vllm-omni's `qwen2_5_omni` pipeline — a **thinker** (multimodal LLM → text), a **talker** (AR → speech
codec tokens, conditioned on the thinker's hidden state), and a **code2wav vocoder** (codec tokens →
waveform). vllm-omni expresses it as three opaque stages with `custom_process_input_func` hand-offs the
scheduler never sees inside; v2 expresses it as **three driven loops** (`v2/models/qwen_omni/`) — every
thinker token, talker token, and vocoder chunk a runtime-visible WorkUnit.

What it added beyond UniRL: a **three-expert, three-loop-type cascade** (`ar_decode → ar_decode →
audio_decode`) with **chained cross-stage conditioning** (talker reads the thinker's tokens *and* hidden
state — the "full payload" path; vocoder reads the talker's speech tokens) and **streaming codec→waveform**
via `AUDIO_CHUNK` WorkUnits. The loop kinds were *already* in the enum — `LoopKind.AR_DECODE`'s docstring
literally names "thinker/talker", and `LoopKind.AUDIO_DECODE` + `WorkUnitKind.AUDIO_CHUNK` were defined for
exactly the vocoder — so the port added a `ToyTalker`/`ToyVocoder`, a `VocoderLoop` (filling the anticipated
`AUDIO_DECODE` slot), a card, a program, and one tiny generalization (`ARDecodeLoop` gained a configurable
`prompt_slot` so two chained AR loops don't collide on the prefill slot). The contract, scheduler, caches,
and parity gate were untouched again. ✅ `v2/tests/test_thinker_talker.py` (6 tests, incl. the three-loop
interleave parity gate and the cascade-conditioning checks); full suite **97 passed**, zero regressions.

The two stress tests together span the topology space — *shared* (MoT), *disjoint-joint* (UniRL),
*disjoint-cascade* (Qwen-Omni) — and the design absorbed all three as **cards + methods, never engines**.

## 10. Serving & fleet — our own stack, Dynamo optional

Per the explicit instruction "*don't completely rely on Dynamo; we still need our own version*," v2 ships a
complete serving/fleet stack and treats Dynamo as one optional backend:

- **`AsyncEngine`** with role/stage pools, an `OpenAI`-compatible server on stdlib asyncio
  (`v2/serving/server.py`, with HTTP timeouts/size limits and SSE streaming), and a `DisaggregatedRunner`
  proven **bit-identical to inline** execution.
- **Connectors** with `chunk_ready` readiness + **credit-based flow control** (sglang-omni's model, not
  vllm-omni's readiness-parking).
- **`LocalFleet`** (our own placement/routing) and a `DynamoWorkerAdapter` (`v2/deploy/`) — the core exports
  a `DeploymentCard` + cost model *to* a fleet planner; it is never *defined by* one.
- Hardening from the serving review is in-tree (`v2/tests/test_serving_fixes.py`): pool-capacity leaks on
  cancel/error, duplicate-request-id deadlock, credit leaks, cost-model aliasing, SSE aclose on disconnect.

## 11. What v2 is and is not (honesty)

- **Is:** a faithful, CPU-testable realization of the v3 contracts — Card/Loop/Program, WorkUnit scheduling,
  typed caches, the consistency ladder, train≡serve, omni MoT, joint LM+gen RL, and a full serving/fleet
  stack — ~7,500 LOC of package + ~1,400 LOC of tests, 91 passing.
- **Is not:** a GPU runtime. The neural forwards are deterministic numpy toys (`v2/models/backend.py`);
  there are no real weights, kernels, FSDP, or NCCL. C3 (batch-invariance) and C4 (quality/preference) are
  named, not built. **The deferred next step (the user's standing request) is to port the component
  factories to torch adapters** wrapping the real `fastvideo.models` modules — at which point, by
  construction, the loops/policies/scheduler/caches/parity/training code does not change.

## 12. Package layout (actual `v2/`)

```text
v2/
  card/        ModelCard, ComponentSpec, LoopSpec, RecipeSpec, ParitySpec, instance, load_card
  loop/        contracts (LoopState/WorkPlan/StepResult/Done), policies (cfg/flowshift/precision/routing), sampler
  program/     ComponentNode, ModelLoopNode, Program, compiler
  runtime/     Engine (run/run_serial/run_interleaved), scheduler, admission, StepTicket
  cache/       keys (CacheKey, content_hash), per-class pools (feature, paged_kv)
  memory/      allocator, reservations, refundable budget
  transport/   manifests + backends (in-proc for the mini)
  parallel/    ParallelPlan, mesh, validation
  parity/      interleave_gate, consistency ladder, compare_outputs
  extend/      observers, interceptors
  request/     requests, params (DiffusionParams + sde_rollout), tasks (TaskType), streams, artifacts
  models/      backend (toy components) + common (cached_text_encode)
               wan21/ ltx2/ wan_causal/         # phase 1: T2V, distilled, self-forcing student
               omni/ cosmos3/ bagel/            # phase 2: MoT (one module, two loops); omni/ = ARDecodeLoop + VocoderLoop
               unified/                         # §9: two experts, two loops, joint RL
               qwen_omni/                       # §9.5: thinker→talker→vocoder, three experts/three loops
  training/    rollout, behavior, rewards, weight_sync, methods/{finetune,dmd2,diffusion_nft,self_forcing,unified_rl}
  serving/     AsyncEngine, pools, DisaggregatedRunner, connectors, OpenAI server
  deploy/      DeploymentCard, LocalFleet, DynamoWorkerAdapter
  tests/       15 files, 97 tests          run via `pytest v2/tests/` OR `python3 v2/run_tests.py`
```

**Enforced boundaries:** `card/` imports no product/runtime; `runtime/` executes `card/` loops but defines
no semantics; `training/` requires behavior records but forks no loop and is never imported by the engine;
`serving/`/`deploy/` place and route but define no semantics. `parity/` is a first-class package, not a test
folder — it is how the (recipe, runtime) pair is kept honest.

## 13. Falsifiers, open questions, reference-synthesis delta

The v3 falsifiers stand (step-level scheduling must beat request-level on a real duty-cycle trace; the
general WorkUnit scheduler may be over-general for non-step kinds; cost-model admission is a modeling bet;
the clean-slate premise is out of scope by request; quality/C4 is unmeasured). The stress test **retired one
open question** — *"does the Card/Loop/Program split generalize beyond serving and MoT, or will joint
multi-expert RL force a redesign?"* — answered: it generalizes, no redesign (§9.4). Two new questions it
**opened**:

- **Throughput of two-loop rollouts.** A joint rollout interleaves an `ar_decode` and a `diffusion_denoise`
  loop per sample; the homogeneous-group batching win (K identical diffusion samples) is intact, but the LM
  refine step is a serial prefix. *Open:* on GPU, does the refine step want its own role-pool (LM served on
  separate hardware from the generator, à la disaggregated prefill), and does the WorkUnit cost model price
  the two loop kinds correctly enough to co-schedule them? — measure on the torch port.
- **BehaviorRecord size for joint RL.** Dual log-prob capture (token + per-step SDE) plus the SDE
  trajectory is heavier than NFT's seeded-sample capture. *Open:* confirm the opt-in instrument sizing holds
  for the LM+gen case at real resolution.

**Reference-synthesis delta (added to v3's table):**

| Source | Take | Constrain / reject |
|---|---|---|
| **UniRL-Zero / PromptRL** (arXiv 2510.17937, 2602.01382) | Joint LM-refiner + flow-generator RL under one reward; FlowGRPO SDE-prefix/ODE-tail with per-step log-probs; group-relative advantage → token-PG **and** PPO; partial-refinement contrast; prompt-only vs joint ablation; two LRs | Expresses the two experts and the joint update as a **card + method**, not a bespoke trainer; the SDE rollout sampler is a *library* entry gated behind `sde_rollout`, never the serve path; the PPO ratio rests on the **likelihood-based C2** parity gate, not on hope |
| **vllm-omni `qwen2_5_omni`** (thinker-talker-vocoder) | The canonical 3-stage omni-speech cascade: thinker (MM-LLM→text) → talker (AR→speech codec tokens, conditioned on thinker hidden state) → code2wav vocoder (tokens→waveform), streaming | Three opaque request-scheduled stages with `custom_process_input_func` hand-offs become **three driven loops** (ar_decode×2 + audio_decode), every token/chunk a step-visible WorkUnit; the cross-stage hand-offs are explicit `Program` nodes, not stage-boundary callbacks; the vocoder fills the pre-declared `LoopKind.AUDIO_DECODE` slot |

---

## Final position

v3 argued that the atomic unit is the (recipe, runtime) pair and that a model-native, loop-driven runtime is
what the post-training era requires. v4 is the same argument, now *built and measured* in `v2/`: the
contracts hold under serving, under MoT omni, under a thinker→talker→vocoder **cascade**, and — the hard
case — under **joint LM + generator reinforcement learning**, where one reward simultaneously updates two
separate experts driven by two different loop types in one request. Those workloads — UniRL/PromptRL's
joint RL (central to the future of content creation) and vllm-omni's omni-speech cascade — both fit inside
the design as *a new card and a new method/loop*, with no new runtime primitive. The design's bet is
therefore not that it is elegant — it is that **the weight-sharing topology and the training recipe are
data on a card, so a new frontier recipe is a card, not a rewrite.** The remaining work is the GPU port,
where, by construction, the loops, scheduler, caches, parity, and training plane do not change — only the
component factories do.
