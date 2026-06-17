# FastVideo v4 — The Unified Design (as built in `v2/`, validated by joint LM+generator RL)

> **What this document is.** `design_v3.md` is the north star — the argument for a model-native runtime
> for the (recipe, runtime) era. This document, v4, is the *unified* design: it folds the v3 thesis
> together with **what was actually built and tested** in the `v2/` package, and with the one workload
> that most stressed the design — **UniRL/PromptRL-style joint LM + generator reinforcement learning**
> (arXiv 2510.17937; PromptRL, arXiv 2602.01382). v4 is not aspirational. Every structural claim below
> is backed by code in `v2/` and a passing test (148 tests, 25 files, two independent runners — `pytest`
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
| **N-way joint RL** | N `refiner_i` **+** `transformer` (all separate) | N×`ar_decode` **+** `diffusion_denoise` | **disjoint**, **N-way jointly RL-trained** | `multi_expert` |

These are the spread of weight-sharing topologies — one module, one shared module across two loops, two
disjoint experts jointly trained, three disjoint experts chained, *N* disjoint experts jointly trained — and
every one is just `shared_weight_components` bindings on `LoopSpec`s plus hand-off nodes in a `Program`: **no
new primitive**. MoT (shared) and UniRL (disjoint, jointly trained) are topological *opposites*; Qwen-Omni
adds *depth* (a cascade); `multi_expert` adds *count* (N>2, §9.7). The signature idea: *the weight-sharing
graph is data on the card, not structure in the runtime.*

**Where BAGEL lands (asked directly):** BAGEL is **MoT/shared-weight — the same row as Cosmos3.** Its card
binds one resident `transformer` to *both* `generate_text` (ar_decode) and `generate_image`
(diffusion_denoise) via `shared_weight_components=["transformer"]` (`v2/models/bagel/card.py`); the only
difference from Cosmos3 is the *capabilities* (image vs video+sound), not the topology. The honest nuance:
the *real* BAGEL/lance is a Mixture-of-Transformers — **co-resident** understanding/generation experts that
share self-attention but carry separate FFN weights. That is neither "fully shared" (the v2 toy collapses it
to one module) nor "fully disjoint" (UniRL): it is *partial* sharing, and the card expresses it through the
**expert-routing policy** (`NoRouting`→a real router) over one resident instance — a fifth point on the same
axis, no new primitive. So BAGEL sits at *one resident instance, two loops, partially-shared experts.*

A separate composition axis (not weight-sharing) is **cross-model chaining** — a `Workflow` over two
*distinct* cards (T2I→I2V, §9.6). That lets one engine serve a 1.3B single-DiT, a 30B MoT, a Qwen+FLUX
joint-RL recipe, a thinker→talker→vocoder speech model, *and* a FLUX→Wan two-model pipeline, with no extra
runtimes. ✅ Proven by `test_unified_rl.py::test_two_separate_experts_not_a_shared_mot` and
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
- ✅ All in `v2/tests/test_unified_rl.py` (9 tests); zero regressions in the full suite.

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
interleave parity gate and the cascade-conditioning checks); zero regressions in the full suite.

The two stress tests together span the topology space — *shared* (MoT), *disjoint-joint* (UniRL),
*disjoint-cascade* (Qwen-Omni) — and the design absorbed all three as **cards + methods, never engines**.

### 9.6 Cross-model composition — T2I → I2V (Programs vs Workflows)

"Text-to-image followed by image-to-video" probes a different axis than weight-sharing: composing across
*model instances*. Two observations frame it:

  * **Same-card multi-stage is already covered.** LTX-2's program chains `ltx2_base` → upsample →
    `ltx2_refine` — two diffusion loops on one instance with a latent hand-off
    (`v2/models/ltx2/program.py`). So a one-model two-stage T2I→I2V adds nothing new.
  * **Cross-model is the real case** (FLUX→Wan: a T2I model, then a *different* I2V model) and the
    single-instance program runner *cannot* express it — every `ModelLoopNode` resolves on the program's
    one resident instance, by design (that single-instance assumption is what the interleave-parity gate
    relies on).

The resolution is the two-layer split design_v3 §13 always named, now realized: a **`Program`** composes the
loops of *one* model (the hot path, step-interleaved, parity-gated, unchanged); a **`Workflow`**
(`v2/program/workflow.py`, the `ProgramKind.WORKFLOW` that was declared but unbuilt) composes *across* model
instances — each stage is a full `engine.run` on a (possibly different) registered card, with artifacts
threaded stage→stage. The `image_video` package ships two distinct cards (`flux-t2i`, `wan-i2v`) and a
`build_t2i_then_i2v_workflow` chaining them: stage 1 produces an image; the workflow passes it as an
`ImagePart`; stage 2's program folds the image into the I2V conditioning (a program node — `WanDenoiseLoop`
itself is unchanged) and denoises a video. Crucially, crossing instances is a *Workflow boundary*, not a
program loop step, so **each model keeps its own interleave-parity guarantee** (proven per-stage). Cost on
the per-step scheduling path: zero. ✅ `v2/tests/test_t2i_i2v_workflow.py` (12 tests: the video provably
depends on the generated image; two distinct instances; each stage standalone + parity-gated; plus the
naming/registration tests below).

**Naming & registering custom pipelines** (asked directly). A workflow is a *first-class servable*, named
and registered with the **same discipline as a card** — two levels, mirroring how cards work:

  * **Name = `workflow_id`, in the same namespace as `model_id`.** A workflow is addressed exactly like a
    model: `engine.run(request)` with `request.model_id == "image_video.t2i_i2v"` routes to the workflow;
    `engine.serves(name)` reports it; the OpenAI `/models` list includes it. Convention: a **dotted,
    namespaced** id — `<package>.<pipeline>` (`image_video.t2i_i2v`) — so it never collides with a card id
    (kebab, e.g. `flux-t2i`); `register_workflow` rejects a collision outright.
  * **Declared dependencies, validated.** A `Workflow.requires` is the list of cards it composes (derived
    from its stages — `["flux-t2i", "wan-i2v"]`), and `register_workflow` calls `workflow.validate(engine)`
    to fail fast if any required card is absent (design.md P7: declared, never inferred).
  * **Two-level registry, exactly like cards.** A *declarative catalog* — `_WORKFLOWS` in
    `models/__init__.py` (the cross-model analog of `_BUILDERS`/`_OMNI_BUILDERS`, and of vllm-omni's
    `pipeline_registry`) — maps `workflow_id → (builder, required-card-builders)`; **adding a custom
    pipeline is one line there.** `register_workflows(engine, only=[...])` brings up the cards and the
    workflow together. A live engine then holds validated `Workflow`s in `engine._workflows`, just as it
    holds instances in `_registry`. A standalone `WorkflowRegistry` class is available for out-of-tree/ad
    hoc catalogs (the `register_pipeline`-style escape hatch).

So "more custom ones" is: write the stage builders, add one `_WORKFLOWS` entry with a namespaced id, done —
it is then addressable, discoverable, and dependency-checked like any model. (Async/streaming dispatch of a
workflow servable through `AsyncEngine.submit` is the one piece not yet wired — the offline `engine.run`
path and `serves()`/listing are; it is the same routing hook, noted as the remaining step.)

### 9.7 N-way joint RL — arbitrary experts (asked: possible, or a rewrite?)

**Answer: possible, no substrate rewrite.** "Joint RL over more than two experts" decomposes into two
questions, and the design already answers the hard one:

  * *Can the substrate hold N trainable experts driven from one rollout?* Already yes. A card holds N
    components and N loops (Qwen-Omni shipped three); `WeightSyncPlan` is per-component (N independent
    versions + cache scopes); `get_grad_clip_targets` returns a *dict* (N entries); rollout/behavior/parity
    are per-loop, agnostic to N. **None of this is "two."**
  * *Was anything hardcoded to two?* Only the *method body* — `UnifiedRLMethod`'s two `pg_step`/`_dit_ppo`
    calls and two sync plans. `JointMultiExpertRL` (`v2/training/methods/joint_multi_rl.py`) replaces that
    with a **loop over an expert list**: N refiner LMs + one generator, one reward → one group-relative
    advantage → N token-PG updates + one FlowGRPO-PPO update, N+1 independent `WeightSyncPlan`s. The
    `multi_expert` card carries N refiners + a generator (N+1 experts, N+1 loops).

Demonstrated on **3 refiners + 1 generator** (and parameterized to arbitrary N — tested at N=1 and N=4): all
four experts update from one rollout and version independently. The genuinely interesting finding is *not*
architectural but a **credit-assignment** result, surfaced honestly:

  * `credit="per_expert"` (each refiner gets an advantage from its own reward term): **all N learn cleanly**
    (P(target) 0.17→~0.97 each).
  * `credit="shared"` (one reward → one advantage to all, faithful to UniRL): **works but is noisy** —
    some experts learn, others stall or regress (the classic shared-reward multi-agent variance: each
    expert sees the others as noise). The fix is per-expert reward decomposition — *a reward-shaping
    choice the design already supports, not a substrate change.*

So: N-way joint RL is a method that loops over a list, on a card that already allows N experts. The thing
people fear ("it'll need a runtime rewrite") is exactly what the (recipe, runtime) split and per-component
weight-sync were designed to prevent. ✅ `v2/tests/test_joint_multi_rl.py` (6 tests: per-expert learning,
shared-credit variance, N-scaling, prompt-only freeze, independent versioning).

> **One correctness fix this forced.** Adding `guidance_scale=1` for the C2 likelihood identity (§9.4) had
> made the toy generator's PPO target equal the velocity it already produced — a no-op gradient (the unified
> generator was "moving" only on ~1e-7 float noise). The right FlowGRPO surrogate moves the velocity toward
> the **max-likelihood velocity of the realized sample** (`flow_sde_ml_velocity`) — the policy-gradient
> direction, nonzero precisely at ratio==1. Now both the UniRL and N-way generators learn for real, and the
> C2 ratio==1 identity still holds (it is measured *before* the update). A real bug, found by the stress
> test, fixed in the sampler library — not the runtime.

### 9.8 Interactive world-model session (the realtime/Session plane)

The Session/realtime plane (§16) was the most-asserted, least-tested part of the design — *zero* coverage.
`WorldModelSession` (`v2/runtime/session.py`) drives the causal `chunk_rollout` loop as a long-lived
interactive session: each `.act(action)` resumes the world from **persistent cross-request state** carried
on the `Session` (its `kv_handle`), streams frames as chunks complete, and honors **step-boundary
cancellation** — using the exact runtime primitives the engine uses (`RuntimeLoopContext` + `LoopRunner`).
What it exercised that one-shot generation never does: (a) a loop that *continues* across requests (the
world is stateful — same action + different history → different frames; ✅ tested); (b) **transactional
cancellation** — a cancelled act raises `Cancelled` at a step boundary and leaves the world state
unchanged, so the session is resumable; (c) **no cross-session smearing** — interleaving a second session's
acts between this one's is bit-identical to running them apart (the §9.3 guarantee, now for sessions). The
only code added was a continuation seam in the chunk loop (`init` seeds context from a `world_context` slot;
default empty ⇒ the unchanged one-shot path) and the session driver. ✅ `v2/tests/test_world_session.py` (5).

### 9.9 End-to-end RL over a cross-model workflow

Combines §9.6 (cross-model workflow) and §9.7 (joint RL) into the sharpest test of the training-plane
boundary: train **both** stages of the T2I→I2V workflow from **one final-video reward**. `WorkflowRLMethod`
(`v2/training/methods/workflow_rl.py`) rolls out the *whole workflow* with SDE capture in **both** instances
(T2I diffusion → decode image → condition + I2V diffusion → decode video → score), then applies FlowGRPO PPO
to each stage's transformer under the *same* final-video group advantage, with two `WeightSyncPlan`s on two
*different* instances. The load-bearing result: **the earlier model (T2I) is trained by a reward computed on
the final video** — end-to-end credit across a model boundary — and it is *caused* by that reward, proven by
a control: a constant reward ⇒ zero advantage ⇒ neither generator moves (✅ tested). So "rollout == serve +
capture" holds for a *workflow*, not just a card; the training plane spans model instances with no new
primitive (it drives the same shared loops, threading the artifact exactly as the serve-time Workflow does).
✅ `v2/tests/test_workflow_rl.py` (4: two-instance SDE rollout, both-stages-train, the causal control,
independent versioning).

### 9.10 Heterogeneous WorkUnit co-scheduling (the §17 falsifier, mechanism half)

§17's named falsifier: *"the general WorkUnit scheduler may be over-general"* — is scheduling VAE tiles
through the same admission machinery as denoise steps worth it? The `VAE_TILE` kind had zero coverage.
`VAETileLoop` + the `wan-tiled` card (`v2/models/tiled/`) make tiled VAE decode a loop of `VAE_TILE` work
units, and the tests show: tiling is **exact** (tiled == one-shot decode, a C0 parity); the units are real
(one per latent row); and — the point — a `VAE_TILE` pipeline and a `DIFFUSION_STEP` pipeline **interleave
bit-identically** (the §9.3 gate holds over *heterogeneous* kinds) and **co-run in one interleaved batch**.
This validates the *mechanism* the falsifier questions: non-step kinds flow through the one budget without
special-casing. The *economic* half — whether it pays on a real GPU duty-cycle, vs collapsing tiles to an
in-loop call — remains a measurement for the port (the falsifier's experiment is unchanged). ✅
`v2/tests/test_tiled_scheduling.py` (4).

### 9.11 LTX-2 joint audio+video denoise (T2VS, per-modality guidance)

LTX-2 declared an `audio_vae` (`required_for={"t2vs"}`) but never used it — now it does. A single
two-stage denoise carries a **synchronized audio latent** alongside the video (the audio latent is
conditioned on the video latent, so they stay in sync), applies **per-modality CFG** (`guidance_per_modality`
— video and audio get separate guidance scales), threads both latents through base→refine, and decodes via
the video VAE *and* the audio VAE → `video` + `audio` artifacts. The audio path is gated on the request
asking for audio, so the default **T2V path is byte-for-byte unchanged** (the existing LTX-2 tests are
untouched). What it lights up: the `guidance_per_modality` policy (scalar→per-modality, no loop branching),
the `sound_vae`/`t2vs` capability, and multi-artifact fan-out from one synchronized loop — true joint A/V,
not a post-hoc audio bolt-on. ✅ `v2/tests/test_ltx2_av.py` (5: video+audio, audio↔video sync, independent
per-modality guidance, interleave parity, T2V-unchanged).

### 9.12 Content-adaptive control flow — cache-dit skip + early-exit

The signature payoff of loop inversion (§2.2): because the model owns control flow, the
`CacheDiTDenoiseLoop`'s `next()` can **reuse the cached velocity** when consecutive predictions barely
change (cache-dit / TeaCache — skip the DiT forward, a 1.5–2× inference win) and **early-exit** when the
latent converges — a *variable, content-dependent step count*, impossible in a `for t in timesteps`
runtime. It is an isolated `WanDenoiseLoop` subclass (threshold 0 ⇒ identical to the base, so nothing
else is touched). Skips stay close to the full run (rel video diff < 0.05 at ~40% steps skipped), and —
the load-bearing part — the **interleave parity gate still holds across requests with different step
counts** (ragged loops don't smear). ✅ `v2/tests/test_adaptive_compute.py` (4).

### 9.13 Nested workflows (recursive composition)

A `workflow_id` is a servable in the same namespace as a model, so a `WorkflowStage` can invoke a
*workflow* exactly as it invokes a model — `engine.run` routes it either way. A workflow whose first stage
IS the T2I→I2V workflow (then extends its video) runs end-to-end; `requires` lists the inner workflow and
validation recurses. Cycles are prevented at **registration** (a self-referencing workflow `requires`
itself, which isn't served yet), with a **run-time cycle guard** (`engine._wf_running`) as defense-in-depth.
So "composition is data" scales to arbitrary depth, not just two flat stages. ✅
`v2/tests/test_nested_workflow.py` (5).

### 9.14 Live weight-sync under in-flight serving (the RL flywheel's hardest correctness)

Collocated RL serves rollouts and receives weight updates on the same instance. The hazard: swapping
weights *while a denoise loop is mid-flight* makes that request a half-and-half of two policies — its
captured log-probs describe a rollout that never happened, and training silently corrupts. The
`WeightSyncController` makes the lifecycle explicit — **freeze admission → drain in-flight → transfer
(per-component) → bump version + invalidate that component's caches → resume** — and the tests prove it:
a mid-flight swap *does* corrupt (shown, as the thing to avoid); draining first leaves the in-flight
request **bit-identical to the no-sync baseline** (it finished on its start weights) while a post-sync
request reflects the new weights; and the sync bumps only the transformer's version, so the **frozen
text-encoder's feature cache survives** (a shared prompt still reuses across the sync). This is the moat's
hardest correctness surface — the reason verl-omni/miles chose the two-runtime tax — proven at the
drain-semantics level. ✅ `v2/tests/test_weight_sync_live.py` (3).

### 9.15 Reward-model-as-a-served-card (REWARD_BATCH)

In real RL the reward is a *model* (PickScore/CLIP/a VLM judge), not a heuristic. A reward model is now
just another card — a `scorer` component + a `score` loop emitting **`REWARD_BATCH`** work units (the kind
had zero coverage) — so a learned reward is loaded, priced, scheduled, and place-able on its own pool. A
`ServedRewardScorer` drop-in-replaces the numpy scorer (`method.scorer = ServedRewardScorer(reward_inst)`),
turning *any* RL method into RLHF/RLAIF with **no method change** — the K rollout samples score as batched
`REWARD_BATCH` units. ✅ `v2/tests/test_served_reward.py` (4: REWARD_BATCH loop, drop-in interface,
determinism, DiffusionNFT driven by the served reward).

These five (§9.11–§9.15) cover the remaining stress tests from the audit: a product-grade joint-A/V model,
the cleanest proof of model-owned control flow, recursive composition depth, the RL flywheel's hot-swap
correctness, and the reward plane composing with serving — each a new card / loop / method / controller,
**no new runtime primitive**.

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
  program/     ComponentNode, ModelLoopNode, Program; Workflow (cross-model composition, §9.6)
  runtime/     Engine (run/run_serial/run_interleaved), scheduler, admission, StepTicket
  cache/       keys (CacheKey, content_hash), per-class pools (feature, paged_kv)
  memory/      allocator, reservations, refundable budget
  transport/   manifests + backends (in-proc for the mini)
  parallel/    ParallelPlan, mesh, validation
  parity/      interleave_gate, consistency ladder, compare_outputs
  extend/      observers, interceptors
  request/     requests, params (DiffusionParams + sde_rollout), tasks (TaskType), streams, artifacts
  runtime/     engine, scheduler, admission; session.py = WorldModelSession (§9.8 interactive plane)
  models/      backend (toy components) + common (cached_text_encode)
               wan21/ ltx2/ wan_causal/         # phase 1: T2V, distilled, self-forcing student
               omni/ cosmos3/ bagel/            # phase 2: MoT (one module, two loops); omni/ = ARDecodeLoop + VocoderLoop
               unified/                         # §9: two experts, two loops, joint RL
               qwen_omni/                       # §9.5: thinker→talker→vocoder, three experts/three loops
               image_video/                     # §9.6: flux-t2i + wan-i2v, cross-model T2I→I2V workflow
               multi_expert/                    # §9.7: N refiners + generator, N-way joint RL
               tiled/                           # §9.10: VAETileLoop, VAE_TILE units co-scheduled w/ denoise
               ltx2/ (audio_vae)                # §9.11: joint A/V denoise (T2VS, per-modality guidance)
               adaptive/                        # §9.12: CacheDiTDenoiseLoop (cache-dit skip + early-exit)
               reward/                          # §9.15: reward-model card, REWARD_BATCH work units
  training/    rollout, behavior, rewards (+ServedRewardScorer), weight_sync (+WeightSyncController),
               methods/{finetune,dmd2,diffusion_nft,self_forcing,unified_rl,joint_multi_rl,workflow_rl}
  serving/     AsyncEngine, pools, DisaggregatedRunner, connectors, OpenAI server
  deploy/      DeploymentCard, LocalFleet, DynamoWorkerAdapter
  tests/       25 files, 148 tests         run via `pytest v2/tests/` OR `python3 v2/run_tests.py`
```

**Enforced boundaries:** `card/` imports no product/runtime; `runtime/` executes `card/` loops but defines
no semantics; `training/` requires behavior records but forks no loop and is never imported by the engine;
`serving/`/`deploy/` place and route but define no semantics. `parity/` is a first-class package, not a test
folder — it is how the (recipe, runtime) pair is kept honest.

## 13. Falsifiers, open questions, reference-synthesis delta

The v3 falsifiers stand, but several have been **half-resolved** by the stress tests: (a) *"the general
WorkUnit scheduler may be over-general for non-step kinds"* — its **mechanism** half is validated (`VAE_TILE`
and `REWARD_BATCH` units interleave/schedule through one budget, §9.10/§9.15); its **economic** half (does
it *pay* on a real duty-cycle vs an in-loop call) is unchanged, deferred to the port. (b) The **realtime**
plane is no longer un-exercised — sessions, persistent KV, step-boundary cancellation (§9.8) — though the
<100ms latency target is still a GPU-only measurement. (c) The **moat's hot-swap correctness** — weight-sync
*under in-flight serving* — is demonstrated at the drain-semantics level (§9.14: in-flight requests finish on
their start weights, per-component versioning, frozen-encoder cache survives); what remains GPU-only is
boundary-stop at *step* granularity (vs request-drain) and the throughput of doing it under real load. Still
open: step-level scheduling must beat request-level on a real duty-cycle trace; cost-model admission is a
modeling bet; the clean-slate premise is out of scope by request; quality/C4 is unmeasured. The stress tests
**retired the central open question** — *"does the Card/Loop/Program split generalize beyond serving and MoT,
or will joint multi-expert RL / cross-model pipelines / interactive sessions / joint A/V / content-adaptive
compute / hot weight-sync force a redesign?"* — answered across §9.4–§9.15: it generalizes, no redesign. New
questions they **opened**:

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
contracts hold under serving, under MoT omni (where BAGEL and Cosmos3 both land), under a
thinker→talker→vocoder **cascade**, under **cross-model T2I→I2V** chaining, under **joint LM + generator RL**,
under **N-way joint RL** over arbitrary experts, under **interactive world-model sessions** (persistent
state, cancellation, streaming), under **end-to-end RL across a workflow** (a final reward training an
earlier model), under **heterogeneous WorkUnit co-scheduling** (VAE tiles interleaved with denoise steps),
under **joint audio+video** (LTX-2 T2VS with per-modality guidance), under **content-adaptive compute**
(cache-dit skip + early-exit), under **nested workflows** (recursive composition), under **hot weight-sync
while serving** (the RL flywheel's drain-correct lifecycle), and under a **served reward model**
(REWARD_BATCH). Every one of those fit inside the design as *a new card, method, loop, Workflow, session
driver, or controller*, with **no new runtime primitive** — and the only real bug any stress test surfaced
(a no-op generator gradient) was a fix in the sampler *library*, not the runtime. The design's bet is
therefore not that it is elegant — it is that **the weight-sharing topology, the composition graph, the
training recipe, the reward, and the session/sync lifecycle are all data over cards, loops, workflows, and
controllers, so a new frontier capability is a card or a driver, not a rewrite.** The remaining work is the
GPU port, where, by construction, the loops, scheduler, caches, parity, and training plane do not change —
only the component factories do.
