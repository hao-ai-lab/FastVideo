# v2 ← M\*: Architecture Gap-Analysis & Improvement Roadmap

**Status:** exploration, flagged for review. **Date:** 2026-06-19.
**Source paper:** *M\*: A Modular, Extensible, Serving System for Multimodal Models* (arXiv 2606.12688,
Stanford/UW/CMU; Jha, Sagan, Kamahori, …, Kasikci, S. Wang). It is a universal serving runtime for composite
multimodal models built on the **Walk Graph** abstraction (a model is a dataflow graph `G`; a request is a
*Walk* — a labeled subgraph — and the runtime executes walks). It beats vLLM-Omni (~20% lower T2I latency on
**BAGEL**, up to 2.64× on I2I), SGLang-Omni (2.7× TTS throughput on **Qwen3-Omni**), and native V-JEPA2
rollout (12.5×). It explicitly names **FastVideo's own** sparse/sliding-tile attention, xDiT/PipeFusion/USP,
Inferix, and FlashDrive as techniques integratable into the graph runtime.

**Method:** a 28-agent workflow — 6 parallel v2-subsystem maps → 10 M\*-dimension analyses, each
*adversarially verified against the actual v2 code* → synthesis + a completeness critic. The critic's
corrections and three P0 claims were then **spot-verified by hand** (file:line below). This doc folds those
corrections in; it is the corrected, authoritative synthesis.

---

## 1. Executive summary

v2 already implements the **harder half** of M\*'s thesis and in several axes **exceeds** it:

- v2's `Program` *is* M\*'s graph `G` (typed `ComponentNode`/`ModelLoopNode` + edges).
- v2's `shared_weight_components` *is* M\*'s cross-Walk node sharing — BAGEL/Cosmos3/LTX2 each bind two
  `ModelLoopNode`s to **one resident transformer** (`instance.component()` returns the same live object). This
  is the exact MoT serving property the omni cards in this repo already express.
- v2 adds three things M\* (serving-only) has **no equivalent for**: a required+validated per-loop **cost
  model**, a non-negotiable **interleave bit-parity gate**, and an **integrated training plane** (RL→distill
  flywheel driving the *same* serving Loop).
- The `extend/` plugin seam (interceptors/observers/registry with capability negotiation) is precisely the
  hook M\*'s "extensible / integrate FastVideo-STA, xDiT, Inferix, FlashDrive" call-out asks for — **v2
  already has the seam M\* only gestures at.**

What v2 lacks is M\*'s **declarative authoring layer above the substrate**, and — the key insight — *much of
that substrate is already authored but inert*: v2 has declared the metadata for "minimum components per
request" (`required_for`/`optional_for` on every omni card) and "branch as a cache axis" (`guidance_sig`,
`CacheKey`) but **never wired it to an executor**. The substrate is ~80% built and switched off.

**Highest-leverage cluster:** three small, parity-safe wires that turn on inert substrate and unblock the
BAGEL/Qwen-Omni/Cosmos3 latency wins M\* measured **on the exact models this repo already runs** — plus one
P1 that aligns v2 with the paper's headline "extensible" claim using a seam v2 already has.

### Verified P0 correctness findings (spot-checked by hand)
1. **Runner divergence (real bug).** `v2/runtime/engine.py:88` → `nodes = self.program.nodes`;
   `v2/runtime/disaggregated.py:96` → `nodes = self.program.active_nodes(self.request)`. The inline and
   disaggregated runners execute *different node sets*. ✅ confirmed.
2. **EOS is faked.** `v2/recipes/omni/ar_loop.py` docstring says "done on EOS/max_tokens"; `next()` (`:46-48`)
   checks **only** `max_tokens`. M\*'s marquee `DynamicLoop` use case (EOS) is unimplemented in the loop that
   serves the Qwen-Omni Thinker/Talker and Cosmos3 reasoner. ✅ confirmed.
3. **`required_for`/`optional_for` have zero runtime consumers** (grep outside `specs.py`/recipes/tests is
   empty). The min-components metadata is declared on every card and never read. ✅ confirmed.

---

## 2. Dimension table (corrected)

| # | Dimension | v2 status | Gap | Priority | Effort | Payoff | Action |
|---|---|---|---|---|---|---|---|
| 1 | Min-components per request (`required_for` + `when_task`) | substrate built, **inert** | real, cheap | **P0** | S | Consume `required_for` in `active_nodes`; unify `engine.py:88` onto `active_nodes`; deliver via registry/card builder so all ~40 cards inherit it |
| 2 | Real EOS + declarative `DynamicLoop` | early-exit emergent; **EOS faked** | real | **P0** | S | `ARDecodeLoop` honors `eos_id` + `req.sampling.stop`; add `LoopSpec.dynamic_stop` + `register_loop_stop`. **Training-enabling** (world-model rollout horizon) |
| 3 | CFG/branch as label over one paged KV pool | absent (`PagedKVCache` is a counter) | real | **P1** | L | `(namespace,label)` paged store w/ one budget; reuse `guidance_sig` for hash (NOT `partition_field`); by-ref via existing `InProcKVConnector`. AR path only (diffusion has no KV) |
| 4 | `extend/` plugin seam → integrate FastVideo-STA / Inferix | **seam exists, unused for attn** | real (paper headline) | **P1** | M | Expose FastVideo sparse/sliding-tile attention + Inferix block-diffusion as `Interceptor`/`EngineKind` plugins — the paper's named integration targets, on this repo's own code |
| 5 | `ParitySpec.output_determinism` (C3 distributional) | C3 rung defined, **0 users** | real, dormant | **P1** | S | Add field; `compare_outputs` consults it. **Training-enabling** (SDE/FlowGRPO stochastic rollouts) |
| 6 | Registry-driven delivery of #1 | present, not leveraged | integration | **P1** | S | Express `when_task`/min-components through `WorkflowRegistry`/card builders, not 3 bespoke recipe patches |
| 7 | Serving conductor + pluggable data plane | conductor exists (`serving/http.py`); **single-process transport** | real | **P2** | L | v2 already has the step-scheduled worker surface; gap is ZeroMQ/Mooncake + direct worker→worker tensor routing (today `InProcKVConnector` only) |
| 8 | Fleet/Dynamo placement + replicas | **live** (`deploy/fleet.py`,`dynamo.py`) | partial | **P2** | M | Fleet-level placement/affinity/replica is real & ≥M\*; missing piece is only the intra-engine `(node,Walk)→rank` map decoupled from model code |
| 9 | Per-node TP / SP + cross-rank transport | axis vocab **exists** (`sp` incl.); not wired to runtime | partial | **P2** | XL | Wire declarative degrees into runtime; Wan/LTX are **SP-native** (TP is a no-op there); populate `parallel_plan_hash` on the serving cache path |
| 10 | Named Walks + per-model state machine | `Program`=G, sharing real; no Walk/SM | real | **P2** | M | Defer until a *re-entrant* phase graph (Thinker↔Talker, rollout) needs it; #1 captures the min-components win without it |
| 11 | Declarative `Parallel/Sequential/Loop` IR | imperative loop classes | real (authoring) | **P2** | M | Thin Section IR lowering to flat `Program`; scope to one AR recipe |
| 12 | Streaming `ChunkPolicy` + `StreamBuffer` | causal-chunk emit **already ships** (`wan_causal`); `EdgeKind.STREAM` inert | real | **P2** | L | Declarative `ChunkPolicy` vocab over the existing chunk mechanism; needs concurrent producer/consumer runner (= pipelined scheduling). Inferix integration point |
| 13 | Speculative deferred-termination; loop-spanning CUDA graphs; N+1 prefetch; attn double-buffer | absent / per-step capture (14 cards) | real | **P3** | L | Gate behind a real GPU executor; unobservable on CPU-toy CI; loop-span needs an `allows_interleaving=False` carve-out |
| — | Cost model + interleave/consistency parity | **exceeds M\*** | none | **guard** | — | Do not regress; keep `step_cost_model` mandatory + `bit_identical` default |
| — | Integrated training plane (flywheel, weight-sync) | **exceeds M\*** | none | **guard** | — | Protect train==serve loop identity with a toy fixture |

---

## 3. P0/P1 deep-dives (sequenced)

```
PR-1 (P0) min-components ──┐
PR-2 (P0) real EOS        ─┼─► prereqs for honest "DynamicLoop" + min-component claims; both training-enabling
PR-3 (P1) output_determinism (independent)
PR-5 (P1) extend/ plugin: FastVideo-STA / Inferix as Interceptors (independent; highest paper-alignment)
PR-4 (P1) CFG-as-label paged pool ──► depends on PR-2 (AR loop is the only KV consumer)
```
PR-1, PR-2, PR-3, PR-5 are mutually independent; PR-4 depends on PR-2.

### PR-1 (P0) — Turn on the inert min-components substrate + fix runner divergence
- **Change.** Extend `Program.active_nodes(request)` (`v2/program/specs.py`) to also drop any node whose bound
  `ComponentSpec.required_for` (`v2/card/specs.py:144`) excludes `request.task` (and isn't in `optional_for`).
  **Fix the bug:** change `v2/runtime/engine.py:88` to `nodes = self.program.active_nodes(self.request)` so the
  inline `ProgramRunner` matches `DisaggregatedRunner` (`disaggregated.py:96`). Deliver the `when_task` gating
  through the **registry/card builder** (`recipes/__init__.py`, `program/workflow.py:WorkflowRegistry`) so all
  ~40 cards inherit it uniformly — not three bespoke `program.py` patches.
- **Why (this repo's models).** BAGEL T2I currently steps the AR-text loop and Cosmos3 t2v materializes the
  reasoner even though the cards declare `transformer required_for={'reason','t2i'}`, `vae required_for={'t2i'}`.
  On the GPU backend that is wasted resident-weight load + wasted steps on every single-modality request —
  exactly M\*'s "execute the MINIMUM components per request," delivered by consuming existing metadata.
- **Risk/invariant.** Validate in `ModelCard.validate()` that every active node's `reads` are produced by an
  active node for each declared `TaskType` (avoid dropping a producer). Pure node-id filtering ⇒ serial and
  interleaved still walk the same filtered list ⇒ §9.3 interleave bit-parity holds by construction. CPU-toy clean.

### PR-2 (P0) — Real EOS + declarative `dynamic_stop`  *(also training-enabling)*
- **Change.** In `v2/recipes/omni/ar_loop.py`, `advance()` reads the emitted token; if it equals the model
  `eos_id` (toy backend exposes `EOS=0`) or matches `req.sampling.stop` (`params.py:21`, currently dead),
  register termination; `next()` returns `Done()` on stop OR `max_tokens`. Add `StopRegistry` to `LoopState` +
  `register_loop_stop(name)` to the `LoopContext` protocol (`contracts.py:204`) and to
  `DisaggregatedRunner`'s `RuntimeLoopContext`. Add `LoopSpec.dynamic_stop: bool=False`, opt the AR cards in.
- **Why.** The docstring-vs-code lie sits in the loop serving Qwen-Omni Thinker/Talker and the Cosmos3 reasoner;
  M\*'s second named `DynamicLoop` use case (world-model **rollout horizon**) is exactly what `self_forcing` RL
  needs — so this is both a serving-credibility fix and a training enabler (raise its payoff accordingly).
- **Risk/invariant.** `dynamic_stop=False` is byte-identical back-compat. Must pass **all three** parity gates:
  serial==interleaved AND disaggregated==inline. **Not** in this PR: speculative deferred-termination (unobservable
  on CPU-toy, fights the interleave invariant — P3, gated on GPU executor).

### PR-3 (P1) — `ParitySpec.output_determinism` (close the dormant C3 hole)  *(training-enabling)*
- **Change.** Add `output_determinism: str = "bit_identical"` to `ParitySpec` (`card/specs.py:88`); make
  `compare_outputs` (`parity/interleave_gate.py:54`) consult it (`bit_identical` → today's exact check;
  `distributional` → a moment/tolerance check — land a simple moment match first; a real KS test is new code).
- **Why.** `ConsistencyLevel.C3` is defined and used by zero recipes; an SDE/FlowGRPO stochastic rollout cannot
  honestly declare its parity contract and would falsely fail the bit-identical gate. Additive; default unchanged.

### PR-5 (P1) — Expose FastVideo's own attention + Inferix as `extend/` plugins  *(highest paper-alignment)*
- **Change.** Use the existing `extend/{interceptors,observers,registry}.py` seam (capability-negotiated, with
  per-(request,branch) `plugin_state` that already passes the interleave gate) to register FastVideo's
  sparse/sliding-tile attention and Inferix-style block-diffusion as `Interceptor`s / an `EngineKind` plugin.
- **Why.** M\*'s title is "Modular, **Extensible**" and it explicitly lists FastVideo-STA, xDiT/PipeFusion/USP,
  Inferix, FlashDrive as integratable. v2 already has the seam M\* only describes — this is where v2 most
  directly answers the paper, using this repo's own attention code. Low risk (the seam + capability negotiation
  already exist and are tested).

### PR-4 (P1) — CFG/branch as a LABEL over one paged KV pool
- **Change.** Rewrite `PagedKVCache` (`cache/classes.py:155-172`) from a block *counter* into a real
  `(namespace,label)->[block-handle]` store with **one shared `total_blocks` budget** (M\*'s single-pool
  property). Reuse the existing-but-unpopulated `CacheKey.guidance_sig` (`keys.py:53`) for the hash. Thread the
  label through `ar_loop.py` (alloc/append/get per `(request_id, branch)`; prefill once per shared-prefix label;
  combine via `CFGPolicy.combine`). Wire `ResourceRequest.cache_blocks` (`contracts.py:64`, zero consumers) into
  admission per (class,label).
- **Why.** The dossier-identified driver of M\*'s BAGEL win (3 CFG contexts as 3 labels over ONE pool vs dense
  per-context). Targets AR_DECODE (BAGEL `generate_text`, omni Thinker); **correctly excludes diffusion**
  (Wan/LTX are bidirectional, no KV — their CFG stays dense-but-batched).
- **Corrections to bake in.** Do **NOT** add `branch_label` to `CacheKey.partition_field()` (CFG branches share
  embeddings; partitioning by branch is a semantic bug). Do **NOT** add a new by-ref type — reuse
  `InProcKVConnector` + `TransferManifest.cache_key`. Wiring `cache_blocks` admission is greenfield ⇒ effort **L**.
  CPU version proves label/sharing semantics; the real latency win needs a FlashInfer paged kernel (out of scope)
  — **merge** with a future "real KVCacheEngine" effort rather than landing isolated.

---

## 4. What v2 already does ≥ M\* — do NOT regress
1. **Required+validated cost model** on every `LoopSpec` (13-kind `WorkUnitKind`) — typed, pre-GPU-validated.
2. **Interleave bit-parity as a hard gate** (`parity.interleave_required=True` on 40+ cards). M\* has no such
   gate (its speculative scheduling deliberately wastes steps). Load-bearing invariant; every new primitive
   must pass it.
3. **C0–C4 consistency ladder** wired into RL methods, with first-divergence tap reporting. No M\* equivalent.
4. **Integrated training plane** — DiffusionNFT/DMD2/self_forcing, RL→distill flywheel, `WeightSyncController`
   hot weight-sync with drain-to-boundary + scoped cache invalidation, driving the **same** serving Loop.
   M\* is serving-only. Protect with a toy fixture asserting `rollout_loop` drives the served Loop object.
5. **CPU-toy parity for the whole stack** — loops/CFG/caches/parity/RL run in CI without a GPU. Every new
   primitive must ship a toy exercise (this is what makes all PRs above testable without H100s).
6. **Partition-not-flush cache invalidation** + four independent per-class pools.
7. **`extend/` plugin seam** with capability negotiation (a 4-step distilled card *rejects* a residual-skip
   interceptor) — M\* describes extensibility; v2 has the mechanism.
8. **Dynamo citizenship** (`deploy/dynamo.py`: one `DeploymentCard`+cost model, two consumers) — beyond M\*'s
   self-contained runtime.

---

## 5. Dropped / merged / deferred (and why)
- **DROP declarative `Parallel` as a CFG-execution win.** The runner walks nodes linearly (ignores
  `Program.edges`), so `Parallel` lowers to sequential sugar and the CFG 3-pass braid is already one
  co-scheduled `WorkPlan.run`; splitting it risks the interleave gate. Salvage only the no-op refactor
  extracting `branch_forward` from `WanDenoiseLoop._velocity`. Reassign `Parallel` to the placement workstream.
- **MERGE the full Walk/state-machine layer** into "defer until a re-entrant phase graph needs it" (PR-1 gets the
  min-components win with ~20 lines, no new abstraction). If built: the validator must check a walk's node-id
  order is a *subsequence* of `program.nodes` (not just membership) or the runner can reorder and break parity.
- **MERGE `StreamBuffer`/`ChunkPolicy` into pipelined-scheduling.** Causal-chunk emit *already ships*
  (`wan_causal/loop.py` per-chunk `StepResult.emit` + slab-KV); the gap is the declarative `ChunkPolicy` vocab
  + a concurrent producer/consumer runner. If built: keep all policies pure (per-request `StreamBuffer` history,
  not shared edge state) and restrict the bit-identical claim to the token-only handoff.
- **MERGE CFG-fan-out exec + cross-rank transport + PD loop-splitting into a multi-GPU-runtime program.** These
  need real collectives (`v2/distributed/` is a stub) and KV-by-reference (KV lives in `CacheManager`, not the
  transferable `slots`). **Keep cheaply now:** the *declarative* halves — per-component degree, `(node,Walk)`
  placement key with node-only fallback, `ReplicaSet` under `LocalFleet`, and populate `parallel_plan_hash` on
  the **serving** cache path (it is already populated in `training/behavior.py:40` — the gap is serving-only).
- **DEFER** speculative deferred-termination, loop-spanning CUDA graphs, N+1 prefetch, attention-plan
  double-buffer — all gated on a real GPU executor; benefit unobservable on CPU-toy CI. Keep the cheap
  `EngineKind` tag (`STATELESS|KV_CACHE|DIFFUSION`) now. Correct the stale `cudagraph.py:51-52` docstring
  (per-step capture ships in 14 cards, not just wan21).
- **RESCOPE per-node TP.** Wan/LTX use `ReplicatedLinear` + **sequence parallelism** (`sp`), not TP; the `sp`
  axis already exists in `parallel/plan.py:AXIS_NAMES`. The work is wiring degrees into the runtime, not
  inventing vocabulary; a `tp_size=2` "one-line activation" is a no-op for the shipped models.

---

## 6. The first integration test, if/when multi-GPU placement work starts
The **live Qwen-Omni 2-GPU bring-up** (Thinker on rank 0, Talker+Code2Wav on rank 1; see
`v2_debug_videos/vlm.md` Session 4) is the natural first validation target for any `(node,Walk)→rank`
placement work — it is the one place this repo already has real multi-rank composite-model execution.

---

## Anchor files for P0/P1
`v2/program/specs.py`, `v2/runtime/engine.py` (**line 88 fix**), `v2/runtime/disaggregated.py`,
`v2/recipes/omni/ar_loop.py`, `v2/loop/contracts.py`, `v2/card/specs.py`, `v2/cache/{classes.py,keys.py}`,
`v2/parity/interleave_gate.py`, `v2/extend/{interceptors,registry}.py`, `recipes/__init__.py` +
`v2/program/workflow.py` (registry-driven delivery).
