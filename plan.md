# FastVideo Runtime — Aggressive Implementation Plan

**Companion to** `design.md` (v19) and `design_summary.md` · **Stance:** this plan trades interface stability for
speed. Where it deviates from design.md's conservative migration (§10), the deviation is flagged with **⚡**.
design.md remains the architectural authority; this is the execution order.

---

## 1. Rules of engagement

**We break, freely and early:**

- The public Python API: `generate_video(**kwargs)` and `SamplingParam` are **deleted**, not deprecated.
- Config schemas: `FastVideoArgs` (1,272 lines, 81 fields) stops being a public or threaded surface.
- `fastvideo/api/compat.py` (651 lines): **deleted in M1** ⚡ (design.md §6.6 shrinks it monotonically to Phase 5 —
  that policy existed only to honor signatures we are now licensed to break).
- CLI flags, YAML schemas, package layout, `fastvideo.api` exports, ComfyUI node params, every example.
- In-repo dependents (`apps/dreamverse`, `comfyui/`, `examples/`, `scripts/`) get **fixed in the same PR train** —
  we own them; no deprecation period, no shims.

**We never break, at any speed:**

- **Numerics.** Bit-identical loop parity and SSIM gates are not "conservative" — they are the definition of
  correct. Aggression applies to interfaces, never to outputs.
- **Model coverage** for the families that matter (tier list in §6 — the tail is a decision, not a casualty).
- The frozen legacy `fastvideo/training/` stack (N2) and the bit-exact porting methodology (N3).
- External users get **batched breakage**: all user-visible breaks land in at most two releases (R1 = request/config
  cut, R2 = engine default), each with a migration guide and a `fastvideo migrate` codemod — never a drip.

## 2. The sequencing argument (answering "fix omni request first, then separate the planes?")

**Yes to the first half. The second half should not be a project.** The three planes are not separated by moving
code into plane-named directories — today's monolithic stages would just get reshuffled and then rewritten when
loops invert. The planes are *born* from two cuts, and a third that is really a config change:

1. **The request-plane cut (M1)** — `OmniRequest` becomes the only currency crossing the boundary. Everything
   behind it is implementation. This is your "fix the omni input and request first," and it goes first because it
   is low-risk, it defines the vocabulary every later stage consumes, and it gets the user-facing pain over with
   while the codebase is still familiar.
2. **Loop inversion (M2)** — this *is* the pipeline/execution plane separation. Once families expose
   `init/step/finalize` step bodies, something other than the family must own iteration; that owner is the
   executor, and the execution plane exists by construction. Before inversion there is nothing for an execution
   plane to schedule — "separating" it would be an empty directory.
3. **The config cut (inside M1)** — the real coupling between planes today is `FastVideoArgs`: one 81-field object
   threaded through entrypoints, pipelines, stages, and executors, mixing deploy-time, model-time, and
   request-time concerns. Splitting it into `DeployConfig` / `ModelSpec` / `OmniRequest` (design.md §6.6's four
   layers) is the single highest-leverage "separation" action, and it's schema work, not architecture work.

So the order is: **M1 request+config cut → M2 loop inversion (planes now exist) → M3 engine on top.** Plane
separation is the *outcome* of M1+M2, not a milestone.

## 3. Milestones

Timeline assumes 3–4 engineers on the runtime critical path. Overlap is deliberate; gates are not ⚡-able.

### M0 — Baselines, harness, enforcement (weeks 0–3, overlaps M1)

The license for everything aggressive afterward. Not skippable, not shrinkable. M0 does not block M1 (which
changes no numerics) — the only hard rule is **no family's M2 migration starts before its baseline exists**.

- Merge the `feat/cosmos3-reasoning` chain (design.md sizes this alone at 2–3 engineer-months — it runs as its own
  track); seed SSIM references for the ~7 uncovered families.
- ParityAligner v0: record/compare named taps on *current* pipelines (it must exist before anything changes).
- **The enforcement package, on day one** (design.md §10 — the prior freeze was broken 19× for lack of exactly
  this): CI path gates (reject new `fastvideo/training/` files now; reject new `DenoisingStage` subclasses once the
  first M2 family lands), CODEOWNERS on the frozen and migrating paths, a named owner per milestone, and the
  inflow rule — new model families land on the new abstractions from the first Wan/Flux2 landing onward.
- Announce the M1 freeze window for in-flight PRs touching `fastvideo/api/`, `fastvideo_args.py`, entrypoints.

*Gate: every tier-A family has a recorded SSIM + activation baseline; CI gates live.*

### M1 — The request-plane cut (weeks 1–4) → **breaking release R1**

The typed API is partway there: `VideoGenerator.generate(GenerationRequest)` is already the documented primary
entrypoint (`generate_video` carries a deprecation warning), and `fastvideo/entrypoints/openai/` already serves
`POST /v1/videos` and `POST /v1/images`. But the legacy path is still what's *used*: Dreamverse calls
`generate_video(**kwargs)` (`apps/dreamverse/dreamverse/video_generation.py:508`), as do ComfyUI and most
examples. M1 finishes the cut instead of bridging it:

- **`OmniRequest` / `OmniOutput` / `OmniEvent`**: evolve `api/schema.py`'s `GenerationRequest` in place — typed
  modality parts, `TaskType`, per-model `ModelOptions` registered blocks (formalizing the `api/matrixgame2.py`
  pattern), seeds/priority/streaming flags. `api/results.py`'s `Video*Event` types become `OmniEvent`.
- **Config: four layers, one owner each** (§6.6): extract `DeployConfig` (placement, parallelism axes, memory/
  offload, compile, plugins) from the runtime third of `FastVideoArgs` + `EngineConfig`/`ParallelismConfig`;
  `ModelSpec` manifest v0 (manifest-first component resolution; today's name-detectors as fallback);
  `OmniRequest` absorbs every per-call field. CLI flags, OpenAI protocol models, and presets are **generated**
  from the schema.
- **Delete** ⚡: `compat.py` (651), `sampling_param.py` (411), `generate_video()`, the `fastvideo.api` legacy
  exports, `FastVideoArgs` as a *public* type. Internally it survives as a boundary-constructed shim for as long
  as anything still receives it: migrated families drop it per-family in M2, but unmigrated tier-B stages
  (`LegacyPipelineNode`) and the frozen `training/` stack (whose `TrainingArgs` subclasses it) carry it until M6 —
  it dies as a type with the tail, not before.
- **Fix in-train**: ComfyUI nodes (legacy-API callers), all `examples/` (~75 files, mostly mechanical),
  `scripts/`, docs. Ship `fastvideo migrate` (codemod: old kwargs/YAML → `OmniRequest`/`DeployConfig`).
- Internals unchanged: `ForwardBatch` is built *from* `OmniRequest` at the boundary; the executor and stages are
  untouched in M1.

*Gate: all SSIM suites unchanged; Dreamverse + ComfyUI + examples green on the new surface; R1 notes + codemod
published.*

### M2 — Loop inversion (weeks 4–10): the pipeline plane is born

- `DenoiseLoop` / `ARDecodeLoop` with `init/step/finalize`; runtime owns iteration; custom-step escape hatch from
  day one (the Cosmos3-port and self-forcing pattern is legitimate, §6.2.3).
- **Family order** (each lands step body + policies, and **deletes its legacy stage code in the same PR** ⚡ —
  continuous deletion, no end-of-plan cliff): **Wan 2.1/2.2 + Flux2 first, jointly** — design.md's rationale
  stands: together they exercise CFG variants, expert routing, chunk-KV, and the image path, so the step-body
  contract freezes only after all four are exercised → Wan-causal (self-forcing student) → LTX-2 → HunyuanVideo →
  Stable Audio → remaining image families. Unmigrated families keep running via `LegacyPipelineNode`.
- Policies: `CFGPolicy` (absorbs the 3 CFG copies), `AttnMetadataProvider`, `FlowShiftPolicy`, `PrecisionPolicy`.
- Extension core lands with the loop (it's why the loop is being rebuilt): observer bus, ParityAligner promoted to
  per-request observer, Profiler/NaNWatch, and **cache-dit as the first interceptor** (retiring `enable_teacache`).
- `forward_context.py` off the *migrated* inference path (194 references across ~68 files today: ~8 importer files
  in frozen `training/`, the rest spread across train/ models, tier-B inference stages, quantization, and tests);
  the module survives as a shim for frozen `training/` **and unmigrated tier-B stages** until M6 — what M2
  guarantees is that no migrated family and no new code touches it.
- **`train/` migrates per-family, immediately behind inference**: DMD2 and the landed DiffusionNFT (#1450) adopt
  the shared step functions as each family's body lands — `rl/common/sampling.py`'s loop is deleted, #1396
  grad-norm refs extended to the migrated methods (RL included).

*Gate, per family: old-vs-new loop bit-identical (ParityAligner) + SSIM + a recorded loop-overhead / batch-of-1
latency measurement (the baseline M3 gates against); for train/: seeded rollout latents identical, reward metrics
- grad-norms neutral. No family is ever dual-maintained.*

### M3 — Execution plane: engine + scheduler (weeks 8–14, overlaps M2) → **breaking release R2**

- `AsyncEngine` (queue, admission, cancellation-as-common-path, failure isolation); offline `VideoGenerator` keeps
  its name, becomes a thin sync wrapper that can bypass the queue.
- `StepScheduler` v0: multiplexes denoise steps across requests in a pool; budget currency = **predicted GPU-time**
  from a calibrated per-(model, phase, shape) cost table (the cost *model* matures later; the currency is right
  from day one). Carries the `ARDecodeLoop` contract; AR batching itself waits for its workload (N5).
- CacheManager v0: per-request chunk-KV slabs behind `KVHandle`; CFG-parallel axis (2-branch in practice).
- **Dynamo stock worker** (registration, health/drain, cost metrics), retiring the locked
  `dynamo/examples/diffusers/worker.py` pattern.
- **Dreamverse hard-cut** (per design.md Phase 2; the aggressive delta is doing it in one PR): `gpu_pool.py`,
  queue, warmup, and stream relay deleted and replaced by engine-client calls; the duty-cycle concurrency study
  runs on the result.
- Colocated weight-sync RPC + component-granular sleep/wake + `RolloutClient` (engine-client RL mode for #1450).

*Gate: serving load tests; batch-of-1 latency regression ≤ 2% vs the M2-recorded measurement; Dreamverse
single-session parity; RL engine-client seeded final-latent parity vs in-process; deploys under stock Dynamo.*

### M4 — Graphs, parallelism, multi-session (weeks 14–20)

- `PipelineSpec` graph IR: per-family pipeline classes shrink to **spec + step body + policies**
  (`create_pipeline_stages()` retires); LTX-2 and Hunyuan15+SR land as real fan-out graphs.
- Role pools + connectors (port `multimodal_gen`'s disagg state machine); declarative stacked-parallelism axes
  compiled to DeviceMesh; general cross-mesh `WeightSyncPlan`.
- ComfyUI workflow→spec compiler MVP (tier-1 ~20-node vocabulary) + weight/adapter fleet cache.

*Gate (design.md Phase 3's, in full): ≥2 Dreamverse sessions/GPU on the recorded duty-cycle trace, p95 within SLO
— this is also where the loop-inversion **falsifier** is evaluated (see §7); LTX-2 A/V full-fan-out end-to-end;
disaggregated-vs-colocated throughput benchmark; CPU-only topology validation suite; ComfyUI tier-1 workflows
compile and run with equivalence reports; spec-built pipelines SSIM-identical to M2 loop versions.*

### M5 — Omni/MoT native + RL hardening (weeks 20–30)

- Cosmos3 re-port onto specs: packed factored sequences, dual-pathway attention, reasoner paged KV, joint denoise,
  world-model `ChunkRollout`; `/v1/chat/completions`; AR continuous batching arrives **with** this workload (N5).
- Consistency ladder enforced end-to-end: C1 default in CI, C2 bitwise mode for goldens, Behavior Record opt-in;
  first GRPO-class method lands on the engine-client rollout path (log-prob drift becomes the gated metric).

*Gate: Cosmos3 150-test parity suite on the new runtime; reasoner pool efficiency — tokens/s/GPU at target
concurrent denoise throughput, with the ≥10×-vs-re-prefill sanity floor; C1 drift ≈ 0 on a Wan RL run with the
drift dashboard live.*

### M6 — The tail and the precondition (week 30+)

Continuous deletion (M1/M2) shrinks the final phase but does not eliminate it: what remains by M5 is the tier-B
tail on `LegacyPipelineNode` and the frozen `training/` stack — which is a *live consumer* of
`ComposedPipelineBase` and `forward_context`, so its retirement is the precondition, exactly as design.md Phase 5
states. M6 = execute the §6 tail decision (migrate or deprecate each tier-B family), retire `training/` per the
checklist, then delete `ComposedPipelineBase`, the legacy `DenoisingStage`, `forward_context.py`,
`FastVideoArgs`/`TrainingArgs`, and `RayDistributedExecutor` together. **4 loop copies → 1.**

## 4. Breakage manifest (user-visible)

| Release | What breaks | Replacement | Aid |
|---|---|---|---|
| **R1** (M1) | `generate_video(prompt, **kwargs)`, `SamplingParam`, `FastVideoArgs` as public type, `fastvideo.api` legacy exports, CLI flag names, YAML config schema, streaming event types (`Video*Event` → `OmniEvent`, `schema_version`'d from day one) | `VideoGenerator.generate(OmniRequest)`, `DeployConfig`, generated CLI/protocol, `OmniEvent` | `fastvideo migrate` codemod, migration guide, R0 pinned |
| **R2** (M3) | Default execution path becomes the engine (offline bypass preserved); server lifecycle (queue/admission semantics, job states) | `AsyncEngine` | guide; `OmniEvent` schema unchanged from R1 |
| after R2 | nothing user-visible — M4/M5 are additive | — | — |

## 5. Deviations from design.md §10, stated honestly

| design.md | this plan | why it's safe now |
|---|---|---|
| Phase 0 keeps `VideoGenerator`/CLI signatures; `compat.py` shrinks to Phase 5 | M1 breaks signatures, deletes `compat.py` ⚡ | the only argument for the shim was signature stability — explicitly revoked |
| Legacy code deleted at Phase 5 | per-family deletion at parity, M2 onward ⚡ | parity gate is per-family anyway; carrying dead code to a final phase only invites the 19×-broken-freeze failure mode |
| Phases strictly sequential | M2/M3 overlap ⚡ | the engine consumes step bodies, not finished families; the step-body contract freezes at the Wan+Flux2 landing |
| Phases −1 through 4 sized at 36–54 engineer-months | ~21–28 engineer-months (3–4 eng × 30 wks) ⚡ | the delta is real deleted work — no compat maintenance, no adapter upkeep, no dual-stack carry — plus M2/M3 overlap; treat 30 weeks as the aggressive case and 36–40 as the planning case |
| Unchanged | parity/SSIM gates (restored in full at every milestone), enforcement package (CI path gates, CODEOWNERS, inflow rule — now at M0), train/RL migration timing (design.md Phase 1 already migrates NFT), Dreamverse hard-cut (Phase 2 already prescribes it), N2/N3/N5, cost-model currency, Dynamo asks + fallbacks, schema versioning | aggression budget is spent on interfaces only |

## 6. Decisions needed before M0

1. **Tier the model zoo.** Tier A (migrated, coverage guaranteed): Wan 2.1/2.2, Wan-causal/self-forcing, LTX-2,
   Flux2, HunyuanVideo, Stable Audio, Cosmos3 (contingent on the M0 merge — it is not on `main` today), image
   families. Tier B (runs on `LegacyPipelineNode` until someone claims it, candidate for deprecation at M6):
   gen3c, matrixgame2/3, longcat, the rest. **Approve or edit the split** — it bounds M2.
2. **Release framing.** R1 as `v0.3.0` (pre-1.0 semantics, loud notes) vs holding breaks for a `v1.0` story.
   Recommendation: `v0.3.0` now — waiting taxes every milestone.
3. **Freeze windows.** M1 freezes `api/`/args/entrypoints PRs ~2 weeks; M2 freezes per-family stage PRs while that
   family migrates (days each). Needs maintainer sign-off.
4. **Staffing.** Critical path is M2's per-family step bodies — parallelizable per family after the Wan+Flux2
   reference lands. 3–4 engineers ≈ 30 weeks to M5 in the aggressive case (design.md's own sizing implies 36–40
   weeks at the same staffing — see §5); 2 engineers ≈ stretch ~1.5×. The Cosmos3-chain merge (M0) is its own
   2–3 engineer-month track and should be staffed separately from the runtime critical path.

## 7. Risks specific to the aggressive posture

- **In-flight PR collisions** with layout/schema moves → freeze windows (above) + landing schema cuts at
  milestone *starts*, not ends.
- **Community churn at R1** (ComfyUI users, script users) → codemod covers the mechanical 90%; the 10% that isn't
  mechanical (kwargs with changed semantics) is enumerated in the guide; previous version stays pinned and
  installable.
- **Parity harness becomes the bottleneck** — every aggressive deletion is licensed by it. Mitigation: it is the
  *first* deliverable (M0), and per-family migration PRs are template-driven (record → port → compare → delete).
- **Overlap risk (M2/M3)**: the engine team building against a moving step-body contract → the contract
  (`init/step/finalize` + `StepResult`) freezes at the *first* family (Wan), enforced by the same schema-version
  discipline as external surfaces.
- **The known unknown**: loop inversion at scheduler granularity has no production precedent (design.md §1). The
  falsifier stands, on design.md §11.6's schedule: the M3 duty-cycle study *publishes the targets*; the falsifier
  is **evaluated at the M4 gate** — if step-level multiplexing can't beat request-level serialization on real
  Dreamverse traces, StepScheduler retreats to request-level dispatch and the loop contract keeps only its
  streaming/preemption seams, with no family code changing — step bodies and the M1/M2 cuts retain full value.
