# Adversarial Review of `design.md` (v12) — FastVideo Next-Generation Inference Runtime

**Date:** 2026-06-11
**Method:** Multi-agent adversarial review. 9 fact-check agents verified 70 concrete claims against the repo, the local reference checkouts (`cosmos-framework/`, `dynamo/`, `vllm-omni/`, `~/sglang`, `~/vllm`, `~/miles`, `~/verl-omni`, `~/diffusers`, `~/torchtitan`, `~/xDiT`, `~/ComfyUI`, `~/sglang-omni`, `~/cosmos-rl`), and GitHub. 9 attack lenses (abstractions, scheduler/perf, memory/cache, training/RL, strategy, migration, internal consistency, omissions, external borrowings) plus a completeness critic raised 70 findings; every finding went to a refute-by-default verifier. 36 findings were refuted; this document contains only the 34 that survived (1 critical, 26 major — consolidated below where lenses converged — 7 minor), plus fact-check corrections.

---

## Verdict

The architecture survives its strongest attacks — loop inversion's expressibility, the typed-state hybrid, the N1/N5 scope discipline, the clean-room GPL posture, and the C2-for-batch-1-video argument all held under refutation attempts. What does not survive is:

1. **The migration plan**, which consumes its own substrate two phases before building it and rests on a "frozen legacy stack" premise this repo has already empirically falsified.
2. **Two load-bearing factual errors** about reference systems (vLLM's BlockPool page sizes, diffusers' loop ownership) that each drove a recorded design decision.
3. **A family of undesigned failure/memory/trust paths** that the multiplexing bet itself creates. One is critical.

---

## Critical

### C1. No failure-isolation or cancellation semantics for the multiplexed pool — the blast-radius problem the architecture itself creates
**Where:** §6.3.1; absent from §12.

Today one request per pool means one request's CUDA error is its own problem. Step-multiplexing changes the failure class categorically: a mid-step OOM/illegal-access/NaN from one request poisons the CUDA context and desyncs in-flight NCCL collectives for *every* co-scheduled tenant on the pool, including resident Dreamverse session caches. The doc designs none of the machinery: no SPMD-consistent abort broadcast (the dual of its scheduling broadcast), no request-fatal vs pool-fatal classification, no pool re-init + cache-invalidation policy, no partial-artifact semantics for fan-out graphs. "OOM" and request cancellation appear nowhere in 1799 lines; "abort" appears once (RL stragglers).

Ordinary cancellation is also missing — and vibe directing makes abandoning in-flight generations the *common* path. Worse, Phase 2 retires Dreamverse's `gpu_pool.py`, which today has a working sentinel-fd worker-death watch (`gpu_pool.py:542-586`), into engine-client calls — a reliability regression for the flagship customer if the gate ships as written. vLLM v1, the doc's own scheduler template, needed first-class machinery for exactly this (`ENGINE_CORE_DEAD`, `EngineDeadError`, `abort_requests`). Risk 4 covers only scheduling-decision divergence; the long-job-resilience known-gap is single-job-framed.

The abort path shapes the StepScheduler loop, the worker RPC surface, and CacheManager handle lifetimes — it must be designed *with* Phase 2, and by the doc's own standard ("absence reads as a decision"), this absence is an oversight.

---

## Major — reference-system misreads that drove recorded decisions

### M1. The single-BlockPool CacheManager rests on a property vLLM explicitly does not have: per-group page sizes
**Where:** §6.3.2 lines 555-559.

The sentence asserts two mutually exclusive properties. vLLM's one-pool/no-fragmentation guarantee exists *only because* physical bytes-per-block are uniform across all groups: `kv_cache_utils.py` asserts a single page size (`get_uniform_page_size`), and its docstring says verbatim that breaking this "is non-trivial due to memory fragmentation concerns." Groups differ only in tokens-per-block at equal byte size; the unification mechanism inflates the smaller group's `block_size`.

Apply that to FastVideo's groups: a text-KV page (~64 KB/layer) vs a latent-frame slab (9.6–32 MB/layer for 1.3B/14B causal Wan) is a 150–500× ratio — unification means a 500-token reasoner prompt strands a multi-MB slab per layer-group. The one vLLM path with multiple page sizes (DeepseekV4) statically partitions capacity at startup over a single global block-id free list, which is harmless when group demand is token-coupled (every token passes through all layer groups) but wasteful exactly when demand is workload-decoupled — FastVideo's regime, where text-KV and chunk-KV demand vary independently with request mix.

Since this misread is what reversed the two-pool sketch (recorded at line 280), the decision rests on a false premise: either chunk-KV stays uniformly fine-paged (losing the slab semantics the MoT "falls out naturally" story depends on), or the two-pool design returns and needs its own fragmentation/deadlock argument.

### M2. diffusers Modular is not loop inversion — the "strongest external validation" of the keystone doesn't validate it
**Where:** §5 line 277, §6.2.3 lines 426-428.

`LoopSequentialPipelineBlocks.__call__` raises `NotImplementedError`; every concrete family hand-writes `for i, t in enumerate(timesteps)` inside its own blocking wrapper (`wan/denoise.py:434`, `stable_diffusion_xl/denoise.py:701` — SDXL ships four such wrappers, the subclass forest again). The iteration is block-owned, invisible to any runtime — no init/step/finalize, no external driver, none of the properties §6.2.2 says inversion exists for (scheduling, interleaving, preemption, streaming, fair sharing). In scheduling terms it is the current `DenoisingStage` with a refactored body — i.e., it validates the Guiders/policy pillar but as evidence for inversion it is *equally consistent with the alternative the design rejects* ("keep loops in stages, make bodies pluggable"). The class also carries an explicit experimental warning.

Consequence: no surveyed system — vLLM, sglang, multimodal_gen, diffusers — implements runtime-owned diffusion iteration at scheduler granularity. Loop inversion is the design's most novel element with zero production precedent, and risk 3 (which admits novelty only for the hybrid AR+denoise slice) should say so instead of borrowing validation the reference doesn't provide.

### M3. Cost-currency scheduling drops the memory half of vLLM's admission — and memory is never a scheduling resource anywhere in the design
**Where:** §6.3.1 (lines 476-547), §6.3.2; two lenses converged here.

vLLM's token budget is not a prediction — it is an exact cap checked *in the same loop as memory admission* (`allocate_slots` per request, preempt on allocation failure; activation memory separately bounded by a profiled worst case). The design takes the accounting structure, swaps the currency for a *forecast* (predicted GPU-time), and drops the memory dimension entirely: latents, conditioning sets, CFG duplicates, and activation peaks live in `RequestState`, explicitly outside the CacheManager, and nothing bounds how many concurrent LoopStates a pool admits — for a workload the doc itself calls memory-bound (line 499). Two items that each fit alone can jointly OOM, and a GPU-seconds currency cannot see it; combined with C1, that OOM is a pool-wide event. "Preemption only at step boundaries" never defines what happens to a preempted request's multi-GB resident state (offload? drop-and-resume-from-LoopState? — different economics from KV recompute).

Related internal contradiction, verified: cost is "static and known at admission... a table lookup" (line 538), but the same cost model is cache-dit-aware (line 520) — DBCache skip decisions are runtime data-dependent residual comparisons, unknowable at admission.

**Fix:** the budget needs a memory axis (resident-state + peak-activation per schedulable item), admission needs a memory planner over RequestState, and preemption semantics must be specified. The Phase-2 "≥2 sessions per GPU" gate rests on unaccounted memory until then.

### M4. Punica cannot express ComfyUI LoRA semantics
**Where:** §9.4 lines 1376-1380 (also §6.3.2 lines 575-579). *Verifier rated minor-to-major; grouped here with the borrowings cluster.*

vLLM's `LoRARequest` carries one `lora_int_id` and no strength field; scaling is baked into `lora_b` at registration; the Punica wrapper maps one adapter index per token. ComfyUI traffic — the workload §9.4 names — is N stacked LoRAs per request with continuous user-set `strength_model` *and* `strength_clip`, routinely tweaked per generation. Pushing that through Punica means registering each (ordered-set, strengths) tuple as a synthetic concatenated adapter: near-zero cache-hit rate across strength tweaks, registration churn in the stacked GPU weight slots, and concatenated ranks colliding with `max_lora_rank`. "Strictly better than hot-swap-only" is unsupported without a composition layer that doesn't exist anywhere, including in vLLM.

---

## Major — execution-plane gaps

### M5. MoT mode multiplexing has no parallelism answer
**Where:** §6.3.1 lines 502-503 vs §6.3.4; Phase 4 gate.

The "mode multiplexer" claim assumes both loop types share one static pool layout (`parallel: [dp, cfg, sp, tp]`), but their optimal layouts are disjoint: denoise wants SP+CFG; AR decode is sequence-length-1 — SP has nothing to shard and CFG doesn't exist. On a `[cfg(2), sp(4)]` 8-GPU pool the reasoner either runs replicated (1/8 useful work, paged KV duplicated 8×) or needs TP — and TP-everywhere regresses the bread-and-butter denoise workload on the flagship pool. Per-phase re-layout of the same resident weights is not expressible in the §6.3.4 spec (one static stack per pool), and resharding machinery exists only for train↔rollout weight sync (§8.6). §6.3.1's own jumbo-step mitigation (split cost classes across pools) is structurally unavailable for MoT — AR steps and denoise steps are the same weights — so concurrent reasoner token latency is gated by indivisible 50–500 ms denoise steps.

A workable resolution exists (AR continuous batching data-parallel across the cfg×sp weight-replica axes onto TP subgroups, plus §6.3.3 per-pathway TP, plus routing pure-REASON traffic to differently-shaped pools), but the doc never states one, and the Phase-4 gate ("reasoner ≥10× faster than re-prefill") is measured against an O(n²) strawman baseline that certifies nothing about pool efficiency. Risk 3's "prototype early in Phase 4" defers a *design contradiction*, not an implementation unknown.

### M6. The engine's own multi-node story is unstated, and the Ray executor silently disappears
**Where:** N1 line 143, §6.3.5 line 673, §6.0 line 304.

Whether one worker pool may span nodes is a load-bearing decision the doc never makes — Dynamo routes *between* workers; it does not own the NCCL mesh *inside* one. If pools are single-node by fiat, SP degree caps at ~8 GPUs, directly contradicting line 543's jumbo-step mitigation ("shrink jumbo step wall-time with SP"), capping MoT model scale — and `RayDistributedExecutor`, today's shipping multi-node path, is silently dropped: it appears in the §3.1 diagram and then never again in §6, §10, §11, or §12 (violating the plan's own "every phase deletes or freezes what it replaces" discipline). If pools may span nodes, the engine owns cross-node collective bring-up, NCCL-timeout fault domains, and a multi-node health/drain contract — none designed, and C1's recovery problem becomes a multi-node recovery problem. Either answer changes Phase 2/3 scope. "Node-group" appears once, undefined.

### M7. Policies carry per-request mutable state with no state-scoping contract — and the doc contradicts itself on when policies are resolved
**Where:** §6.2.3 lines 412-417 vs §6.2.2 line 387 vs risk 2 line 1621; §6.4 lines 837-840.

The doc says policies are resolved at pipeline build (lines 412-413; risk 2: "resolved to bound methods at build time") *and* in `DenoiseLoop.init` (line 387) — a genuine contradiction on a load-bearing contract. It matters: AdaptiveGateCFG — a named CFGPolicy example and the Wan2.2 worked-example default — is per-request mutable state in shipped code (`denoising.py:338-343, 507-551`: `delta_cached`, `delta_cached_model_id`, gate counters). Build-time-resolved singletons mean request A's cached CFG delta gets applied to request B the moment Phase 2 interleaving lands — silent quality corruption no Phase-2 gate (load tests, latency budget) can catch. This is the *exact* failure mode §6.4 cites to justify interceptor state scoping ("silently corrupts under concurrent requests") — the contract was designed for the plugin tier and forgotten for the policy tier, which sits on a hotter path. Cheap fix (policy state into LoopState, same as plugins), but it must be in the spec.

### M8. The six-policy taxonomy does not factor the shipped step bodies — no step skeleton or cross-policy interaction contract is defined
**Where:** §6.2.3 (policy table, line 424 claim); §6.2.2 lines 386-389; §6.4 lines 837-844.

The proposed step is three phases (forward → CFG combine → scheduler step); the shipped loops need ~six, with dependencies that cross policy boundaries. Verified examples:

- **Cosmos** conditioning-frame injection consumes the *sampler's* EDM coefficients, applies per-CFG-branch both pre-forward (input mix) and post-forward (x0 clamp), and the CFG combine runs in x0 space — ConditioningInjector × Sampler × CFGPolicy interleaved inside each branch, unownable by any one of them (`denoising.py:845-933`).
- **TI2V** clamps latents *after* `scheduler.step` — a post-step constraint with no policy slot (`denoising.py:570-573`).
- **Cosmos2.5** builds per-frame timestep vectors with a conditioned-frame override and re-clamps GT every step pre-forward.
- **CausalDMD** renoises between steps choosing `add_noise` vs `add_noise_high` by expert boundary — Sampler × ExpertRouting (`causal_denoising.py:268-301`).
- **AdaptiveGateCFG** must observe ExpertRouting's switch to invalidate its delta (today an inline `id(current_model)` check) — yet no channel for one policy to observe another is defined anywhere.
- **LTX2** guidance is 1–4 runtime-decided passes whose branches alter the network via forward kwargs (`skip_cross_modal_attn`, `skip_video/audio_self_attn_blocks`) — colliding with BlockInterceptor's domain in a way the "two block-skippers conflict" pre-flight check cannot see, and breaking §6.4's per-CFG-branch state scoping, which assumes a fixed cond/uncond branch vocabulary (`ltx2_denoising.py:503-605, 620-631`).

None of the six policies covers prediction-space conversion, per-token timestep construction, post-step latent constraints, inter-step renoising, or chunk-boundary refresh. The fix is not abandoning policies — the Sampler registry is the natural home for some of this, and composition still strips the duplicated offload/attn-metadata/autocast/trajectory plumbing — but the design needs the fixed step skeleton with ordered, typed extension points and an explicit policy-interaction contract, worked through Cosmos2.5 and LTX2 *in the doc*. Until then, "a new model contributes policies + a graph spec; it does not edit shared loop code" (line 424) is asserted, not demonstrated.

### M9. OmniRequest cannot parameterize multi-loop graphs
**Where:** §6.1 lines 318-334; §6.6 line 905; worked examples (c)(d) lines 943-950.

One flat `SamplingParams` + one flat `DiffusionParams` per request, while the design's own flagship examples are multi-loop graphs needing per-node knobs: LTX-2's refine loop has its own step count and guidance scale *today* as first-class fields (`fastvideo_args.py:204-205`, threaded through `compat.py` and `dynamo/examples/diffusers/worker.py:201-203`); a thinker and talker need different `max_tokens`/`temperature`/`stop`. No request→graph-node parameter binding is defined anywhere; the only escape hatch is line 905's per-model `ModelOptions` blocks — i.e., the `ltx2_*` field-leakage pattern the doc indicts at P3, with a type wrapper, regenerated into the OpenAI/CLI views that derive from the request schema (line 907). Needs a real decision — parameters keyed by graph-node id, or per-node override blocks validated against the PipelineSpec — made in Phase 0, because that schema ships first and external consumers build against it.

---

## Major — caches and weights

### M10. No feature-cache invalidation story under LoRA hot-swap — te-LoRAs make the embedding cache serve stale embeddings in the workflow cloud
**Where:** §6.3.2 lines 570-574 vs §9.4 lines 1349-1380.

The only invalidation rule in the document is RL `update_weights` → `reset()`. But ComfyUI-grade LoRAs routinely patch the *text encoder* alongside the DiT (`comfy/lora.py` maintains `lora_te/lora_te1/lora_te2` key maps; `load_lora_for_models` takes a separate `strength_clip`), so a content-hash-keyed embedding cache returns embeddings computed under the wrong adapter state the moment two workflows share a prompt but differ in te-LoRA stacks — silent wrong output in the exact product (§9.4 "exact mode") whose trust claim is reproducibility. §11.8 even makes cross-request embedding reuse load-bearing as the radix-cache substitute. And once Punica-style batched multi-LoRA lands, requests with different adapter stacks coexist concurrently on one pool, so the cache must be key-*partitioned* by (encoder identity × adapter set × strengths), not flushed — a different design from the `EncoderCacheManager` reset() semantics being adopted, which come from a world where encoders are never patched per request. The key schema needs a weight-state epoch / adapter-set hash as a mandatory component, decided before Phase 3.

### M11. Checkpoint/LoRA patching mutates pool-shared weights — a pool-quiescing barrier the StepScheduler has no vocabulary for
**Where:** §9.4 lines 1371-1380 vs §6.3.1 and §6.0 line 299.

Components are "one resident copy per worker pool"; patch/unpatch mutates that copy, which is global to every loop interleaved on the pool — yet step-interleaving is the engine's core Phase-2 value. Two interleaved loops requiring different patch states cannot coexist, so every cross-group transition is a drain barrier: finish in-flight steps, apply/undo `W += scale·BA` across 14–28 GB shard-consistently across TP/SP ranks (ComfyUI keeps weight backups for the undo — 2× weight memory or a CPU→GPU restore at PCIe seconds), re-admit. Under workflow-cloud traffic (long-tail checkpoints, per-request adapter stacks), transition frequency is the whole game — and the §6.3.1 cost model (lines 516-521) has no weight-state-transition term, no notion of weight state as schedulable state, and no quiesce-vs-queue policy, even though transition cost is exactly what A1 checkpoint-affinity routing must weigh. The §8.6 safe-point-swap pattern shows the doc knows the shape but never applies it here. §9.4 calls this "the one real new subsystem"; §12 carries no risk entry for it.

---

## Major — training/RL

### M12. "Step bodies are plain tensor programs, so autograd composes" is contradicted by the distillation code the substrate must absorb
**Where:** §8.2 lines 1016-1019; §6.2.2; §6.3.2.

Self-forcing does not "drive `DenoiseLoop.step`": its rollout samples per-block exit indices broadcast across ranks, runs no-grad steps to the exit, runs exactly *one* grad-enabled forward, then a separate no-grad `store_kv=True` context-caching pass with context noise, gated by `start_gradient_frame`. None of this fits `init/step/finalize` + `StepResult(done, emit)` without grad-gating flags, per-step cache-write control, and per-block exit policies — training-only surface in substrate code, or the method keeps its own loop and the "3 copies → 1" dedup claim dies for the hardest case. The KV path needs grad/AC-aware semantics the engine pool lacks: today's causal model snapshots KV indices whenever `torch.is_grad_enabled()` so activation-checkpoint recompute doesn't double-advance the cache (`wan_causal.py:119-120,405-431`), and never recycles blocks mid-rollout — while §6.3.2 specs vLLM-style out-of-window block recycling, and §8.5's own profile taxonomy says "training forward … *no caches*," showing the grad+KV case was never designed. §8.3 explicitly stakes the architecture on ChunkKVPool serving self-forcing training.

(Note: the related forward-context-backward attack was refuted — the Phase-1 retirement of the global plus explicit metadata passing *helps* autograd composition. The surviving residue is the grad-window/cache-mode design above.)

### M13. Behavior Record cost is understated ~1.5 orders of magnitude for its own flagship case (MoE diffusion)
**Where:** §8.5 lines 1156-1160; §5 miles row line 1088.

The miles ~60 MB/sample figure is per-token routing, one forward per generated token. Diffusion re-routes the *entire packed sequence at every denoise step, twice under CFG*: the record is steps × CFG × tokens × MoE-layers × top_k. For a Cosmos3-class request (Qwen3-VL-MoE config: 60 experts, top_k 4, ~24 sparse layers via `decoder_sparse_step=1`, ~50K packed tokens, 35-50 steps × 2 branches) that is ~1.3–1.9 GB/sample int32 — ~20–30 GB per 16-sample GRPO group, before latents. "Cheap because trajectory capture is already an OutputSpec feature" conflates plumbing cost with byte cost; at these sizes the Record forces a buffering/transport/storage design (GB-scale trajectories through connectors from disaggregated rollout fleets) that appears nowhere — not in §8.7's TrajectoryBuffer, not in §12, not in the known-gaps list. (The RNG-draws sub-claim was refuted: seeded generators in a single shared loop reproduce draws; uint8 expert IDs also cut 4×. The routing-record problem stands.)

### M14. The omni-RL pilot is a Phase-4 deliverable with no objective design
**Where:** §8.7 lines 1236-1240; §10 Phase 4.

The section establishes *expressibility* (one trajectory, two segment types — true, and a real structural advantage over engine-per-stage stacks) and quietly upgrades it to a deliverable without posing the algorithm problem:

- **Scale mismatch:** token log-probs are O(1–10) nats over 10²–10³ tokens; per-step diffusion SDE log-probs are Gaussian densities over 10⁶–10⁷ latent dims — any joint clipped-ratio objective needs principled per-segment normalization that none of the cited recipes (FlowGRPO/DanceGRPO/NFT/AIPO/GSPO) provides; get it wrong and one modality silently dominates the shared trunk.
- **Credit assignment:** the reasoner influences video reward only through *sampled discrete tokens* re-entering as conditioning — a non-differentiable boundary, so token segments get sparse trajectory-level REINFORCE signal while denoise segments get dense per-step ratios, both updating shared attention-trunk weights, with no interference analysis.
- **Reasoning regression:** RL-updating the und pathway on video-reward-correlated signal risks degrading its reasoning; reference-model KL anchoring for hybrid episodes is never mentioned.

The entire treatment is the phrase "optimized with mixed objectives," and §12's 15 open questions contain nothing on it — for the capability marketed as "the capability nobody else has." Either it gets an algorithm sketch and an open-question entry with an owner, or the Phase-4 item should be demoted from "pilot" to "trajectory capture demonstrated."

---

## Major — the migration plan (the weakest section)

### M15. The "frozen legacy stack" premise is empirically false in this very repo
**Where:** lines 5, 110, 1026; §11.4; risk 5.

The anti-third-stack defense is a declared freeze plus intent to delete — and this repo has already run that experiment and it failed within weeks. Verified from git: `fastvideo/train/` landed 2026-03-09 (#1159); since then **19 commits modified the "frozen" `fastvideo/training/`**, including a *brand-new* `cosmos2_5_training_pipeline.py` added to the legacy stack on 2026-05-11 (#1227) — **nine days after `training/AGENTS.md` explicitly forbade adding new models there**, and eleven days after the same model landed in `train/` (#1224). World-model training (#1179) and LongCat finetuning (#1244) also landed in the frozen stack in May; EMA bugfixes as recently as June 8-9; `AGENTS.md` still calls `training/` "authoritative for shipped models."

The doc invokes the training/-vs-train/ "lesson" but proposes nothing mechanically different from what was tried: no CI gate rejecting new files under legacy paths, no codeowner veto, no named owner per family, no calendar date for Phase 5. "Phase 5 is a scheduled deletion, not an aspiration" (risk 5) — but nothing in the document is scheduled. Under the same model-port pressure that broke the training/ freeze (measurably higher on the inference side), this freeze breaks the same way. Name the enforcement mechanism that did not exist last time, or the deprecation commitment is the prior failure restated with more confidence.

### M16. Phase dependency inversion: Phases 1–2 consume the substrate Phase 4 builds
**Where:** §10 lines 1406-1446 vs §6.3.1 lines 487-489, §6.3.2; three lenses converged on this.

Phase 1 migrates causal Wan ("exercises chunk-KV"); Phase 2 ships "AR continuous batching" — which §6.3.1 *constitutively defines* as "(continuous batching; paged KV; chunked prefill)"; the CacheManager owning both lands in Phase 4, and risk 3 even defers the StepScheduler+KVPool prototype to "early in Phase 4," contradicting Phase 2. Compounding it: **no AR-pathway model exists on the new runtime before the Phase-4 Cosmos3 re-port** (Wan-causal is chunked denoise, not token AR; thinkers/talkers are Phase 4), so Phase 2's headline deliverable has neither a cache backing nor a workload — and none of Phase 2's gates (lines 1427-1430) tests AR batching.

The Phase-1 half is softenable: an interim per-request chunk-KV behind the unchanged `KVHandle` seam, with a Phase-4 allocator swap, is normal incremental staging — but the doc never states this, and its own "no third stack / every phase deletes what it replaces" principle cuts against unstated throwaway implementations. Fix structurally: pull a CacheManager v0 (chunk-KV slabs + minimal paged text-KV) into Phases 1–2, or move AR batching to Phase 4 and rewrite the Phase-2 gate to what it actually exercises.

### M17. Phase 4 re-ports a baseline that is not on main, and the plan schedules neither its merge nor its rebase
**Where:** §10 Phase 0 line 1405, Phase 4 lines 1439-1446; §1 lines 42-49; Appendix.

`fastvideo/pipelines/basic/cosmos3/` on main contains only `__pycache__` — the design's forcing function exists solely as the unmerged 5-branch stacked chain (`feat/cosmos3-tier-a-port` → … → `feat/cosmos3-reasoning`). Phase 0's "Cosmos3 audio leaves `batch.extra`" cannot execute against main: it presupposes the chain is merged (a major-model review effort the plan never schedules) or means maintaining the migration on a side branch, continuously rebased across the most churn-heavy refactors in the repo's history (ForwardBatch→RequestState, loop inversion, executor→engine) — months of conflict-resolution work, unowned and unsized, on the artifact whose 150/150 bit-exactness is the design's proudest credential and whose parity suite the Phase-4 gate requires ("every phase ships green" cannot apply to a suite that is not in the tree). The plan sequences other in-flight work explicitly (`fastvideo/api/` in Phase 0, PR #1438 in Phase 1) but skips this. Needs an explicit merge milestone before Phase 0 touches the port.

### M18. G5's enforcement instrument has holes: ~6-7 shipped families have no SSIM test, and the CI-cost mitigation is incoherent for substrate PRs
**Where:** G5 lines 128-129; Phase 0 gate line 1405; risk 6.

`fastvideo/tests/ssim/` covers ~14 of 20+ families. Cosmos(2/2.5), Hunyuan, Hunyuan15(+SR), HYWorld, MagiHuman, Waypoint, and MatrixGame-v1 have no SSIM test — "all SSIM suites unchanged" passes *vacuously* for roughly a third of shipped pipelines, exactly the ones sitting on the shared loop being refactored. And risk 6's "gated to touched families" mitigation is designed for model-local PRs; Phases 0–2 are by construction not model-local — the ForwardBatch adapter, loop inversion, and executor replacement sit under every family, so "touched families" = all of them on precisely the riskiest PRs. Either substrate PRs run the full GPU matrix (a cost the plan should budget — SSIM runs on Modal L40S today) or gating quietly degrades to sampling, which is how regressions slip through. Needs: a reference-seeding work item before Phase 1, or G5 restated as "zero regression for the SSIM-covered subset," plus a stated per-phase GPU-CI budget.

### M19. Phase 5's deletion milestone breaks the "frozen and untouched" legacy training/ stack
**Where:** lines 5-6, 144, 1026-1027 vs Phase 5 line 1448.

The frozen stack is a live consumer of exactly the code Phase 5 deletes: `fastvideo/training/training_pipeline.py:39` imports `ComposedPipelineBase`/`ForwardBatch`/`LoRAPipeline`, holds `validation_pipeline: ComposedPipelineBase`, and its validation instantiates real legacy pipelines that run the legacy `DenoisingStage`; `distillation_pipeline.py:31` likewise. So Phase 5 cannot remove `ComposedPipelineBase` and `DenoisingStage` while leaving `training/` untouched — either the deletion milestone hollows to "delete except what legacy training/ needs" (the old path never dies — the very smell being fixed) or the scope statement is false and `training/` breaks on this plan's schedule. Relatedly, "loop inversion makes the step functions the single shared implementation" is arithmetically 3→2, not 3→1: the legacy inlined copies are out of scope forever. The doc needs an explicit answer: what happens to `fastvideo/training/` at Phase 5?

### M20. "Retire `fastvideo/forward_context.py` (Phase 1)" is infeasible as scheduled
**Where:** §6.3.3 lines 618-621; Phase 1 lines 1412-1414; vs N2/N4; Appendix line 1791.

194 references across ~50 files. The global is read inside `fastvideo/attention/layer.py` — the shared Attention module on *every* family's hot path — and set in 27 places inside the frozen `training/` stack (8 module-level imports). Phase 1 migrates only Wan+Flux2; the other ~16 families run "unmodified" behind the legacy adapter (N4) and still set the global. So in Phase 1 the file cannot be deleted (touches the frozen stack, violating N2; breaks every unmigrated family), and `attention/layer.py` must serve both worlds simultaneously — a dual-sourcing branch in the hottest shared layer, undesigned. The honest description: Phase 1 *adds a second context mechanism beside the global*, and the global survives until Phase 5 at the earliest — where the deliverables list never mentions it. Appendix A states "retired Phase 1" as accomplished fact. Rewrite as "new-path-only StageContext; `forward_context` frozen for legacy consumers; deletion gated on Phase 5," and design the dual-mechanism cost.

### M21. §10 is a dependency ordering, not a plan — no timeline, no staffing, no sizing, and no policy for the ~1-2 new model ports per month that arrive during the migration
**Where:** §10; N4 line 153; risk 1.

The scope — typed I/O, loop inversion + policies, extension system, async engine + StepScheduler + online-calibrated cost model, four-class CacheManager, PackedSeq/MoT layers, declarative parallelism compiler, workflow compiler, RL layer, Dynamo contract, config collapse — is plainly multi-engineer-years, with zero dates, headcount, per-phase sizing, or owners; "by Phase 2" decision deadlines (§11.1, risks 7/15) are unanchored because Phase 2 is not a date.

The sharper, unanswered problem is **inflow**: git shows ~1–2 new families landing per month (Flux2 Klein and Lucy Edit on 2026-06-09 alone; MatrixGame3 05-27; MagiHuman 05-12; Stable Audio 05-01; Gen3C 04-01…). Over multi-quarter Phases 0–4, another 10–15 models arrive, and the doc never says what they target: land them on legacy abstractions and the Phase-5 tail grows faster than phases retire it (negative net migration velocity); force them onto the new stack and every port blocks on machinery that doesn't exist until Phase 1/3/4. Either answer materially changes the plan; choosing neither means the terminal state recedes indefinitely. Minimum fix: per-phase engineer-month estimates, a named owner per phase, a calendar target for Phase 5, and an explicit "new ports target the new stack starting at Phase X" rule with its porting-velocity cost stated.

---

## Major — product/trust surfaces

### M22. Per-request plugin enablement is an unsandboxed third-party-code and noisy-neighbor surface; only workflow JSON is named untrusted
**Where:** §6.4 lines 859-861 vs §12 input-hardening gap lines 1693-1695.

Entry-point plugins execute arbitrary code inside the serving engine, and the doc makes their selection part of the *request* (`diffusion.plugins=[{"name": "cache_dit", "Fn": 8, "Bn": 8}]`) in the same engine pitched as a multi-tenant cloud — and since the OpenAI protocol is *generated from the request schema* (lines 907-908), the field derives into the public API with no carve-out. Consequences forcing a design change: (a) **correctness** — a caller can attach a distribution-altering interceptor to a request the product has labeled "exact mode" (the §9.4 trust claim), or pass unvalidated kwargs into third-party code; (b) **isolation** — a `needs_eager` observer on one request drops compile/cudagraph capture for scopes shared with co-scheduled tenants (line 809), a noisy-neighbor vector with no cost attribution anywhere in the metrics design; (c) **supply chain** — entry-point resolution imports whatever package claims the name. The needed contract: enablement/allowlisting at DeployConfig scope only; requests merely parameterize pre-enabled plugins against per-plugin validated schemas; plugin overhead attributed per-request in the cost model. §12's input-hardening gap names only workflow JSON — a categorically different surface.

### M23. No versioning or stability contract for the serialized schemas shipped to external consumers mid-migration
**Where:** §6.4 line 861; §6.6 lines 920-927; §10; open question 12.

By Phase 3 there are at least four externally consumed serialized surfaces: hub-published ModelSpec manifests (interchange with diffusers' `modular_model_index.json` — a format co-owned with an external party), compiled-workflow PipelineSpecs (content-hash-keyed in the weight-fleet cache — schema changes silently change hashes and invalidate fleet affinity), the OmniEvent streaming schema (Dreamverse's frontend; proposed as Dynamo ask A3's wire format), and per-model ModelOptions blocks. Phase 4 then lands PackedSeq, session-scoped inputs, and the Cosmos3 re-port — guaranteed churn after consumers exist. The migration plan gates *behavior* at every phase (SSIM, parity, load) and gates *interfaces* at none; the only versioning commitment in the document is hook-point names (open question 12 is scoped to hook points). Without per-surface decisions now — `schema_version` fields, frozen-vs-experimental tiers per phase, a deprecation window — Phase 4 either breaks published artifacts or gets paralyzed by accidental freezing. G5 protects only the Python `VideoGenerator` call.

---

## Minor (confirmed)

1. **ForwardBatch has 111 fields, not ~250** (AST-verified; stated twice, lines 33/188). P3 survives at 111, but the headline metric is inflated 2.3× in a doc that brands its pain points "evidence-backed" — it invites discounting of the numbers that *do* verify exactly (1381 lines and 35 probes both check out).
2. **"Prediction is a table lookup" vs the design's own flagship features** (§6.3.1 vs §6.4): DBCache/FBCache/TaylorSeer decide per step from runtime residual similarity — a stochastic per-step cost multiplier unknowable at admission; VSA tile selection is content-dependent; and AR decode lengths are unbounded (the doc concedes vLLM "must guess decode lengths," then silently exempts its own AR group).
3. **Worked example (g) is internally contradictory**: cache-dit + C1 + "identical trajectories" are pairwise incompatible under §8.5's own `distribution_altering` contract (§8.7 states the rule correctly: cache acceleration is C0). Matters because (g) is the template PR #1438 is told to target in Phase 1.
4. **The Phase-2 Dreamverse gate is untestable as written**: at ~4.55 s GPU-saturating per 5 s clip (line 1263), "≥2 concurrent sessions per GPU at unchanged segment latency" is only passable under an unstated think-time/collision-rate assumption — the gate can be passed or failed at will by choosing the test's session behavior. More broadly, no quantitative multiplexing target (sessions/GPU under a stated load profile, GPU-utilization, cost/clip) exists anywhere, so there is no way to conclude after Phase 2 whether step-level scheduling earned its complexity over the §11.6-rejected simpler design.
5. **The exec summary launders Dynamo contingencies into outcomes** (line 75: "each with a fallback — so Dynamo fronts both production serving and RL rollout fleets"): the body is honest (A1–A7 with fallbacks; §11.9; §12.15), but the asks are unfiled RFCs on an NVIDIA-governed roadmap; A5's own fallback "weakens fleet-scale async RL," and if A3 misses Phase 2, Dreamverse ships on the direct-WebSocket bypass and the production-hardened fallback becomes permanent — the exact "permanent workaround" dynamic §11.9 claims the direct relationship avoids. Ask-sequencing (§12.15) has no owner or decision dates.
6. **diffusers as "convergent validation" cuts both ways** (see M2): its four-wrappers-per-family shape is the subclass forest again; the citation supports the rejected alternative as well as the chosen one.
7. **Punica/ComfyUI LoRA semantics gap** — see M4.

---

## Fact-check corrections

70 concrete claims were checked; **none was fabricated**; 13 need correction. Everything else verified, including the claims most likely to be embellished: vLLM RFC #42770 (author/date/content/two-tier resolution), PR #42304 **merged** 2026-05-16 with `VLLM_USE_BREAKABLE_CUDAGRAPH`, vllm-omni RFC #4084, the Thinking Machines numbers (80/1000 unique outputs, divergence at token 103, 26s→42s, KL results), the Dynamo worker's `asyncio.Lock`, cache-dit, the cosmos-framework MoT details (PackedAttentionMoT, MoTDecoderLayer, ReasonerKVCache, MoE gen-MLP), miles/verl-omni/sglang-omni mechanics, sglang's cache-dit monkeypatch scars, and `enable_teacache` genuinely having no consumer.

| # | design.md says | Reality |
|---|---|---|
| 1 | "1381-line `DenoisingStage`" (lines 34, 201) | 1381 is the **file**; the class is ~670 lines (47–715) plus 6 subclasses in-file. The 35-probe count is exact for the file. |
| 2 | "~250-field ForwardBatch" (33, 188) | **111 fields** (whole file incl. TrainingBatch/PreprocessBatch: ~153). |
| 3 | "19 denoising-stage classes" (201) | **22** model/variant classes (+ base = 23); the list omits Magi-class and two other same-category stages predating the doc. |
| 4 | "Cosmos2.5 clamping … hardcoded in the shared loop" (201) | Clamping lives in the `Cosmos25DenoisingStage` **subclass**; the Wan2.2 expert switch (`denoising.py:229-235, 352-376`) and TI2V inline VAE encode (`:239-268, 399-404, 570-572`) are in the shared loop as claimed. |
| 5 | `SamplingParam` "~170 fields" (887) | **75**. The ~170 figure belongs to TrainingArgs (90 own + 81 inherited = 171). |
| 6 | `FastVideoArgs` "~96 fields" (885) | **81** (TrainingArgs subclassing claim correct). |
| 7 | "TP and SP (Ulysses/ring)" (192) | Main is **Ulysses-only** (`all_to_all_4D`); no ring-attention SP is wired into FastVideo. |
| 8 | CFG "3 copies: `stages/conditioning.py` vs …" (993) | Right count, wrong citation: the inference-stack copy is in `denoising.py`, not `conditioning.py`. |
| 9 | ComfyUI "~45 `comfy_extras` packs", "90+ blueprints" (1335-1339) | **117** packs (matching nodes.py's 117-entry registration list); **80** in-tree blueprints (the larger library ships via the registry). 64 core nodes, 39 API providers, GPL-3.0, FIFO-no-batching all verify. |
| 10 | kv-router events "`{sequence_hash, block_hash, removed}`" (707, A1 733-739) | Paraphrase: actual shape is `KvCacheEventData::Stored{parent_hash, blocks[{block_hash, tokens_hash}]}` / `Removed` / `Cleared` (`protocols.rs:627-646`). Token-prefix-derived keying verifies. |
| 11 | miles TIS clamp "to `[0.5, 2.0]`" (1086) | Configurable `[tis_clip_low, tis_clip]`, CLI defaults [0, 2.0]; the 0.5/2.0 pair comes from the MIS example config (`mis.yaml`). |
| 12 | sglang-omni "`DllmScheduler` for a DiT talker" (269) | DllmScheduler serves the **LLaDA2-Uni thinker** (diffusion-LLM); the DiT talker is Ming-Omni's, on a different scheduler. |
| 13 | `_iter_packed_batches` under `model/vfm/` (236); §11.3's claim that the port's "own status notes" list reasoning-KV/batching/streaming/prefix-reuse as "missing for production" | Lives at `cosmos_framework/inference/inference.py:66`. PORT_STATUS.md confirms 150/150 but contains no such missing-for-production list — that framing is the design doc's own and should not be attributed to the port's status notes. |

---

## Attacks that failed (the doc survives these)

The refute-by-default verifiers killed 36 findings, several of them attacks a hostile reviewer would lead with — worth knowing they don't land:

- **ChunkRollout/DenoiseLoop nesting is expressible** in the stated Stage/LoopStage/StepResult contracts ("one solver step / one token / one chunk" + composition).
- **N1 vs engine-internal pools** is consistent on a careful read (N1 is about datacenter orchestration; §6.3.5 states the reconciliation).
- **The trainer-scope line (N2 vs §8)** is drawn consistently — N2's own text enumerates exactly what §8 changes.
- **G6 vs the ≤2% Phase-2 gate** is goal-vs-acceptance-gate, not contradiction (Phase 1 is gated bit-identical).
- **The clean-room GPL posture holds**: sampler/scheduler math (DPM-Solver, Karras sigmas, flow-match shift) is published outside GPL sources.
- **C2 for the video denoise path is fine**: batch-1 fixed shapes are trivially batch-invariant — the doc's own analysis at lines 1145-1147 is correct; the AR/image/sharding exposures are correctly identified there too.
- **Self-forcing's cross-chunk gradients truncate by construction** (KV written under `no_grad` on detached context), so the engine KV pool is not blocked the way one might fear — the surviving residue is M12's grad-window cache mode.
- **"Every phase deletes or freezes something" survives audit** at the phase-deliverable level (the failures are the specific items in M19/M20).
- **The tier-1 ComfyUI vocabulary claim survives** blueprint-corpus measurement under the doc's actual claim (curated canonical workflows, not top-N node frequency).
- **The sglang reconvergence deferral** is substantively defended in §11.1 with reasons valid under either outcome.
- **WeightSyncPlan's "literal no-op"** is correctly scoped to colocated same-layout in the doc's own sentence; FSDP-vs-TP/SP is explicitly routed to in-place reshard.

---

## Ranked recommendations

1. **Design the abort/cancellation/OOM path with Phase 2** (C1) **and add memory as a budget axis with admission planning and preemption semantics** (M3). These two are the soundness conditions of the multiplexing bet; everything else in the execution plane sits on them.
2. **Re-derive §6.3.2 from the real vLLM constraint** (M1). The two-pool→one-pool reversal was made on a false premise; either accept uniform page bytes (and redesign the slab story) or bring back two pools with an explicit fragmentation/deadlock argument.
3. **Fix the migration plan's three structural defects**: CacheManager v0 into Phases 1–2 or AR batching out of Phase 2 (M16); a merge milestone for the cosmos3 chain before Phase 0 touches it (M17); a new-port inflow rule plus a freeze-enforcement mechanism that did not exist last time — CI path gate, codeowners, a date (M15, M21). Also reconcile Phase 5 with the frozen `training/` stack (M19) and restate the `forward_context` retirement honestly (M20).
4. **Specify the step skeleton and the policy contracts** — ordered, typed extension points; a policy state-scoping rule (state in LoopState, like plugins); a policy-observation channel — and work the mapping through Cosmos2.5 and LTX2 in the doc (M7, M8). Decide per-node request parameter binding in Phase 0 (M9).
5. **Give MoT a stated parallelism answer** (M5) and make the single-pool-spans-nodes decision explicit, including the fate of `RayDistributedExecutor` (M6).
6. **Close the workflow-cloud trust/correctness holes before Phase 3**: adapter-aware feature-cache keys (M10), weight-state transitions as a scheduled, costed operation (M11), DeployConfig-scoped plugin allowlisting (M22), per-surface schema stability tiers (M23), and an honest assessment of Punica's fit (M4).
7. **Right-size the RL claims**: design the grad+KV cache mode or scope self-forcing out of the shared loop (M12); budget the Behavior Record at real byte counts (M13); demote the omni-RL pilot or give it an objective sketch and an owner (M14); fix worked example (g).
8. **Reclassify loop inversion as unprecedented at scheduler granularity** in risk 3 and drop the diffusers "validation" (M2). The bet may still be right — but it should be made with open eyes, and the parity-gate plan is then carrying more weight than the doc admits.
9. **Correct the thirteen numbers above before circulating.** The doc's credibility rests on its "evidence-backed" brand; ~250-vs-111 is the kind of error that makes a reader re-check everything else — and most of everything else checks out.
