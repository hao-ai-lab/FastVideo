# mini-fastvideo

A scoped, **CPU-testable** realization of the `design_v3.md` model-native runtime — the architecture
where *omni is native, train and serve are the same loops by construction, and correctness is a typed
contract you can sign*.

It is "mini" in two senses: (1) it supports a deliberate subset of models/methods, and (2) the heavy
neural forwards are small numpy stand-ins so the whole thing runs and is tested on a laptop with no
GPU and no torch. **The architecture is real; the kernels are toys.** On a GPU box you swap
`ComponentSpec.factory` for the torch adapters (which wrap the real `fastvideo.models.*` modules) and
nothing else in the runtime, scheduler, caches, parity, or training code changes — that is the whole
point of the (recipe, runtime) separation.

## Scope (phase 1)

| | Supported |
|---|---|
| **Inference** | Wan2.1-1.3B (T2V, bidirectional diffusion) · LTX2.3 (two-stage distilled: 8-step base → upsample → 3-step refine) · Wan-causal (chunked/streaming rollout, the self-forcing student) |
| **Training** (all on Wan2.1-1.3B) | finetuning (flow-match) · DMD2 distillation · DiffusionNFT (likelihood-free RL) · self-forcing (causal, on Wan-causal) |
| **Omni / MoT** (phase 2) | Cosmos3 (reasoner `ar_decode` → joint `diffusion_denoise`) · BAGEL/lance (`generate_text` → `generate_image`) — each ONE resident MoT instance running both loop types on shared weights |

## The one invariant (design_v3 §1)

```
Model cards own components, loops, recipes, and parity.
Programs compose loops into tasks.
The scheduler executes the steps of loops as WorkUnits under one budget.
Caches are correct by key, not by hope.
Training records behavior on the same loops it serves.
```

## Package layout → design_v3 §18

| Package | design_v3 | What it is | Validated by |
|---|---|---|---|
| `card/` | §4 | `ModelCard` = (components, loops, **recipe**, **parity**) as one versioned, validatable object; `ModelInstance` (one resident card; many loops bind shared components by reference) | `tests/test_contracts.py` |
| `loop/` | §5 | the driven-loop contract `init/next/advance/finalize`; typed `WorkPlan`/`StepResult`/`LoopState`; the single `LoopRunner` driver; policies (CFG/flow-shift/precision/expert/conditioning); the flow-match sampler | `test_policies.py`, `test_inference.py` |
| `runtime/` | §6 | `Engine` (serial + step-interleaved execution), layered scheduler (`AdmissionController`, `BatchScheduler`), the `RuntimeLoopContext` (the inversion point), cost currency | `test_admission.py`, `test_interleave_gate.py` |
| `cache/` | §7 | typed `CacheKey` (partitioned by adapter+weights), per-class pools (`feature`/`residual`/`slab_kv`/`paged_kv`) behind one manager | `test_cache.py` |
| `memory/` | §7.3 | tagged pools, reservation-before-admission, component-granular sleep/wake | `test_admission.py` |
| `parallel/` | §8 | `ParallelPlan` (named axes), fake `DeviceMesh` builder, pre-flight validation (cfgp≤2, **pp_patch invalid for causal**, ownership conflicts) | `test_contracts.py` |
| `parity/` | §9 | `ParityAligner` (record/compare taps), the C0–C4 consistency ladder, and the **non-negotiable batch-of-N interleave gate** | `test_parity.py`, `test_interleave_gate.py` |
| `extend/` | §11 | observers (Profiler, NaNWatch) + interceptors (cache-dit-style `ResidualSkipInterceptor`) with state per request & per CFG branch | `test_extensions.py` |
| `program/` | §13 | `Program` of typed nodes (`ModelLoopNode`/`ComponentNode`) + edges; `when=` branches replace geometry heuristics | `test_inference.py` |
| `request/` | §12 | typed `Request`/`Session`/`Artifact`/`Stream`/`CancelScope`; `TaskType` declared, never inferred | used everywhere |
| `training/` | §10 | rollout = serve loop + capture; `BehaviorRecord`; `WeightSyncPlan` (declared role); rewards; the 4 methods | `test_training.py` |
| `models/` | §4 | the concrete (recipe, runtime) cards: `wan21/`, `ltx2/`, `wan_causal/` + the numpy toy backend | all |

**Enforced boundary** (design_v3 §10, §18): `training/` imports `card`/`loop`/`runtime`/`models`; the
engine never imports `training`. `card/` imports no runtime.

## What is actually demonstrated (each backed by a test)

- **Loop inversion is safe under interleaving.** Two+ concurrent requests, interleaved at step
  granularity, are bit-identical to running them serially — for all three models
  (`test_interleave_gate.py`). A deliberately buggy *module-global* interceptor **breaks** this gate,
  while the correct per-request one passes (`test_extensions.py`) — the structural proof of §5.1/§11.
- **The (recipe, runtime) pair is enforced.** A card whose recipe `assumes_loop` a loop it doesn't
  declare fails `validate()` (`test_contracts.py`).
- **One resident instance runs many loops on shared weights.** LTX2's base + refine loops bind the
  same `transformer` by reference (`models/ltx2/card.py`).
- **Caches are correct by key.** Same prompt + different te-LoRA stack ⇒ different key (partitioned,
  not stale); an `update_weights` bump invalidates wholesale (`test_cache.py`).
- **Admission is reservation-first.** Two requests that fit alone but jointly OOM ⇒ one is deferred,
  and deferral does **not** change outputs (`test_admission.py`).
- **Train ≡ serve by construction.** Every training method drives the *same* loop the engine serves,
  in the ROLLOUT profile; DiffusionNFT is likelihood-free (C2), samples from the decay-blended **old**
  policy (not the student), uses group-relative advantages, and encodes the shared prompt **once** per
  K-sample group via the feature cache (`test_training.py`).
- **`pp_patch` parallelism is rejected for causal models** (stale KV breaks causality, `test_contracts.py`).

## Run it

```bash
cd /Users/willlin/src/FastVideo-mini
python3 -m pytest mini_fastvideo/tests/ -q      # 49 tests
python3 mini_fastvideo/run_tests.py             # same suite, ZERO deps (no pytest needed)
python3 -m mini_fastvideo.examples              # the design_v3 §15 worked examples
```

Requires only Python 3.10+ and numpy.

## Running the real models (GPU)

Each `ComponentSpec` records the real `load_id` (e.g.
`fastvideo.models.dits.wanvideo:WanTransformer3DModel`). To run real Wan/LTX:
1. install torch + the parent `fastvideo` package + weights;
2. replace each `ComponentSpec.factory` with a lazy adapter that loads the real module and exposes the
   same call surface the loops use (a velocity/`predict_noise` forward; `vae.decode`; `text_encoder.encode`);
3. swap the numpy flow-match sampler for the real `FlowUniPCMultistepScheduler` (Wan) / distilled
   scheduler (LTX) inside the loop's step body.
The loops, policies, scheduler, caches, parity gates, and training methods are unchanged.

## Omni / MoT (phase 2 — implemented)

The omni-ready spine paid off: adding omni was **additive, not a refactor**. Two omni cards now ship,
each as ONE resident MoT instance whose `transformer` binds *both* an `ar_decode` loop and a
`diffusion_denoise` loop (`shared_weight_components=["transformer"]` on both) — the §16 claim no
DAG-of-engines can express, made native:

- **`models/cosmos3/`** — `tokenize → reason (ar_decode) → pack → diffusion_denoise → vae_decode`.
  The reasoner upsamples the prompt on the und pathway; its tokens condition the joint denoise on the
  same weights. `sound_vae` is declared `optional_for` non-t2vs (the lazy-component P8 fix).
- **`models/bagel/`** — the canonical vllm-omni model: `generate_text (ar_decode) → generate_image
  (diffusion_denoise)`. Unlike vllm-omni's opaque `DIFFUSION` stage, **both loops are runtime-visible**:
  the scheduler prices `ar_token` *and* `diffusion_step` WorkUnits (verified in `test_omni.py`).

The diffusion loop is literally `WanDenoiseLoop` (the same step body the engine serves for Wan), bound
to the MoT module — one loop definition, reused. The interleave gate generalizes across loop types
(`test_omni.py::test_omni_interleave_parity_holds_across_loop_types`). Run it:
`python3 -c "from mini_fastvideo.models import build_omni_engine; ..."` or see `mini_fastvideo/examples.py`.

Still future work (declared on the spine, not yet built): the packed factored-sequence
(`[text|vision|action|sound]`) `LoopState.extension`, joint multi-modality denoise with per-modality
CFG, and world-model `chunk_rollout` for action-conditioned omni.

## Deliberately out of scope (per design_v3 non-goals)

Fleet/Dynamo orchestration, the ComfyUI workflow compiler, WebRTC realtime transport, full LLM-grade
AR serving (radix trees, speculative decoding), and real distributed parallelism (the mesh builder is
a CPU fake). These are named in design_v3 as the fleet's job or as later phases.
