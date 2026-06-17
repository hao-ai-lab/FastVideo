# v2

A scoped, **CPU-testable** realization of the model-native, *(recipe, runtime)* runtime. The full
write-up is **[`../designv4.md`](../designv4.md)** — the unified design, reflecting exactly what is
built and tested here (with **[`../design_v3.md`](../design_v3.md)** as the north-star argument it
elaborates).

It is "mini" in two senses: (1) it supports a deliberate subset of models/methods, and (2) the heavy
neural forwards are small numpy stand-ins so the whole thing runs and is tested on a laptop with no
GPU and no torch. **The architecture is real; the kernels are toys.** Backends are selected through
the `platform/` dispatch substrate: two tuple-keyed registries — `COMPONENTS(kind, device, variant)`
and `KERNELS(op, device, arch, variant)` — that a detected `Platform` resolves, with the numpy
reference as the terminal fallback rung and parity oracle (`§17`). On a GPU box you register the
torch component/kernel backends (the `cuda` cells are declared in `platform/backends/torch_cuda.py`,
`available=False` until torch+CUDA are present) and `Platform.detect()` resolves them — the loops,
scheduler, caches, parity, training, and workflow code are unchanged.

Honest scope of what is *wired* today (the substrate supports more than it demonstrates): a pure-
python `accel` stand-in backend proves the dispatch is genuinely device-generic (cross-device
resolution + arch fallback + the parity oracle, bit-identical to numpy) without a GPU. All diffusion
loops (wan21, ltx2, wan-causal, adapters, adaptive) and the FlowGRPO RL log-prob recompute route
their solver ops (`flow_match_step`/`flow_sde_step`) through the kernel table — the recompute is
pinned to the rollout's kernel for C2 correctness. The `dit` and `vae` components have `accel`
overrides (`text_encoder` demonstrates the device→cpu fallback). Not yet routed: the AR/vocoder op
families (omni/qwen-omni) have no KERNELS entries, the full torch/CUDA path is declared-unavailable,
and a cudagraph capture-safety (`workspace_bytes`) contract is deferred.

**184 tests, 30 files**, run two ways (`pytest` and a zero-dependency runner). Python 3.10+ and numpy only.

## Scope

| | Supported |
|---|---|
| **Inference** | Wan2.1-1.3B (T2V, bidirectional diffusion) · LTX2.3 (two-stage distilled: 8-step base → upsample → 3-step refine) · Wan-causal (chunked/streaming rollout, the self-forcing student) |
| **Training** | finetuning (flow-match) · DMD2 distillation · DiffusionNFT (likelihood-**free** RL) · self-forcing (causal) — all on Wan2.1-1.3B |
| **Joint / multi-expert RL** | UniRL/PromptRL joint LM+generator RL (`unified`) · **N-way** joint RL over arbitrary experts (`multi_expert`) · **end-to-end RL over a cross-model workflow** (`workflow_rl`) |
| **Omni / MoT** | Cosmos3 & BAGEL — one resident MoT instance, `ar_decode` + `diffusion_denoise` on **shared** weights · Qwen-Omni **thinker→talker→vocoder** — three **separate** experts, three loop types, cascaded + streaming |
| **Composition & scheduling** | cross-model **Workflow** (T2I→I2V: `flux-t2i` → `wan-i2v`) · tiled VAE decode (`VAE_TILE` units co-scheduled with denoise steps) |
| **Interactive** | `WorldModelSession` — long-lived world-model rollout with persistent cross-request state, frame streaming, step-boundary cancellation |

## The one invariant (designv4 §1)

```
Model cards own components, loops, recipes, and parity.
Programs compose loops into tasks.   Workflows compose models into pipelines.
The scheduler executes the steps of loops as WorkUnits under one budget.
Caches are correct by key, not by hope.
Training records behavior on the same loops it serves.
```

## Package layout → designv4 §12

| Package | What it is | Validated by |
|---|---|---|
| `card/` | `ModelCard` = (components, loops, **recipe**, **parity**) as one versioned, validatable object; `ModelInstance` (one resident card; many loops bind shared components by reference) | `test_contracts.py` |
| `loop/` | the driven-loop contract `init/next/advance/finalize`; typed `WorkPlan`/`StepResult`/`LoopState`; the single `LoopRunner`; policies (CFG/flow-shift/precision/expert); the flow-match + **FlowGRPO SDE** samplers | `test_policies.py`, `test_inference.py` |
| `program/` | `Program` of typed nodes (`ModelLoopNode`/`ComponentNode`) for one model; **`Workflow` + `WorkflowRegistry`** for cross-model pipelines | `test_inference.py`, `test_t2i_i2v_workflow.py` |
| `runtime/` | `Engine` (serial + step-interleaved, **workflow-aware**), `AsyncEngine`, `DisaggregatedRunner` + role pools, layered scheduler, `RuntimeLoopContext`, **`WorldModelSession`** | `test_admission.py`, `test_interleave_gate.py`, `test_serving.py`, `test_world_session.py` |
| `cache/` | typed `CacheKey` (partitioned per-component by adapter+weights), per-class pools (`feature`/`residual`/`slab_kv`/`paged_kv`) | `test_cache.py` |
| `memory/` · `transport/` · `parallel/` | tagged pools + reservation-before-admission; connectors (`chunk_ready` + credit flow); `ParallelPlan` + pre-flight validation | `test_admission.py`, `test_serving.py`, `test_contracts.py` |
| `parity/` | `ParityAligner`, the C0–C4 consistency ladder, the **batch-of-N interleave gate** | `test_parity.py`, `test_interleave_gate.py` |
| `extend/` | observers (Profiler, NaNWatch) + interceptors (cache-dit `ResidualSkipInterceptor`), state per request & per CFG branch | `test_extensions.py` |
| `training/` | rollout = serve loop + capture; `BehaviorRecord`; `WeightSyncPlan` (per-component role); rewards; **7 methods** (4 base + `unified_rl`, `joint_multi_rl`, `workflow_rl`) | `test_training.py`, `test_unified_rl.py`, `test_joint_multi_rl.py`, `test_workflow_rl.py` |
| `serving/` · `deploy/` | our own OpenAI server (stdlib asyncio); `DeploymentCard`; **our own `LocalFleet`**; `DynamoWorkerAdapter` (frontable, not relied upon) | `test_serving.py`, `test_serving_fixes.py` |
| `models/` | the concrete cards: `wan21/ ltx2/ wan_causal/` · `omni/ cosmos3/ bagel/ qwen_omni/` · `unified/ multi_expert/` · `image_video/ tiled/` + the numpy toy backend | all |

**Enforced boundary** (designv4 §3, §8): `training/` imports `card`/`loop`/`runtime`/`models`; the
engine never imports `training`. `card/` imports no runtime. Cross-model `Workflow` orchestration sits
*above* the engine (no change to the single-instance hot path).

## What is demonstrated (each backed by a test)

Core contracts:
- **Loop inversion is safe under interleaving** — concurrent requests interleaved at step granularity
  are bit-identical to serial, across every model and loop *type* (`test_interleave_gate.py`,
  `test_omni.py`). A buggy module-global interceptor **breaks** the gate; the per-request one passes.
- **The (recipe, runtime) pair is enforced** — a card whose recipe `assumes_loop` an undeclared loop
  fails `validate()` (`test_contracts.py`).
- **Caches are correct by key**, partitioned per component, so a transformer weight-sync does not flush
  a frozen text-encoder's feature cache (`test_cache.py`).
- **Train ≡ serve by construction** — every method drives the *same* loop the engine serves, in the
  ROLLOUT profile (`test_training.py`).

Stress tests (the design held — see **designv4 §9**):
- **§9.3 Joint LM+generator RL** (UniRL/PromptRL) — one reward → a token policy-gradient on the LM *and*
  a FlowGRPO PPO on the DiT; **likelihood-based C2** (per-step log-prob identity ⇒ PPO ratio == 1).
- **§9.5 Qwen-Omni cascade** — three separate experts, three loop types, cross-stage conditioning,
  streaming codec→waveform.
- **§9.6 Cross-model Workflow** — T2I→I2V across two *distinct* models; the I2V stage provably consumes
  the generated image; each model keeps its own interleave-parity guarantee.
- **§9.7 N-way joint RL** — generalizes joint RL to N experts (per-component sync, dict grad-targets);
  the substrate was already N-ready, only the method body looped over two.
- **§9.8 Interactive world-model session** — persistent cross-request state, transactional
  step-boundary cancellation, no cross-session smearing (`test_world_session.py`).
- **§9.9 End-to-end RL over a workflow** — one *final-video* reward trains an *earlier* model (the T2I
  generator); proven causal by a control (constant reward ⇒ nothing moves) (`test_workflow_rl.py`).
- **§9.10 Heterogeneous WorkUnit co-scheduling** — `VAE_TILE` units interleave bit-identically with
  `DIFFUSION_STEP` units through one budget; tiling is exact (`test_tiled_scheduling.py`).

## Run it

```bash
cd /Users/willlin/src/FastVideo-mini
python3 -m pytest v2/tests/ -q      # 184 tests
python3 v2/run_tests.py             # same suite, ZERO deps (no pytest needed)
python3 -m v2.examples              # the worked examples
```

## Running the real models (GPU)

Each `ComponentSpec` records the real `load_id` (e.g.
`fastvideo.models.dits.wanvideo:WanTransformer3DModel`). To run real Wan/LTX/etc.:
1. install torch + the parent `fastvideo` package + weights;
2. implement the `cuda` (or other-device) cells in `platform/backends/` — a component builder that
   loads the real module and exposes the same call surface the loops use (a velocity forward;
   `vae.decode`; `text_encoder.encode`), and the device kernels for the solver/primitive ops. They
   flip to `available=True` when torch+CUDA are present, and `Platform.detect()` resolves them; the
   numpy `ComponentSpec.factory` stays as the CPU terminal fallback rung;
3. the diffusion loops + RL recompute already dispatch their solver ops via
   `model.platform.kernels.get(...)`, so the registered `cuda` kernels are picked up automatically;
   only the not-yet-routed op families (AR sampling, vocoder synth) need the same one-line swap.
The loops, policies, scheduler, caches, parity gates, training methods, and workflows are unchanged.

## Serving & fleet (our own version — Dynamo is an option, not a dependency)

`AsyncEngine` (queue, lifecycle, live streaming, step-boundary cancellation) · role/stage pools +
disaggregation (output **bit-identical to inline**) · `transport/` connectors (`chunk_ready` + credit
flow) · our own `LocalFleet` (cost/affinity/least-loaded routing, health/drain) · a `DynamoWorkerAdapter`
exporting the same `DeploymentCard` + cost model so Dynamo *can* front us · a framework-free OpenAI
server (`/v1/chat` SSE, `/v1/images`, `/v1/videos` job+poll, `/v1/models`, `/health`, `/metrics`).
**Workflows are first-class servables** — a `workflow_id` is addressable in the same namespace as a
`model_id` (see designv4 §9.6).

```python
import asyncio
from v2.models import build_default_engine, build_omni_engine
from v2.runtime import AsyncEngine
from v2.serving import OmniOpenAIServer
async def serve():
    eng = build_default_engine(); build_omni_engine(eng)
    host, port = await OmniOpenAIServer(AsyncEngine(eng)).serve(port=8000)
asyncio.run(serve())
```

## Deliberately out of scope

The ComfyUI workflow compiler, **WebRTC realtime transport** (the interactive *session* logic is built;
the realtime *wire* is not), full LLM-grade AR serving (radix prefix trees, speculative decoding), and
real *distributed* parallelism (the mesh builder is a CPU fake; multi-node is *multiple* pools fronted
by the fleet). A *live* Dynamo integration is also out — we ship the adapter + contract, and `LocalFleet`
covers the fleet role without it. The honest open questions (does step-level scheduling pay; the <100ms
interactive latency target; quality/C4) are GPU-port measurements, named in designv4 §13.
