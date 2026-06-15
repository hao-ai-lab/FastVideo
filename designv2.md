# FastVideo v2 Architecture

Status: proposal
Scope: a clean architecture for FastVideo as a model-native video, image, audio, and omni runtime.

This document intentionally treats the existing FastVideo implementation as replaceable. Compatibility adapters can be useful during rollout, but they are not design constraints. The goal is to define the architecture FastVideo should have if we are free to rebuild it around the best ideas from the local reference systems.

## Executive Summary

FastVideo v2 should be a model-native runtime, not a pipeline wrapper and not a deployment graph with models hidden behind opaque stages.

The center of the system is the Model Plane:

- It owns model families, components, checkpoints, capabilities, loop kernels, cache contracts, parity contracts, and parallelism constraints.
- It exposes typed loop semantics for diffusion denoise, autoregressive reasoning, video continuation, audio generation, VAE tiling, encoder chunks, reward evaluation, and training forwards.
- It inverts model loops so the runtime owns loop lifecycle, scheduling, admission, cancellation, cache checks, and tracing while model code owns typed state transitions and kernel execution.
- It lets one resident model instance run multiple loop types without pretending they are separate engines.

This matters most for omni models such as Cosmos3. A single model can use shared weights and packed multimodal state across reasoning, diffusion, action conditioning, and sound. A pure stage graph *can* co-locate shared-weight MoT as one opaque stage — vllm-omni's `bagel_single_stage`/`lance` do exactly this, running one resident Bagel instance with co-resident `*_moe_gen` experts (`bagel_transformer.py:718,726,385`) that do AR `generate_text` (`pipeline_bagel.py:677`) and diffusion `generate_image` (`:808`) in one request on shared weights. The loss is not expressibility; it is runtime *visibility and schedulability*. That stage is opaque: `RequestScheduler` schedules it at request granularity with `max_num_running_reqs` forced to 1 (`diffusion_engine.py:166-169`), so the runtime can never see, step-schedule, batch, or preempt the AR+denoise interleaving inside it. v2's model-native loops make exactly that interleaving runtime-visible.

The rest of the runtime compiles from the model contract:

- Program Plane: typed loop programs and workflow graphs.
- Request, Session, and Artifact Plane: user jobs, realtime sessions, streaming media, cancellation, and stored outputs.
- Runtime and Scheduler Plane: a general WorkUnit scheduler for diffusion steps, AR tokens, encoder chunks, VAE tiles, reward batches, and transfers.
- Memory, Cache, and Transport Plane: typed cache managers, cache keys, memory reservations, tensor transfer, and graph capture.
- Training and RL Plane: rollout, logprob identity, reward, weight sync, and behavior records using the same loop definitions as serving.
- Deployment and Fleet Plane: role pools, placement, routing, SLOs, health, and Dynamo-style orchestration integration above the engine.

The shortest version:

```text
Clients and products
  -> sessions, requests, artifacts
  -> typed programs
  -> model plane
  -> work scheduler
  -> model instances, caches, transports
  -> deployment pools and fleet routing
```

FastVideo v2 should borrow aggressively from vLLM, SGLang, vLLM-Omni, SGLang-Omni, Dynamo, ComfyUI, diffusers modular pipelines, xDiT, TorchTitan, verl-omni, miles, cosmos-rl, Dreamverse, and LiveKit agents. It should not copy any one of them as the core.

## Design Tenets

1. Model-native first

   The primary abstraction is a model family with components and loop kernels. Stages, endpoints, workflows, and training jobs are derived views over that model.

2. Loops are first-class

   Video systems are dominated by loops: denoise timesteps, AR decoding, streaming continuation, VAE tiles, reward batches, optimizer steps, and media chunks. A v2 runtime must schedule loop work directly instead of reducing everything to a single `forward`.

3. Loop inversion is the runtime contract

   Model code should not hide long-running while loops behind a stage call. The runtime owns loop lifecycle and asks model-owned LoopKernels to initialize state, plan the next step, execute declared WorkUnits, and finalize results.

4. One model instance can serve multiple loop semantics

   An omni model may use the same resident weights for reasoning text, diffusion latents, action tokens, audio tokens, and encoders. The architecture must support this directly.

5. Typed state over dict state

   Request state, model state, loop state, cache state, artifact state, and behavior records should be typed. Free-form dicts are allowed only at product boundaries or experimental extension points.

6. Cache correctness is a contract, not an optimization

   Cache keys must include model version, adapter version, precision, shape, layout, scheduler step, guidance, seed, input hashes, and parallel plan. Incorrect reuse is worse than no reuse.

7. Scheduling admits only work with reserved resources

   Admission should reserve compute budget, memory/cache budget, transfer budget, graph-capture shape, and output sinks. Running work is prioritized, and waiting work enters only when its required resources are real.

8. Training and serving share model loops — this collocation is the moat

   RL rollouts, reward evaluation, logprob recomputation, and inference should use the same ModelSpec and LoopSpec definitions. This is not just drift-avoidance hygiene; it is the central, durable moat. Rollout forward *is* serve forward plus capture — same loop, same caches, same batcher, same numerics — so every serving optimization is automatically a rollout optimization, and there is one numerics surface with no rollout-vs-train kernel gap to patch. Separate serving-only and training-only interpretations create silent drift and force a second runtime plus a correction layer; collocation of inference and post-training (including RL) deletes both.

9. Deployment orchestration is above the engine

   Dynamo-style fleet routing, role pools, and SLO planning are critical, but they should orchestrate FastVideo engines rather than become the core model runtime.

10. Workflows are compiled, not interpreted as the core runtime

   ComfyUI-style workflow graphs are valuable product interfaces. In v2, they compile into typed ProgramSpecs where possible and fall back to external nodes only where necessary.

11. Parallelism is part of the model contract

   Tensor, sequence, context, CFG, data, pipeline, VAE, expert, and shard parallelism are not launch-script details. They affect cache keys, scheduling, transport, capture, and parity.

## Reference Synthesis

The reference repos point to a combined design, but each has a different center of gravity.

| Reference | Adopt | Reject or constrain |
| --- | --- | --- |
| Cosmos3 official and FastVideo Cosmos3 port | Shared model instance across reasoning, diffusion, action, and sound; packed multimodal sequences; explicit scheduler parity; component parity matrices. | Do not encode Cosmos3-specific branching into global runtime abstractions. Make it a strong ModelSpec, not the framework itself. |
| vLLM core | Running-first scheduler, token/encoder budgets, model-state ownership, KV and encoder cache managers, CUDA graph dispatch, CuMem sleep/wake, KV transfer connector split. | Do not copy token-only scheduling. Generalize to arbitrary WorkUnits. |
| SGLang multimodal generation | Role pools, request lifecycle state, capacity-aware dispatch, transfer manifests, cache integration points, distributed process group setup. | Avoid large mutable `Req`/`ForwardBatch` state bags as the stable API. Avoid single-item diffusion scheduling. |
| vLLM-Omni | Stage pipeline config, stage pools, affinity, connector abstraction, background orchestrator, diffusion engine split; the frozen-`PipelineConfig` + editable-`DeployConfig` split fused by `merge_pipeline_deploy` (adopted directly). | Stages are deployment topology, not the model abstraction. Diffusion scheduling must be richer than vllm-omni's default. vllm-omni *does* have a step scheduler (`SupportsStepExecution` / `StepScheduler`, whose own docstring "advances one denoise step per update" is the placeholder), but it is opt-in (`step_execution` defaults False), homogeneous-`SamplingParamsKey`-gated, Qwen-Image-only, and off in every deploy; v2 generalizes it to always-on, heterogeneous step scheduling. |
| SGLang-Omni | Stage schema with `next`, `wait_for`, `merge_fn`, `stream_to`, relay configs, backpressure queues, stage runtime as IO shell. | Opaque factories and generic stages cannot replace typed model loops. |
| Dynamo | Fleet orchestration above engines, disaggregated role pools, KV-aware routing, KVBM, SLA planning, cold-start and weight streaming, multimodal E/P/D lessons. | Do not make Dynamo the engine core. FastVideo should export engine capability and cost to Dynamo-like orchestrators. |
| ComfyUI | Workflow graph, lazy nodes, dynamic subgraphs, node cache signatures, UI events, model memory management. | Do not make dynamic node execution the serving or training core. Also treat source-code reuse carefully because of licensing. |
| diffusers modular pipelines | Component specs, pipeline state, block state, load IDs, expected components/configs, loop and conditional blocks, offload hooks. | Avoid a Python-only pipeline interpreter as the performance boundary. Use it as a manifest and authoring influence. |
| xDiT | DiT parallelism catalog: USP, ring, Ulysses, PipeFusion, CFG parallel, data parallel, parallel VAE, validation of world-size products. | Do not leave parallel strategy outside the runtime. |
| TorchTitan | Named mesh axes, ParallelDims validation, ModelSpec discipline, trainer-generator state sharing, logprob identity checks. | Do not make FastVideo training depend on TorchTitan internals, but adopt the mesh and spec discipline. |
| verl-omni | Multimodal RL trainer, rollout adapters, diffusion logprob capture, per-step latents/timesteps, async rewards, algorithm contracts. | Avoid reconstructing rollout behavior after the fact. Capture it in the serving loop. |
| miles | True-on-policy contracts, deterministic flags, batch-invariant modes, rollout session routing, weight update lifecycle. | Do not treat policy sync as an ad hoc side channel. |
| cosmos-rl | Single-controller role model, policy/rollout/reference roles, dynamic registration, payload transport registry, weight-version routing. | Avoid Redis-specific or repo-specific control assumptions in the core. |
| Dreamverse | Sessions, prompt memory, GPU assignment, typed worker IPC, media init/chunk/complete events, cancellation and model reload. | Keep product/session behavior outside the model core but make it first-class in the request plane. |
| LiveKit agents | Realtime sessions, audio/video push streams, interruptions, turn/activity state, event-driven instrumentation. | Do not force realtime voice abstractions onto offline generation jobs. |

## Relationship to `design.md`

This v2 design is not an incremental update to `design.md`.

The major difference is that `designv2.md` makes the Model Plane non-negotiable and central. Pipeline stages, disaggregated pools, workflow nodes, training rollouts, and product endpoints all compile from a model/component/loop contract.

The previous direction was primarily concerned with runtime topology, stage orchestration, and disaggregated execution. Those are still needed, but v2 demotes them below the model contract. The runtime should not ask, "Which stage should run next?" before it knows, "Which model loop is this, what state does it own, what caches are valid, and what behavior contract must be preserved?"

The practical differences:

- v2 has a real Model Plane.
- v2 has typed loop kernels instead of only stage handlers.
- v2 uses loop inversion: the runtime owns loop lifecycle while model code exposes typed step planning and execution.
- v2 schedules WorkUnits instead of only requests, tokens, or pipeline stages.
- v2 treats sessions, artifacts, and streams as stable runtime objects.
- v2 makes cache keys and parallel plans part of correctness.
- v2 unifies serving and RL rollout behavior around BehaviorRecords.
- v2 treats Dynamo-style orchestration as a fleet layer above FastVideo engines.
- v2 supports ComfyUI-style workflow import by compilation, not by making dynamic workflow interpretation the core runtime.

## Top-Level Architecture

```text
                         +----------------------------------+
                         | API and Product Plane             |
                         | Python, CLI, OpenAI API, UI, RTC  |
                         +----------------+-----------------+
                                          |
                         +----------------v-----------------+
                         | Request, Session, Artifact Plane  |
                         | jobs, streams, cancel, outputs    |
                         +----------------+-----------------+
                                          |
                         +----------------v-----------------+
                         | Program Plane                     |
                         | typed loops, branches, workflows  |
                         +----------------+-----------------+
                                          |
                         +----------------v-----------------+
                         | Model Plane                       |
                         | specs, components, loop kernels   |
                         | checkpoints, capabilities, parity |
                         +----------------+-----------------+
                                          |
          +-------------------------------+-------------------------------+
          |                                                               |
+---------v----------+                                      +-------------v-------------+
| Runtime Scheduler  |                                      | Training and RL Plane      |
| WorkUnits, budgets |                                      | rollout, rewards, sync     |
+---------+----------+                                      +-------------+-------------+
          |                                                               |
+---------v---------------------------------------------------------------v-------------+
| Memory, Cache, Transport, and Compile Plane                                            |
| cache managers, tensor relay, CUDA graphs, memory reservations, weight movement         |
+---------+---------------------------------------------------------------+-------------+
          |
+---------v----------+
| Deployment Plane   |
| role pools, fleet  |
| routing, SLOs      |
+--------------------+
```

The dependency direction is intentional:

- Product APIs do not define model semantics.
- Workflows do not define model semantics.
- Deployment topology does not define model semantics.
- Training does not redefine model semantics.
- All of them reference the Model Plane.

## Plane Responsibilities

### API and Product Plane

This plane exposes usable surfaces:

- Python SDK.
- CLI.
- OpenAI-compatible or OpenAI-inspired HTTP APIs.
- ComfyUI adapter.
- Dreamverse-style web session API.
- WebSocket and WebRTC streaming endpoints.
- Training launch and rollout APIs.
- Batch and offline artifact-generation jobs.

This plane should be thin. It validates user intent, creates requests or sessions, selects a program, and subscribes to events. It should not own model loading, denoising logic, cache policy, or parallelism.

### Request, Session, and Artifact Plane

This plane owns user-facing runtime state:

- `Request`: one generation, scoring, encoding, training sample, or conversion job.
- `Session`: a long-lived interactive context with prompt memory, media streams, cancellation, and partial updates.
- `Artifact`: a typed output such as video, image, audio, latent tensor, text, metadata, score, trace, or training batch.
- `Stream`: ordered event channel for previews, media chunks, logs, progress, and final outputs.
- `CancelScope`: structured cancellation target, such as request, loop, stream, or session.

The Dreamverse and LiveKit references are the right mental model: sessions are not just IDs in a batch. They can receive new user input, push media, interrupt ongoing work, preserve prompt memory, and stream output while compute is still running.

Core event types:

```text
request.accepted
request.queued
request.started
request.progress
request.cancelled
request.failed
request.completed

session.created
session.updated
session.interrupted
session.closed

artifact.created
artifact.updated
artifact.completed

media.init
media.chunk
media.complete

trace.step
trace.cache
trace.transfer
trace.scheduler
```

Media events should be typed like Dreamverse worker IPC: invalid combinations should be unrepresentable. A `media.chunk` must know the stream, byte range or shared-buffer reference, codec/container, timestamp range, and whether it is preview or final.

### Program Plane

The Program Plane describes what should happen for a request or session.

It is not the same as the Model Plane. A model says what loops and capabilities exist. A program composes them for a task.

Program kinds:

- `InlineProgram`: multiple loops over one resident model instance. This is the default for omni models.
- `DisaggregatedProgram`: separable role or stage pools, such as encoder -> denoiser -> decoder.
- `WorkflowProgram`: a compiled workflow graph, usually from a ComfyUI-like frontend.
- `TrainingProgram`: rollout, reward, logprob recompute, advantage, and update flows.
- `RealtimeProgram`: streaming input/output with interruption and turn/activity state.

Program node kinds:

- `ModelLoopNode`: invokes a named model loop kernel.
- `ComponentNode`: invokes one component directly, such as VAE encode/decode or encoder embed.
- `ExternalNode`: calls an external service or plugin.
- `ArtifactNode`: reads or writes artifacts.
- `ControlNode`: branches, joins, waits, retries, or cancels.
- `StreamNode`: emits partial outputs.
- `TransferNode`: materializes or moves tensors between roles.

Program edge kinds:

- `TensorEdge`: tensor or tensor bundle.
- `ArtifactEdge`: persisted artifact.
- `StreamEdge`: ordered event/media stream.
- `ControlEdge`: dependency without payload.
- `CacheEdge`: explicit cache publication or lookup.
- `BehaviorEdge`: rollout or parity record.

Programs support:

- loops with explicit step boundaries;
- branches and merges;
- stream sinks;
- cancellation boundaries;
- retry policy;
- cache publication policy;
- placement hints;
- graph capture hints;
- training behavior capture.

The SGLang-Omni schema is a useful vocabulary for `next`, `terminal`, `wait_for`, `merge_fn`, `stream_to`, `project_payload`, and relay config. In v2, those fields must reference typed model outputs and state, not opaque stage payloads.

### Model Plane

The Model Plane is the core of FastVideo v2.

It owns:

- model families;
- model variants;
- component inventories;
- checkpoint manifests;
- tokenizer and processor manifests;
- scheduler definitions;
- loop kernels;
- capability matrices;
- parallelism constraints;
- cache contracts;
- parity contracts;
- precision and quantization contracts;
- adapter and LoRA compatibility;
- training and rollout behavior contracts.

#### ModelSpec

Every model family is described by a `ModelSpec`.

Conceptual shape:

```python
class ModelSpec:
    model_id: str
    family: str
    variants: list[ModelVariantSpec]
    components: dict[str, ComponentSpec]
    loops: dict[str, LoopSpec]
    capabilities: CapabilityMatrix
    checkpoint: CheckpointManifest
    processors: dict[str, ProcessorSpec]
    parallelism: ParallelismContract
    caches: dict[str, CacheContract]
    parity: ParityContract
    training: TrainingContract | None
```

The ModelSpec is a declarative contract and a runtime factory. It must be strict enough for validation and flexible enough to represent new omni models.

Examples of capabilities:

- `text_to_video`
- `image_to_video`
- `video_to_video`
- `text_to_image`
- `image_to_image`
- `audio_to_video`
- `video_to_audio`
- `reasoning_text`
- `action_conditioning`
- `multimodal_understanding`
- `latent_scoring`
- `reward_scoring`
- `policy_rollout`
- `logprob_recompute`
- `vae_encode`
- `vae_decode`
- `streaming_video_continuation`

#### ComponentSpec

Components are named, typed, loadable parts of a model.

Examples:

- diffusion transformer;
- VAE;
- text encoder;
- vision encoder;
- audio encoder;
- audio tokenizer;
- reasoner tower;
- scheduler;
- tokenizer;
- processor;
- conditioner;
- adapter;
- reward head;
- safety head.

Conceptual shape:

```python
class ComponentSpec:
    component_id: str
    kind: str
    load_id: str
    checkpoint_keys: KeyMapping
    config_schema: type
    input_schema: type
    output_schema: type
    precision_policy: PrecisionPolicy
    placement_policy: PlacementPolicy
    parallelism_constraints: ParallelismConstraint
    cache_contracts: list[CacheContract]
    parity_tests: list[ParityTestSpec]
```

The diffusers modular `ComponentSpec` and TorchTitan `ModelSpec` are the right inspiration, but FastVideo needs stricter contracts because distributed video inference and RL rollout depend on shape, layout, and scheduler correctness.

#### LoopSpec

Loops are named model behaviors with stable state and scheduler boundaries.

Examples:

- `ar_decode`
- `diffusion_denoise`
- `flow_matching_denoise`
- `video_continuation`
- `vae_tile_encode`
- `vae_tile_decode`
- `encoder_chunk`
- `audio_token_decode`
- `action_condition`
- `reward_batch`
- `train_forward`
- `logprob_recompute`

Conceptual shape:

```python
class LoopSpec:
    loop_id: str
    kind: str
    input_schema: type
    output_schema: type
    state_schema: type
    behavior_record_schema: type | None
    step_policy: StepPolicy
    preemption_policy: PreemptionPolicy
    cache_policy: CachePolicy
    cfg_policy: CFGPolicy | None
    graph_capture_policy: GraphCapturePolicy
    valid_parallel_plans: list[ParallelPlanPattern]
```

Loop kernels implement:

```python
class LoopKernel:
    def init_state(self, request, model_state, runtime_context) -> LoopState: ...
    def plan_step(self, loop_state, budgets, cache_state) -> WorkPlan: ...
    def run_step(self, work_unit, model_state, runtime_context) -> StepResult: ...
    def finalize(self, loop_state, model_state, runtime_context) -> LoopResult: ...
```

The scheduler calls loop kernels through these methods. It does not call arbitrary Python stage code on a request dict.

#### Loop Inversion Contract

Loop inversion is the boundary between model authors and the runtime scheduler.

The old shape is:

```text
pipeline or stage owns the while loop
  -> model runs until completion
  -> runtime observes only start and finish
```

The v2 shape is:

```text
runtime owns the loop lifecycle
  -> model loop initializes typed state
  -> model loop plans the next schedulable step
  -> scheduler admits, batches, places, and executes WorkUnits
  -> model loop consumes StepResults and advances typed state
  -> model loop finalizes outputs
```

This inversion is required for any operation that is iterative, memory-heavy, cache-sensitive, stream-producing, cancellable, distributed, or training-relevant.

Use loop inversion for:

- diffusion denoise timesteps;
- autoregressive prefill and decode;
- streaming video continuation;
- VAE tile encode/decode;
- encoder chunking;
- audio token generation;
- reward and logprob batches;
- rollout sampling;
- graph-captured repeated kernels.

Do not force loop inversion onto simple one-shot transforms. A static prompt rewrite, metadata transform, or small artifact conversion can remain a `ComponentNode` or `ExternalNode` if it does not need scheduler visibility.

Hard rules:

- `init_state` may allocate or bind typed loop state, but it must not perform the whole loop.
- `plan_step` must be pure with respect to model weights and external side effects. It returns a typed `WorkPlan`; it does not run GPU kernels.
- `plan_step` must declare resource needs, cache reads/writes, graph-capture shape, placement constraints, output sinks, and cancellation boundary.
- `run_step` executes only the declared WorkUnit or compatible batch of WorkUnits.
- `run_step` returns a typed `StepResult`; it does not mutate arbitrary request dictionaries.
- loop state transitions must be explicit and versioned.
- cancellation and preemption happen only at declared loop boundaries.
- cache publication happens through `CacheRecord`, not hidden component fields.
- training-relevant loops must emit or update `BehaviorRecord` data as part of the loop contract.

The main failure mode is callback soup: model authors put hidden scheduling, hidden side effects, or hidden state mutation inside `plan_step` and the runtime loses control again. That should be treated as a contract violation.

Loop granularity should be chosen by runtime value, not by philosophical purity:

- too coarse: the runtime cannot cancel, batch, reserve memory, stream, or record behavior at useful points;
- too fine: scheduler overhead dominates and graph capture becomes noisy;
- good default: one denoise step or window, one AR decode batch, one encoder chunk, one VAE tile batch, one reward/logprob batch.

The runtime may fuse adjacent WorkUnits after planning if their contracts are compatible. Fusion is an optimization; the unfused loop boundary remains the semantic model.

#### CFG Policy

Classifier-free guidance is a first-class loop concept, not a launch-script flag and not only a parallelism axis. One shared denoise step body plus a swappable `CFGPolicy` expresses every guidance variant: classic two-forward, batched one-forward (stack the conditional and unconditional branches, run one forward, then chunk), adaptive-gate (reuse a cached guidance delta with model-id invalidation), per-modality guidance, and embedded-guidance (a single-branch identity-combine policy where guidance rides inside the forward kwarg — this is a degenerate CFG policy, *not* "no CFG").

This is proven by precedent: vllm-omni's `CFGParallelMixin.predict_noise_maybe_with_cfg` + `combine_cfg_noise` (`vllm_omni/diffusion/.../cfg_parallel.py:76-212`) already unifies sequential two-forward, batched, and cfg-parallel under one predict/combine pair. So vllm-omni does not have three independent CFG systems; it has one in-loop predict/combine pair plus one parallelism axis. v2 names that explicitly as a three-layer boundary:

1. `CFGPolicy` lives **in the loop**: it owns the branch vocabulary, the combine formula, and per-request mutable state (the adaptive-gate cached delta is the canonical state case). Batched-vs-two-forward is a *dispatch detail inside one policy*, not a separate mechanism.
2. `cfgp` (cfg-parallel) is a **parallelism axis**: it shards `policy.predict`'s branches across ranks and runs the *same* rank-invariant combine on every rank. It composes under any policy — never own both a `BatchedCFG` policy and a `cfgp` group for the same concern.
3. Companions are an **orchestrator pattern**: a request is split into companion sub-requests upstream of diffusion, so the loop receives already-computed conditioning and is unchanged.

Conceptual shape:

```python
class CFGPolicy:
    branch_vocabulary: list[str]          # e.g. ["cond"], ["cond", "uncond"], per-modality
    def predict(self, branches, model_state, ctx) -> dict[str, Tensor]: ...
    def combine(self, predictions, request_state) -> Tensor: ...
    state_schema: type | None             # adaptive-gate cached delta lives here
    invalidation: InvalidationPolicy      # cached delta keyed on model id / weights version
```

Caveat: `combine` runs in the step body's numeric space. Cosmos combines in x0-space after EDM scaling, not in noise-space, so the policy must declare the space it combines in; the runtime never assumes noise-space.

#### Model Instance

A `ModelInstance` is a resident loaded model with component instances, state, caches, compiled graphs, and placement.

```python
class ModelInstance:
    spec: ModelSpec
    weights_version: str
    adapter_versions: dict[str, str]
    components: dict[str, ComponentInstance]
    model_state: ModelState
    cache_manager: CacheManager
    compile_cache: CompileCache
    parallel_plan: ParallelPlan
```

A request may run several loops against the same `ModelInstance`. This is the key difference from a stage-only design.

For Cosmos3-style omni:

```text
same ModelInstance
  -> ar_decode loop for reasoner text
  -> diffusion_denoise loop for video latents
  -> action_condition loop for action tokens
  -> audio_token_decode loop for sound
  -> vae_tile_decode loop for final pixels
```

The program may expose these as stages for observability or deployment, but the model instance remains the owner of components and shared state.

### Runtime and Scheduler Plane

The scheduler should generalize vLLM's token scheduler and SGLang's request lifecycle into a WorkUnit scheduler.

Loop inversion is what makes this scheduler possible. The scheduler does not need model-specific while loops, but it does need model-provided step plans with typed resources, cache contracts, placement constraints, and completion semantics.

#### WorkUnit

A `WorkUnit` is the smallest schedulable runtime action with a resource reservation and a loop boundary.

Kinds:

- `ar_token`
- `ar_prefill`
- `diffusion_step`
- `diffusion_window`
- `encoder_chunk`
- `vae_tile`
- `audio_chunk`
- `reward_batch`
- `logprob_batch`
- `transfer_send`
- `transfer_recv`
- `cache_load`
- `cache_save`
- `artifact_write`
- `graph_capture`

Conceptual shape:

```python
class WorkUnit:
    request_id: str
    session_id: str | None
    program_id: str
    loop_id: str
    model_instance_id: str
    kind: str
    step_range: StepRange
    priority: Priority
    deadline: Deadline | None
    resource_request: ResourceRequest
    cache_plan: CachePlan
    placement_hint: PlacementHint
    output_sinks: list[OutputSink]
    cancel_scope: CancelScope
```

Resources:

- compute budget;
- memory budget;
- cache blocks;
- transfer bandwidth;
- graph-capture shape;
- stream output capacity;
- host staging buffers;
- role-pool slots.

Admission rule:

```text
Do not admit a waiting WorkUnit unless its required resources can be reserved.
```

#### Cost Currency

Different WorkUnit kinds are not commensurable in their native units. A `diffusion_step` re-attends at O(L^2) full price every step; an `ar_token` is roughly O(context) against a KV cache. Counting "two AR tokens" against "one denoise step" is meaningless, which is exactly why vllm-omni's per-stage *count* budgets cannot co-schedule an AR token against a denoise step — and that is precisely the MoT case.

v2 names one shared currency: **predicted GPU-seconds**. Every WorkUnit converts to predicted GPU-time through a per-`(model, phase, shape)` cost model, Profiler-calibrated, *before* admission. The scheduler then co-schedules and preempts AR tokens and denoise steps against one budget instead of incomparable count budgets.

```text
work_unit -> cost_model[(model, phase, shape)] -> predicted_gpu_seconds -> admission
```

This is the same `CostModel` the engine exports on its `DeploymentCard` (see Deployment and Fleet Plane). One object, two consumers: local admission/co-scheduling and fleet routing. Without it, v2 could co-schedule AR tokens against denoise steps no better than vllm-omni's per-stage count budgets.

Scheduling policy:

- Running loops first, as in vLLM.
- Waiting admission only after resource reservation.
- Preempt at loop step boundaries, not mid-kernel.
- Cancel at declared cancellation boundaries.
- Prefer cache hits when latency and fairness allow.
- Batch WorkUnits by compatible model instance, loop kind, shape, precision, graph-capture key, and parallel plan.
- Avoid starving long diffusion loops with short AR requests.
- Expose fairness, priority, and SLO knobs per deployment.

#### Scheduler Layers

The scheduler should have multiple layers:

1. `RequestScheduler`: accepts requests and sessions, selects programs, creates loop states.
2. `LoopScheduler`: asks LoopKernels to plan next WorkUnits.
3. `BatchScheduler`: groups compatible WorkUnits into executable batches.
4. `PlacementScheduler`: chooses worker, role pool, model instance, and device mesh.
5. `TransferScheduler`: schedules tensor, cache, and artifact movement.
6. `AdmissionController`: enforces memory, cache, transfer, and SLO budgets.

Each layer should be testable separately.

#### Runtime Worker

A worker owns:

- one or more `ModelInstance` objects;
- a device mesh;
- cache managers;
- memory allocator;
- compile cache;
- transfer agent;
- event sink;
- health reporter.

The worker API should be narrow:

```text
load_model(spec, checkpoint, parallel_plan)
prepare_loop(request, program_node)
execute_work(batch_of_work_units)
cancel(cancel_scope)
publish_cache(cache_record)
transfer(manifest)
sleep(tags)
wake(tags)
health()
```

Workers should not interpret product APIs or workflow JSON directly.

### Memory, Cache, Transport, and Compile Plane

Video and omni inference are memory systems as much as compute systems. This plane must be explicit.

#### Cache Classes

Cache managers should be typed by data class:

- token KV cache;
- video/DiT attention cache;
- text encoder output cache;
- vision encoder output cache;
- audio encoder output cache;
- first-block and residual caches;
- latent trajectory cache;
- VAE tile cache;
- scheduler state cache;
- reward model embedding cache;
- compiled graph cache;
- weights and adapter cache;
- artifact and media cache.

Each cache class defines:

- valid producers;
- valid consumers;
- key schema;
- value schema;
- memory class;
- eviction policy;
- transfer policy;
- invalidation policy;
- parity implications.

#### CacheKey

Cache keys must be structured.

```python
class CacheKey:
    model_id: str
    component_id: str
    loop_id: str | None
    weights_version: str
    adapter_versions: dict[str, str]
    precision: str
    parallel_plan_hash: str
    shape_signature: str
    layout_signature: str
    scheduler_signature: str | None
    guidance_signature: str | None
    seed: int | None
    input_hashes: dict[str, str]
    step_index: int | None
    cache_contract_version: str
```

The cache key is part of correctness. If any of these fields can change output semantics, it belongs in the key.

#### Memory Management

Borrow from vLLM CuMem and ComfyUI model management:

- tagged memory pools;
- sleep/wake by tags;
- model unload and reload;
- adapter eviction;
- cache eviction under pressure;
- per-role memory budgets;
- host pinned memory staging;
- shared memory for local worker transfer;
- GPU memory reservation before admission.

Admission and memory must be integrated. A scheduler that admits a diffusion step and then discovers it cannot allocate the VAE tile or KV block is wrong.

#### Transport

Transport should be pluggable and manifest-based.

Transport backends:

- in-process reference;
- multiprocessing shared memory;
- CUDA IPC;
- NCCL;
- UCXX;
- NIXL;
- RDMA;
- filesystem or object-store artifact transfer;
- Dynamo KVBM-compatible backend.

Tensor transfer manifests include:

- tensor names;
- shapes;
- dtypes;
- strides/layout;
- device or host location;
- producer and consumer ids;
- lifetime;
- cache key if applicable;
- transfer priority;
- completion event.

SGLang multimodal generation's staged transfer manager and SGLang-Omni relay payload extraction are good starting points. v2 should make these typed and independent of a single request object shape.

#### Compile Cache

CUDA graphs, `torch.compile`, custom kernels, and shape-specialized execution should be managed through a `CompileCache`.

Compile keys should include:

- model id;
- component id;
- loop id;
- work-unit kind;
- shape signature;
- precision;
- parallel plan;
- kernel backend;
- graph capture policy;
- runtime flags.

Graph capture should be planned by the scheduler. Padding, bucketing, and capture sizes affect admission and batching.

### Parallelism Plane

Parallelism should be explicit and named.

`ParallelPlan` axes:

- `dp`: data parallel;
- `tp`: tensor parallel;
- `sp`: sequence parallel;
- `cp`: context parallel;
- `cfgp`: classifier-free guidance parallel;
- `pp`: pipeline or PipeFusion parallel;
- `vae`: VAE parallel;
- `ep`: expert parallel;
- `fsdp`: fully sharded data parallel;
- `role`: stage or role replicas;
- `replica`: model instance replicas.

Conceptual shape:

```python
class ParallelPlan:
    axes: dict[str, int]
    mesh_order: list[str]
    placement: PlacementSpec
    communication: CommunicationSpec
    compatibility: CompatibilitySpec
```

Validation:

- axis product must match world size where required;
- component constraints must be satisfied;
- loop constraints must be satisfied;
- cache layout must match parallel layout;
- transfer paths must exist;
- unsupported combinations fail at load time, not halfway through execution.

TorchTitan's named mesh axes and xDiT's validation discipline are the right model. Degree-one axes should still exist as fake or trivial groups so component code does not need special cases.

Parallelism is not only a training concern. It affects:

- model load;
- cache key;
- attention backend;
- sequence layout;
- graph capture;
- transfer manifest;
- checkpoint conversion;
- rollout parity;
- benchmark comparability.

### Training and RL Plane

FastVideo v2 should make RL and online training first-class without forking model semantics from serving. Collocation of inference and post-training (including RL) is the central, durable moat — it should be foregrounded, not treated as a hygiene footnote.

#### Shared Serving and Training Loops

Serving:

```text
request -> ProgramSpec -> LoopSpec -> WorkUnits -> artifacts
```

RL rollout:

```text
prompt batch -> ProgramSpec -> LoopSpec -> WorkUnits -> BehaviorRecords -> rewards -> update
```

The loop kernel is shared. The difference is output capture and training policy, not a different interpretation of the model.

##### Today: rollout vendors its own loop

Today rollout does *not* reuse the inference denoise loop; it vendors its own. The landed `DiffusionSampler` says so outright (`fastvideo/train/methods/rl/common/sampling.py:92-99` docstring: "intentionally does not call FastVideo's full inference pipelines"; `:128-147` a hand-rolled for-loop, `conditional=True`, `attn_kind=dense`). `DMD2._student_rollout` (`dmd2.py:430-550`) is a *second* independently vendored loop (`attn_kind=vsa`), so the "fourth copy" is concretely two RL/distill copies of the denoise loop.

##### Under the design: rollout = serve loop + capture

Under v2, rollout is the *same* shared `DenoiseLoop` plus a `CFGPolicy` (for NFT: conditional-only / identity — `predict_noise` already carries `conditional: bool` + `cfg_uncond` as the policy knob, `train/models/base.py:142-152`) plus an `OutputSpec(capture=behavior)` plus the step scheduler plus the caches. The caveats are already named by the design, not new mechanisms:

- old-policy weights are a `WeightSyncPlan` role (rollout samples from `self.old`, decay-blended, `diffusion_nft.py:858-862`);
- the dense-attn pin is a Precision/Attn policy on the card;
- likelihood-free NFT is the C2 *behavioral* rung (no log-probs; `old_deviate` / reference-MSE in prediction space, `diffusion_nft.py:777-778`).

##### Why rollout is a strictly better batching case than open-world serving

RL rollout is a *strictly better* cross-request batching case than open-world serving. A GRPO/NFT group is K identical-config samples of one prompt (`num_video_per_prompt: 24`, `sample_train_batch_size: 6`, `num_batches_per_epoch: 48`, `num_steps: 25`, single-frame 448x832, 4 GPUs — `diffusion_nft_pick_clip.yaml:33-37,52-55,72-91`; `distributed_k_repeat_indices` with `repeats_per_prompt=24`, `diffusion_nft.py:331-338`). Same shape, same schedule, same CFG branch, K=24 wide → *zero* bucketing needed, whereas serving must bucket heterogeneous resolutions, step counts, and CFG. The K samples also share one prompt embedding, so a content-hash feature cache computes the text encoder *once* and reuses it across all 24 (a 24x text-encode reduction the vendored loop cannot express — it carries the embedding per-sample, `diffusion_nft.py:308-309`). Per GPU per outer epoch that is 6 prompts x 48 batches = 288 prompt-slots, each a 24-wide homogeneous denoise batch — near-ideal co-batchable work that the vendored Python loop runs one sample at a time.

##### What a standalone trainer-side sampler forgoes

A vendored trainer-side sampler forgoes, itemized:

1. cross-group / cross-request batching;
2. content-hash embedding / feature-cache reuse;
3. step scheduling and interleaving;
4. cost-aware admission;
5. component-granular sleep/wake (it holds student + old + reference resident simultaneously);
6. cache-dit / interceptor skip acceleration (forced dense, full 25-step ODE);
7. distilled few-step sampler selection (a 4-step card collapses 25 steps to 4).

##### The moat made concrete

Rollout forward *is* serve forward plus capture: same loop, same caches, same batcher, same numerics — so every serving optimization is automatically a rollout optimization, and there is one numerics surface with no rollout-vs-train kernel gap to patch. The alternative is the two-runtime tax: verl-omni re-implements Wan inside vLLM-Omni plus a `rollout_correction.py` of importance-sampling / rejection masks; miles' TIS/MIS, bitwise-logprobs, R3, and unified-FP8 are all mismatch patches. FastVideo's own landed NFT confirms it from the other side — a vendored bare-model loop with zero serving-grade optimizations. Collocation at FastVideo's 1-30B FSDP2 scale deletes the second runtime, the correction layer, and the third/fourth denoise-loop copies, and it makes the (recipe, runtime) flywheel honest because preferences are collected under the very serving profile that ships.

#### BehaviorRecord

A `BehaviorRecord` captures enough to prove or debug policy behavior.

Fields:

- request id;
- session id if any;
- model id;
- policy version;
- weights version;
- adapter versions;
- program id;
- loop id;
- seeds;
- scheduler trajectory;
- timesteps;
- latents or latent references;
- logits or logprobs where applicable;
- action tokens;
- sampled tokens;
- guidance settings;
- reward inputs;
- reward outputs;
- cache assumptions;
- precision;
- parallel plan;
- attention backend;
- deterministic flags;
- artifact references.

This combines the lessons from verl-omni, miles, cosmos-rl, and TorchTitan RL experiments: rollout information must be captured at generation time. Reconstructing it later is fragile.

#### Policy Versioning and Weight Sync

Weight sync has a lifecycle:

1. Freeze admission for affected policy version or role.
2. Drain or boundary-stop in-flight loops according to policy.
3. Transfer new weights or deltas.
4. Update model instance weights version.
5. Invalidate incompatible caches and compiled graphs.
6. Publish new policy version.
7. Resume admission.

No component should silently mix cache state from old weights with new weights unless the cache contract explicitly permits it.

#### Consistency Ladder

Training and serving parity should be tested at multiple levels:

- C0: component parity, such as VAE, encoder, transformer block, scheduler step.
- C1: loop parity, such as denoise trajectory or AR logits.
- C2: logprob identity under fixed seeds and batch-invariant mode.
- C3: rollout distribution parity under allowed nondeterminism.
- C4: end-to-end artifact quality, such as SSIM, CLIP-like similarity, or reward agreement.

Likelihood-free methods have no log-probs, so the C2 rung is method-dependent. The landed DiffusionNFT, for example, never computes a log-prob; its C2 is *behavioral*: seeded final-sample and prediction-space identity (e.g. `old_deviate` / reference-MSE in prediction space), not log-prob identity. Each RL method declares both which consistency level it requires and which form of the C2 rung applies to it.

#### Roles

Training deployments use roles:

- policy;
- rollout;
- reference;
- reward;
- critic;
- evaluator;
- data;
- coordinator.

These roles are deployment concerns, but they reference the same ModelSpec and LoopSpec objects.

### Deployment and Fleet Plane

FastVideo v2 should integrate with Dynamo-style orchestration while preserving FastVideo's engine ownership.

The engine exports a `DeploymentCard`:

```python
class DeploymentCard:
    engine_id: str
    model_specs: list[str]
    capabilities: CapabilityMatrix
    role_pools: list[RolePoolSpec]
    supported_programs: list[str]
    supported_parallel_plans: list[ParallelPlan]
    cache_events: list[CacheEventSpec]
    transfer_endpoints: list[TransferEndpoint]
    cost_model: CostModel
    health_schema: HealthSchema
    slo_schema: SLOSchema
```

Fleet orchestration owns:

- global routing;
- tenant policy;
- cold start;
- role-pool scaling;
- cross-node transfer planning;
- placement by SLO;
- health and failover;
- multi-engine upgrades;
- global cache routing.

FastVideo engine owns:

- model load;
- loop execution;
- local scheduling;
- local memory;
- local cache;
- model-specific behavior;
- component parity;
- WorkUnit batching.

This line keeps Dynamo useful without making FastVideo a thin plugin.

### Workflow Plane

ComfyUI-style workflows should be supported through a compiler.

Workflow path:

```text
workflow JSON
  -> parse nodes and edges
  -> identify known model/component/program nodes
  -> compile typed subgraphs to ProgramSpec
  -> isolate unknown nodes as ExternalNode
  -> validate state, cache, placement, and artifacts
  -> execute through normal runtime
```

Workflow caching can borrow ComfyUI's node-signature and ancestry cache ideas, but v2 cache keys must still be typed and model-aware.

The product goal is compatibility and authoring flexibility. The runtime goal is not to become a dynamic node executor.

### Realtime Session Plane

Realtime generation needs more than batch request state.

Session features:

- push audio;
- push video;
- push text;
- append prompt;
- interrupt active generation;
- update conditioning state;
- stream partial media;
- stream reasoning text;
- handle user leave and rejoin;
- preserve prompt memory;
- reload model or adapter;
- time out inactive sessions.

Runtime state:

```python
class SessionState:
    session_id: str
    user_id: str | None
    active_programs: list[str]
    prompt_memory: PromptMemory
    input_streams: dict[str, StreamState]
    output_streams: dict[str, StreamState]
    artifact_refs: list[str]
    cancel_scopes: list[CancelScope]
    model_bindings: list[ModelBinding]
```

LiveKit agents are the right reference for interruptions, media streams, and activity state. Dreamverse is the right reference for GPU assignment, authoritative runtime session state, and typed media chunks.

### Observability and Control Plane

Observability must be structured enough for debugging correctness and performance.

Required event categories:

- request lifecycle;
- session lifecycle;
- scheduler decisions;
- admission decisions;
- batch composition;
- loop step timing;
- component timing;
- cache lookup/hit/miss/evict;
- transfer start/complete/fail;
- memory reservation and pressure;
- graph capture and replay;
- weight version changes;
- parity check results;
- artifact publication;
- cancellation and interruption.

Every event should carry:

- request id;
- session id if any;
- program id;
- model id;
- model instance id;
- loop id if applicable;
- worker id;
- role id;
- timestamp;
- trace id;
- relevant versions.

Tracing must be cheap enough to leave on in production at summary level and detailed enough to enable layer-by-layer divergence when debugging model ports.

### Configuration and Authoring

FastVideo v2 should separate authoring files by concern:

```text
model.yaml
  model family, variants, components, checkpoints, capabilities

loops.yaml
  loop specs, state schemas, step policies, cache policies

parallel.yaml
  valid parallel plans, placement constraints, mesh order

programs.yaml
  named generation, scoring, workflow, and training programs

deployment.yaml
  role pools, replicas, transport, SLOs, fleet integration

training.yaml
  rollout, reward, optimizer, weight sync, behavior capture
```

The Python API should load these into typed dataclasses or Pydantic models, then validate cross-file references before runtime.

Authoring rule:

```text
If a field affects correctness, it must be in the typed spec.
If a field only affects product presentation, it can stay in product config.
```

### Checkpoint and Conversion Plane

Checkpoint conversion is part of the Model Plane, not a loose script collection.

A `CheckpointManifest` defines:

- upstream source;
- source revision;
- expected files;
- component ownership;
- key mappings;
- shape mappings;
- dtype policy;
- sharding policy;
- adapter compatibility;
- checksum policy;
- conversion version;
- parity tests required after conversion.

Conversion output should include:

- converted weights;
- manifest;
- conversion report;
- unmatched keys;
- dtype and shape summary;
- parity status.

This follows the lesson from the Cosmos3 port: scheduler config, component layout, and key mapping mistakes can produce plausible but broken outputs. Conversion must end in executable parity, not only a saved state dict.

## Core Data Model

The following objects should become stable internal APIs:

```text
ModelSpec
ModelVariantSpec
ComponentSpec
LoopSpec
CapabilityMatrix
CheckpointManifest
ParallelPlan
ProgramSpec
RequestState
SessionState
ArtifactRecord
LoopState
ModelState
WorkUnit
WorkPlan
StepResult
CacheKey
CacheRecord
TransferManifest
BehaviorRecord
DeploymentCard
```

The anti-pattern to avoid:

```text
one giant mutable request object
  + arbitrary extra dict
  + component-specific fields
  + transport fields
  + training fields
  + streaming fields
```

That pattern is convenient early but becomes impossible to validate, schedule, cache, distribute, and test.

## Execution Examples

### Text to Video, Single Model Instance

```text
POST /v1/videos
  -> Request created
  -> Program selected: text_to_video_inline
  -> ModelSpec selected: cosmos3_video
  -> Loop init: text encoder chunk
  -> WorkUnits: encoder_chunk
  -> Loop init: diffusion_denoise
  -> WorkUnits: diffusion_step 0..N
  -> Loop init: vae_tile_decode
  -> WorkUnits: vae_tile
  -> Artifact: mp4
  -> Stream: previews and final media
```

The denoise loop and VAE loop may be separate WorkUnit kinds, but they remain tied to the same ModelSpec and its cache contracts.

### Cosmos3 Omni Reasoning Plus Video

```text
Session receives text and image
  -> Program selected: omni_reason_then_generate
  -> same ModelInstance
      -> ar_decode loop produces reasoner text
      -> action_condition loop packs action state if needed
      -> diffusion_denoise loop generates video latents
      -> audio_token_decode loop generates sound if requested
      -> vae_tile_decode loop produces pixels
  -> BehaviorRecord optional for RL or evaluation
  -> media chunks stream to client
```

This is the architecture's forcing case. A stage-only design would likely split this into reasoner, denoiser, audio, and decoder services before it has represented the shared model. v2 represents the model first.

### Disaggregated Encoder, Denoiser, Decoder

```text
Request
  -> Program selected: t2v_disaggregated
  -> encoder role pool
      -> encoder_chunk WorkUnits
      -> publish encoder cache
  -> denoiser role pool
      -> transfer manifest consumes encoder cache
      -> diffusion_step WorkUnits
      -> publish latent artifact
  -> decoder role pool
      -> vae_tile WorkUnits
      -> stream media chunks
```

This is where Dynamo, vLLM-Omni, SGLang-Omni, and SGLang multimodal generation become valuable. Disaggregation is a deployment form of typed program execution, not the foundation.

### RL Rollout

```text
Prompt batch
  -> TrainingProgram selected
  -> rollout role executes same generation loops as serving
  -> BehaviorRecords capture trajectory, logprobs, seeds, timesteps
  -> reward role scores artifacts or latents
  -> trainer recomputes or validates logprobs
  -> optimizer updates policy
  -> weight sync publishes new policy version
  -> rollout role invalidates incompatible caches
```

The system should be able to prove which policy version produced each sample.

## Validation Strategy

FastVideo v2 should treat validation as architecture, not afterthought.

### Static Validation

Run before deployment:

- model spec references valid components;
- loop specs reference valid state schemas;
- programs reference valid model capabilities;
- cache policies reference valid producers and consumers;
- parallel plans satisfy component and loop constraints;
- checkpoint manifests match expected keys and shapes;
- deployment role pools support requested programs;
- transport paths exist for disaggregated edges.

### Runtime Validation

Run during execution:

- resource reservations exist before admission;
- cache records match key schema;
- transfer manifests match expected tensor schemas;
- model weight versions match request or policy contract;
- cancellation occurs only at valid boundaries;
- behavior records include required fields for the selected training method.

### Test Suites

Required test categories:

- component parity tests;
- loop parity tests;
- checkpoint conversion tests;
- scheduler admission tests;
- WorkUnit batching tests;
- cache key and invalidation tests;
- transfer manifest tests;
- parallel plan validation tests;
- end-to-end artifact similarity tests;
- realtime session interruption tests;
- training logprob identity tests;
- weight sync lifecycle tests;
- workflow compiler tests.

GPU-heavy tests should be isolated, but the contracts they check should still be explicit in code.

## API Sketch

Python:

```python
engine = fastvideo.Engine.from_deployment("deployment.yaml")

video = engine.generate(
    model="cosmos3-video",
    program="text_to_video",
    prompt="a robot sorting tools on a workbench",
    duration=5,
    seed=7,
)

async with engine.session(model="cosmos3-omni") as session:
    await session.push_text("continue from this clip")
    await session.push_video(input_video)
    async for event in session.stream(program="realtime_video_continue"):
        ...
```

Training:

```python
trainer = fastvideo.Trainer.from_config("training.yaml")
trainer.run()
```

Workflow:

```python
program = fastvideo.workflow.compile("workflow.json")
engine.run(program, inputs)
```

Deployment:

```python
card = engine.deployment_card()
router.register(card)
```

## Proposed Package Layout

The architecture should be reflected in package boundaries. A possible clean-slate layout:

```text
fastvideo/
  api/
    http/
    python/
    cli/
    openai_compat/

  model/
    specs/
    components/
    loops/
    checkpoints/
    capabilities/
    parity/

  program/
    specs/
    compiler/
    workflows/
    validation/

  runtime/
    engine/
    scheduler/
    workers/
    admission/
    events/

  cache/
    keys/
    managers/
    policies/
    storage/

  transport/
    manifests/
    backends/
    relay/

  parallel/
    plans/
    meshes/
    process_groups/

  training/
    rollout/
    rewards/
    behavior/
    weight_sync/
    methods/

  deployment/
    cards/
    role_pools/
    routers/
    dynamo_adapter/

  sessions/
    state/
    streams/
    media/
    cancellation/

  artifacts/
    media/
    tensors/
    metadata/

  integrations/
    comfyui/
    dreamverse/
    livekit/
    diffusers/
```

Rules for the layout:

- `model/` must not import product APIs.
- `runtime/` may execute `model/` loops but should not define model semantics.
- `program/` may compose loops but should not own component implementations.
- `deployment/` may place and route work but should not inspect raw model internals beyond declared specs.
- `training/` may require behavior records but should not fork serving loop logic.
- `integrations/` should adapt external systems into core specs and events, not bypass them.

## Implementation Plan

This is a rebuild plan, not a minimal migration.

### Phase 0: Specs and Contracts

Deliver:

- typed `ModelSpec`, `ComponentSpec`, `LoopSpec`, `ProgramSpec`;
- loop inversion contract tests for `init_state`, `plan_step`, `run_step`, and `finalize`;
- `CapabilityMatrix`;
- `ParallelPlan`;
- `CacheKey`;
- `BehaviorRecord`;
- static validators;
- one model family expressed completely in v2 specs.

Exit criteria:

- a model can be loaded from spec;
- components and loops validate;
- checkpoint manifest validates;
- basic component parity tests pass.

### Phase 1: Single-Process Model-Native Runtime

Deliver:

- `ModelInstance`;
- `LoopKernel` interface;
- simple WorkUnit scheduler;
- local cache manager;
- artifact writer;
- streaming event bus;
- text-to-video and image-to-video programs for one model.

Exit criteria:

- generation works without disaggregation;
- WorkUnits are visible in traces;
- cache keys are structured;
- cancellation works at loop boundaries.

### Phase 2: General Scheduler and Memory Integration

Deliver:

- admission controller;
- memory reservation;
- typed cache classes;
- graph-capture cache;
- batching by WorkUnit compatibility;
- preemption at loop boundaries;
- scheduler tests.

Exit criteria:

- no admitted work can fail because required cache or memory was not reserved;
- scheduler supports at least diffusion, encoder, VAE, and AR WorkUnit kinds.

### Phase 3: Parallelism and Distributed Workers

Deliver:

- named mesh construction;
- TP, SP, CP, CFGP, VAE, and DP plan validation;
- distributed worker API;
- transfer manifests;
- local SHM/CUDA IPC transport;
- NCCL or UCXX transport path.

Exit criteria:

- the same ProgramSpec can run single-process or distributed;
- cache keys include parallel layout;
- unsupported parallel combinations fail at load time.

### Phase 4: Disaggregated Deployment

Deliver:

- role pools;
- stage shells as deployment wrappers;
- pool dispatcher;
- capacity and SLO-aware routing;
- deployment card export;
- Dynamo-compatible integration surface.

Exit criteria:

- encoder/denoiser/decoder role pools can run one model program;
- transfer and cache events are externally visible;
- global routers can make informed placement decisions.

### Phase 5: Training and RL

Deliver:

- rollout programs using serving LoopSpecs;
- BehaviorRecord capture;
- reward role integration;
- logprob recompute;
- policy versioning;
- weight sync lifecycle;
- consistency tests.

Exit criteria:

- a rollout sample can be traced to exact policy version, scheduler trajectory, seeds, and artifacts;
- logprob identity tests pass for supported models.

### Phase 6: Workflow and Realtime Products

Deliver:

- ComfyUI workflow compiler for known nodes;
- external node adapter;
- Dreamverse-style session runtime;
- WebSocket media events;
- WebRTC adapter where useful;
- interruption and cancellation semantics;
- prompt memory.

Exit criteria:

- workflows compile into ProgramSpecs where possible;
- realtime sessions can push input and interrupt output;
- media chunks are typed and ordered.

## Rejected Designs

### Pure stage DAG

A stage DAG is useful for deployment but insufficient as the core abstraction. It fails when one model instance owns shared weights and state across several loop types, as in Cosmos3-style omni workloads.

### Pure ComfyUI clone

ComfyUI is excellent for interactive workflow authoring, but dynamic node execution is too loose for high-performance serving, RL rollout, cache correctness, distributed scheduling, and typed parity.

### Copy vLLM token scheduling directly

vLLM's scheduler is the strongest reference for continuous batching, cache-aware admission, and running-first policy. But FastVideo must schedule diffusion steps, VAE tiles, encoder chunks, audio chunks, reward batches, transfers, and artifact writes. Tokens are one WorkUnit kind, not the whole scheduler.

### Hidden internal loops

Model code should not hide long-running denoise, decode, tile, or rollout loops behind one stage call. Hidden loops prevent admission control, memory reservation, cancellation, cache tracing, stream emission, and behavior capture. The runtime owns loop lifecycle; model code owns typed state transitions and step execution.

### Dynamo as the core runtime

Dynamo should orchestrate engines. It should not define FastVideo's model semantics, loop kernels, or local scheduler. FastVideo should integrate with Dynamo by exporting deployment cards, role pools, cache events, and cost models.

### Training as a separate stack

Keeping serving and RL training on separate model paths causes silent drift. Training may have separate roles and optimizers, but rollout and logprob behavior must reference the same ModelSpec and LoopSpec contracts.

### Opaque request dicts

Large mutable request objects with arbitrary extra fields are fast to prototype and slow to stabilize. They make it hard to validate cache correctness, transfer schemas, training behavior, and parallel layout.

## Open Questions

These are design questions to settle during Phase 0, not blockers to the architecture.

1. Should specs be Pydantic models, dataclasses with custom validation, or a hybrid?
2. What is the minimum stable HTTP API surface: OpenAI-compatible only, FastVideo-native only, or both?
3. Which model should be the first v2 forcing function: Cosmos3 omni, Wan-style T2V/I2V, or a smaller model for iteration speed?
4. Which transfer backend should be first after in-process: CUDA IPC, UCXX, or NIXL?
5. How much of workflow compilation should be in core versus a plugin package?
6. Which cache classes are required for v2 launch versus later optimization?
7. What consistency level should be mandatory for RL methods in the first release?
8. Should Dynamo integration be a built-in deployment backend or an external adapter package?

## Reference Inventory

The local reference pass covered these sources and shaped the design as follows.

### Cosmos3 and FastVideo Cosmos3 Port

Relevant paths:

- `~/FastVideo_cosmos3_port/tests/local_tests/cosmos3/PORT_STATUS.md`
- `~/cosmos-framework/cosmos_framework/model/vfm/`

Design impact:

- The Model Plane is mandatory because Cosmos3-style models combine reasoning, diffusion, action, and sound around shared components.
- The scheduler must understand both autoregressive and diffusion loops.
- Parity must include component parity, scheduler parity, denoise trajectory parity, reasoning logits, sound path behavior, and action path behavior.
- The system needs packed multimodal sequence semantics and model-owned memory state.

### SGLang Multimodal Generation

Relevant paths:

- `~/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py`
- `~/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py`
- `~/sglang/python/sglang/multimodal_gen/runtime/disaggregation/`
- `~/sglang/python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`

Design impact:

- Adopt role pools, request lifecycle state, capacity-aware dispatch, transfer manifests, and cache integration hooks.
- Reject the large mutable request object as a stable core interface.
- Generalize beyond vllm-omni's default (`RequestScheduler`, `max_num_running_reqs=1`); its opt-in homogeneous `StepScheduler` is the narrow precedent we make universal.

### vLLM Core

Relevant paths:

- `~/vllm/vllm/v1/core/sched/scheduler.py`
- `~/vllm/vllm/v1/core/kv_cache_manager.py`
- `~/vllm/vllm/v1/core/encoder_cache_manager.py`
- `~/vllm/vllm/v1/worker/gpu/model_states/`
- `~/vllm/vllm/v1/worker/gpu/input_batch.py`
- `~/vllm/vllm/v1/distributed/kv_transfer/`
- `~/vllm/vllm/v1/worker/gpu/cudagraph_dispatcher.py`
- `~/vllm/vllm/device_allocator/cumem.py`

Design impact:

- Adopt running-first scheduling, explicit budgets, cache manager ownership, persistent model/input state, graph capture keys, and tagged memory pools.
- Generalize tokens and encoder inputs into WorkUnits.

### vLLM-Omni

Relevant paths:

- `~/vllm-omni/vllm_omni/config/stage_config.py`
- `~/vllm-omni/vllm_omni/engine/orchestrator.py`
- `~/vllm-omni/vllm_omni/engine/stage_pool.py`
- `~/vllm-omni/vllm_omni/diffusion/`
- `~/vllm-omni/vllm_omni/distributed/omni_connectors/`

Design impact:

- Adopt stage pools, affinity, connector abstraction (`OmniConnectorBase` put/get over Mooncake/Mori/Yuanrong RDMA — note: no NIXL/NCCL named connector exists here), background orchestration, the frozen-`PipelineConfig`/editable-`DeployConfig` split (`merge_pipeline_deploy`), and stage-level deployment.
- Keep stages below the Model Plane.
- The honest reframe: vllm-omni *proves* these workloads are expressible. It runs true shared-weight MoT (bagel/lance), has an opt-in step scheduler, an opt-in homogeneous cross-request diffusion batcher, and a TP/CFG-aware one-directional cross-stage KV copy (`OmniKVTransferManager`, not a shared live cache). But there is no cost model (cannot co-schedule an AR token vs a denoise step — only per-stage count budgets), routing is hardwired `src+1` with every shipped pipeline a linear chain, and the MoT loop, step scheduling, and batching are opaque/opt-in/homogeneous/off-by-default. v2 makes them runtime-visible, step-scheduled, batchable, and cost-priced where vllm-omni leaves them opaque, opt-in, homogeneous, and uncosted.

### SGLang-Omni

Relevant paths:

- `~/sglang-omni/sglang_omni/config/schema.py`
- `~/sglang-omni/sglang_omni/pipeline/coordinator.py`
- `~/sglang-omni/sglang_omni/pipeline/stage/runtime.py`
- `~/sglang-omni/sglang_omni/pipeline/stage/stream_queue.py`
- `~/sglang-omni/sglang_omni/pipeline/relay_io.py`
- `~/sglang-omni/sglang_omni/relay/base.py`
- `~/sglang-omni/sglang_omni/scheduling/`

Design impact:

- Adopt schema vocabulary for `next`, `wait_for`, merge, streams, relay, and backpressure.
- Preserve the IO-shell idea around compute stages.
- Tie stage execution to typed model loops rather than opaque scheduler factories.

### Dynamo

Relevant paths:

- `~/FastVideo/dynamo/README.md`
- `~/FastVideo/dynamo/components/src/dynamo/global_router/handler.py`
- `~/FastVideo/dynamo/docs/features/disaggregated-serving/`
- `~/FastVideo/dynamo/docs/features/multimodal/`

Design impact:

- Treat Dynamo as fleet orchestration above inference engines.
- Export deployment cards, cache events, transfer endpoints, role pools, and cost/health models from FastVideo.
- Borrow disaggregated serving, KV-aware routing, KVBM, ModelExpress, planner, and multimodal E/P/D ideas.

### ComfyUI

Relevant paths:

- `~/ComfyUI/execution.py`
- `~/ComfyUI/comfy_execution/caching.py`
- `~/ComfyUI/comfy/model_management.py`

Design impact:

- Adopt workflow authoring, lazy node evaluation, node cache signatures, event streams, and memory pressure handling.
- Compile workflows into ProgramSpecs rather than using dynamic node execution as the core runtime.

### diffusers

Relevant paths:

- `~/diffusers/src/diffusers/modular_pipelines/modular_pipeline.py`
- `~/diffusers/src/diffusers/modular_pipelines/modular_pipeline_utils.py`
- `~/diffusers/src/diffusers/components_manager.py`
- `~/diffusers/src/diffusers/hooks/`

Design impact:

- Adopt component specs, block state, pipeline state, expected components/configs, load IDs, and offload strategies.
- Make the FastVideo version stricter and distributed-runtime aware.

### xDiT

Relevant paths:

- `~/xDiT/README.md`
- `~/xDiT/xfuser/config/config.py`
- `~/xDiT/xfuser/core/distributed/parallel_state.py`

Design impact:

- Adopt a first-class parallel plan covering tensor, sequence, ring, Ulysses, PipeFusion, CFG, data, and VAE parallelism.
- Validate parallel products and unsupported combinations before execution.

### verl-omni

Relevant paths:

- `~/verl-omni/README.md`
- `~/verl-omni/verl/trainer/diffusion/`
- `~/verl-omni/verl/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`

Design impact:

- Capture diffusion rollout data during execution: timesteps, latents, logprobs, SDE windows, and CFG details.
- Require algorithm contracts to declare their needed behavior fields.

### miles

Relevant paths:

- `~/miles/miles/true_on_policy/contracts.py`
- `~/miles/miles/true_on_policy/schema.py`
- `~/miles/miles/rollout/sglang_rollout.py`
- `~/miles/miles/backends/megatron_utils/update_weight/`

Design impact:

- Adopt true-on-policy contracts, deterministic flags, batch-invariant modes, session-routed rollout, and explicit weight update lifecycle.

### cosmos-rl

Relevant paths:

- `~/cosmos-rl/README.md`
- `~/cosmos-rl/cosmos_rl/dispatcher/`
- `~/cosmos-rl/cosmos_rl/rollout/`
- `~/cosmos-rl/cosmos_rl/utils/parallelism_map.py`
- `~/cosmos-rl/cosmos_rl/utils/payload_transport/`

Design impact:

- Adopt policy/rollout/reference roles, single-controller coordination, weight-version routing, dynamic role registration, and payload transport registry ideas.

### TorchTitan

Relevant paths:

- `~/torchtitan/torchtitan/distributed/parallel_dims.py`
- `~/torchtitan/torchtitan/protocols/model_spec.py`
- `~/torchtitan/torchtitan/experiments/rl/`

Design impact:

- Adopt named mesh axes, validated parallel dimensions, model-spec discipline, direct trainer-generator sync, and logprob identity validation.

### Dreamverse

Relevant paths:

- `~/Dreamverse/arch.md`
- `~/Dreamverse/design.md`
- `~/Dreamverse/server/gpu_pool.py`
- `~/Dreamverse/server/session/controller.py`
- `~/Dreamverse/server/av_streaming.py`
- `~/Dreamverse/server/worker_ipc.py`

Design impact:

- Make sessions, prompt memory, typed media chunks, shared buffers, cancellation, GPU assignment, and model reload first-class product-runtime concepts.

### LiveKit Agents

Relevant paths:

- `~/agents/livekit-agents/livekit/agents/voice/agent_session.py`
- `~/agents/livekit-agents/livekit/agents/llm/realtime.py`
- `~/agents/livekit-agents/livekit/agents/voice/speech_handle.py`

Design impact:

- Adopt realtime session structure, push audio/video streams, interruptions, turn/activity state, and event instrumentation for interactive products.

## Final Position

FastVideo v2 should be built around the following invariant:

```text
Model instance owns components and loop semantics.
Programs compose loops.
Schedulers execute WorkUnits.
Deployment pools place and route them.
Training records their behavior.
Products stream their artifacts.
```

This architecture combines the strongest pieces of the reference repos while keeping FastVideo's center of gravity where it belongs: model-native video and omni execution.
