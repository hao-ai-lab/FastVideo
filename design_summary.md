# FastVideo — Design Philosophy

One page on *why* FastVideo is built the way it is. The full architecture, the as-built status, and the
forward roadmap live in **[`v2/README.md`](v2/README.md)** — this is the philosophy beneath it.

---

**A deployable model is a post-training artifact.** Unlike an LLM — where inference optimizes frozen weights
after the fact — a *usable* video/omni model is *created* by training: step distillation for latency, QAT for
precision, distillation + self-forcing for causal/world models. So every inference capability is a
**(recipe, runtime) pair**: the weights and the loop that produced-and-assumes them are one versioned object.
This is the source of the moat — whoever owns *both* sides of the pair owns the optimization frontier — and it
is why training and serving cannot be two systems.

**The work is loops, not `forward()`.** Denoise timesteps, AR decode, chunked rollout, VAE tiles, encoder
chunks, audio tokens, reward batches, optimizer steps, media chunks — video and omni inference is iteration. A
runtime that collapses everything to a single `forward` can't schedule, batch, cancel, stream, reserve memory
for, or capture the behavior of what actually runs. So loops are first-class, and they are **driven**: the
model describes the next step it needs, the runtime decides when and with whom it runs, the model folds the
result back. The model keeps content-adaptive control flow; the runtime keeps admission, batching, streaming,
and behavior capture. Per-request state lives in typed `LoopState`, never in module globals — so interleaving
requests through one model instance cannot smear state, by construction.

**The model is the center; everything else is a view over it.** A typed `ModelCard` owns components, loops,
the recipe, and the parity contract. Programs compose a card's loops into a task; Workflows compose cards into
pipelines; the scheduler runs the *steps* of all loops as `WorkUnit`s under one currency (predicted GPU-time,
because a bidirectional denoise step and an AR token are ~1000× apart and incommensurable in counts);
deployment places and routes; products stream artifacts. None of them define model semantics — they reference
the Model Plane. One resident instance can run many loop types on shared weights, which is what makes omni/MoT
native rather than a DAG that doubles weights.

**Correctness is a typed contract, not a hope.** Caches are correct by *key* — if a field can change output
semantics it is in the key, so reuse is partitioned, never blindly flushed. Parity between the train-forward
and the serve-forward is *measured* on a declared ladder (component → loop → behavioral → distribution →
artifact-quality), never assumed. And the non-negotiable gate is **interleave bit-parity**: N requests
interleaved at step granularity must be bit-identical to running them serially — the test the whole
loop-inversion bet lives or dies on.

**One substrate for inference, training, and RL.** The rollout forward *is* the serve forward plus capture —
same loop, same caches, same batcher, same numerics — so every serving optimization is automatically a rollout
optimization, and there is one numerics surface the ladder measures rather than a correction layer papering
over it. The engine doubles as the RL rollout engine under a strict rule: `training` consumes the engine; the
**engine never imports `training`**.

**Borrow aggressively; copy nothing as the core.** vLLM/SGLang scheduling, vLLM-Omni/SGLang-Omni omni serving,
Dynamo fleet orchestration, diffusers components, xDiT parallelism, TorchTitan mesh discipline,
verl-omni/miles RL lessons, ComfyUI workflows, Dreamverse/LiveKit sessions — each contributes a take, none is
the center. Deployment orchestration (Dynamo) sits *above* the engine, never inside it. Extensions are
versioned hook points, never monkeypatching. New frontier capabilities arrive as a card, a method, a loop, a
workflow, or a controller — **not a rewrite**.

> A model card is a (recipe, runtime) pair with a parity obligation. The model owns loop semantics; the runtime
> owns loop lifecycle. One resident instance runs many loops; one scheduler runs their steps in one currency.
> Caches are correct by key; parity is correct by test; the interleave gate is non-negotiable. Training records
> behavior on the same loops it serves. Deployment places and routes; products stream artifacts; neither defines
> the model.
