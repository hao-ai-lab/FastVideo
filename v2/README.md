# FastVideo v2 - Inference Runtime Scope

**Status:** source of truth for `v2/`.

`v2` is the model-native inference runtime for FastVideo. It owns model cards,
programs, loops, runtime execution, serving, cache/memory policy, backend dispatch,
compile/cudagraph integration, and inference parity checks.

`v2` does **not** own training, finetuning, distillation, RL, optimizer steps, or
checkpoint production. Training remains in the existing FastVideo stacks:

- `fastvideo/train/` - the new modular trainer.
- `fastvideo/training/` - the legacy shipped training pipelines.

`v2` may record how a checkpoint was produced through `RecipeSpec` metadata
(`method`, `parents`, `assumes_loop`, `assumes_precision`), because inference must
know which runtime loop and precision policy a post-training checkpoint expects.
That metadata is provenance, not a v2 training API.

## Design Center

FastVideo v2 is video-generation first. Wan/LTX-style diffusion video inference
is the baseline path, and unified models such as BAGEL/Cosmos3 are first-class:
one resident model may run AR, diffusion, VAE, and codec loops in one request.
Audio and TTS are supported as additional modalities on the same stage/loop
model, not as the reason to build a separate universal serving framework.

The core abstraction should stay small:

- `ModelCard` declares the resident components, loops, capabilities, precision,
  caches, and sampling defaults a checkpoint needs.
- `Program` is an ordered list of typed nodes passing values through named
  slots. It is not a general DAG, Walk graph, or declarative control-flow IR.
- `Loop` owns model semantics. The runtime drives the loop, handles admission,
  cancellation, streaming, cache access, and backend dispatch.

Do not add a public contract field until a runtime path consumes it. Future
optimizations such as richer stage placement, multi-GPU transport, or paged KV
should start behind a concrete Wan/BAGEL/Cosmos/Qwen use case and graduate only
after they simplify at least two model recipes.

## Scope

In scope:

- Python inference entrypoint through `v2.VideoGenerator`.
- Typed `ModelCard` declarations for components, loops, capabilities, parity,
  sampling defaults, precision, and checkpoint layout.
- Driven inference loops such as diffusion denoise, AR decode, causal/world
  continuation, VAE/audio decode, and multi-stage programs.
- Runtime execution through `Engine` and `AsyncEngine`.
- Serving through OpenAI-compatible HTTP/SSE surfaces and deployment cards.
- Backend dispatch through the CPU toy backend, accelerator stand-ins, and the
  real torch/CUDA backend.
- Inference acceleration features such as FP8/NVFP4 loading, Sage/Flash/SDPA
  attention backend selection, `torch.compile`, cudagraph capture, cache policy,
  and component placement.
- Inference parity and regression tests.

Out of scope:

- Training methods, optimizers, loss functions, RL rewards, rollout trainers,
  weight-sync training loops, and behavior records for policy updates.
- Training examples under `v2_examples/`.
- Any CLI/API that advertises v2 as a trainer.
- A universal graph runtime, Walk/state-machine authoring layer, or parallelism
  vocabulary that is not consumed by the current inference runtime.

## Core Model

The atomic inference artifact is a `(recipe, runtime)` pair:

- `RecipeSpec` records what the weights assume: parent checkpoints, post-training
  method name, required loop, and required precision.
- `ModelCard` declares the runtime surface: components, loops, capabilities,
  caches, precision, parallelism, sampling defaults, and checkpoint manifest.
- `Program` composes component nodes and loop nodes into a user-facing task as
  an ordered named-slot stage list.
- `ModelInstance` is the resident loaded card with shared components, caches,
  weight versions, and optional captured graphs.

This keeps post-training artifacts serveable without making `v2` responsible for
creating them.

## Execution Model

Loops are model-owned state machines:

```python
state = loop.init(req, model, ctx)
while True:
    plan = loop.next(state)
    if isinstance(plan, Done):
        break
    result = ctx.execute(plan)
    state = loop.advance(state, result)
return loop.finalize(state)
```

The loop owns semantics. The runtime owns execution, admission, cancellation,
streaming, cache access, graph capture, and backend dispatch.

Serving is pooled run-to-completion. `AsyncEngine` bounds concurrency by pool
slots; each request runs its program to completion. The synchronous `Engine` is
the offline path used by tests and `VideoGenerator`.

## Package Layout

```text
v2/
  video_generator.py       public inference facade
  registry.py              model id -> card builder registry
  core/
    card/                  ModelCard, specs, ModelInstance
    loop/                  loop contracts, driver, sampler, policies
    program/               task programs and workflows
    request/               request params, tasks, outputs, sessions
    parity/                inference parity helpers
    parallel/              named parallel plans
  recipes/                 model-specific cards, loops, and programs
  runtime/                 Engine, AsyncEngine, cache, memory, cudagraph, transport
  serving/                 HTTP/SSE server and deployment adapters
  platform/                backend/device/kernel dispatch
  _vendor/                 vendored FastVideo model/loader/config pieces for inference
  tests/                   v2 inference/runtime/serving/parity tests
```

There is intentionally no `v2/training/` package.

## Current Inference Path

The torch backend builds real components from stamped checkpoint paths, keeps
components in eval mode, and dispatches inference through the same cards and loops
used by the CPU tests. Wan2.1 T2V inference is the primary real path today.
Wan/FastWan inference supports:

- real Wan component loading through vendored component loaders,
- FP8 post-load quantization for `FastVideo/FastWan-QAD-FP8-1.3B`,
- attention backend selection, including SageAttention when installed,
- `torch.compile` for inference DiT modules,
- on-device latent residency for cards that set `device_io=True`.

## Boundary Rule

If a change adds training behavior, it belongs in `fastvideo/train/` or
`fastvideo/training/`, not in `v2/`. If inference needs to consume the result of
that training, add or update a v2 card, loop, registry entry, checkpoint loader,
sampling defaults, and inference tests.
