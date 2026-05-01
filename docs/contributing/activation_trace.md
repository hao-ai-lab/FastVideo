# Activation Trace Mode

!!! note
    This page covers Extension 0 (module forward hooks), which is the implemented
    tracing mechanism. Extensions 1-3 are design sketches for future work and are
    **not yet implemented**.

## Overview

Activation trace mode is a zero-overhead-when-off, env-gated mechanism for
dumping per-layer activation statistics during FastVideo inference. Its primary
use case is **parity debugging across model ports**: enable tracing on both
FastVideo and the upstream reference implementation, then `diff` the resulting
JSONL files to find the first divergent layer.

The mechanism is intentionally narrow. It doesn't replace general logging,
profiling, or function tracing. It answers one question: "at which layer do
FastVideo and the reference model first produce different numbers?"

## When to use

- Investigating numerical drift between FastVideo and an upstream reference.
- Debugging mid-pipeline divergence (e.g., one block produces wrong output while earlier blocks match).
- Validating that a refactor preserves bf16 noise-floor behavior across many layers.

## When NOT to use

| Goal | Use instead |
|---|---|
| General logging | `init_logger(__name__)` |
| Per-stage timing | `FASTVIDEO_STAGE_LOGGING` |
| Profiling kernel timings | `FASTVIDEO_TORCH_PROFILER_DIR` (see [Profiling](profiling.md)) |
| Function-call tracing | `FASTVIDEO_TRACE_FUNCTION` (heavy) |

## Quickstart

```bash
FASTVIDEO_TRACE_ACTIVATIONS=1 \
FASTVIDEO_TRACE_LAYERS="^block\.layers\.[0-9]+$" \
FASTVIDEO_TRACE_STATS="abs_mean,sum,max,shape" \
FASTVIDEO_TRACE_OUTPUT="/tmp/fv_trace.jsonl" \
python examples/inference/basic/basic_magi_human.py
```

Each line in `/tmp/fv_trace.jsonl` is a JSON record:

```json
{"module": "block.layers.0", "tensor": "out", "step": 0, "abs_mean": 1.234, "sum": -5.678, "max": 9.012, "shape": [1, 4096, 5120]}
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `FASTVIDEO_TRACE_ACTIVATIONS` | `False` | Master toggle. When unset or false, **zero overhead** in the production hot path. |
| `FASTVIDEO_TRACE_LAYERS` | `""` (all) | Python regex filter applied to `model.named_modules()` names. Empty string matches all modules. |
| `FASTVIDEO_TRACE_STATS` | `"abs_mean,sum"` | Comma-separated stats to compute. Available: `abs_mean`, `sum`, `min`, `max`, `mean`, `std`, `shape`, `dtype`. |
| `FASTVIDEO_TRACE_OUTPUT` | `"/tmp/fv_trace_<pid>.jsonl"` | Output file path. `<pid>` is replaced with the process ID at runtime. |
| `FASTVIDEO_TRACE_STEPS` | `""` (all) | Comma-separated denoising step indices to capture. Empty string captures all steps. |

## Workflow: parity-debug a model port

1. Set up a tightly-controlled comparison: a parity test or a small standalone
   script that loads both the FastVideo model and the upstream reference with
   identical inputs and seeds.

2. Run the FastVideo side with tracing on:

   ```bash
   FASTVIDEO_TRACE_ACTIVATIONS=1 \
   FASTVIDEO_TRACE_LAYERS="<your regex>" \
   FASTVIDEO_TRACE_OUTPUT="/tmp/fv_trace_fv.jsonl" \
   python <fv_runner.py>
   ```

3. Run the upstream side. The upstream repo needs separate instrumentation. See
   "Hooking the upstream side" below.

4. Sort both files by `(module, step)` if needed, then diff:

   ```bash
   diff /tmp/fv_trace_fv.jsonl /tmp/fv_trace_upstream.jsonl
   ```

5. The first divergent line identifies the first layer where FastVideo and the
   upstream produce different outputs. Start debugging there.

## Architecture (Extension 0: module forward hooks)

At pipeline initialization, `attach_activation_trace()` reads the env vars once.
If `FASTVIDEO_TRACE_ACTIVATIONS` is unset or false, the function returns
immediately and no hooks are registered. If tracing is on, it walks
`model.named_modules()`, filters by the layer regex, and registers an
`ActivationStatHook` on each matching module.

During the forward pass, each hook fires after its module completes, computes
the requested stats on the output tensor, and appends a JSON record to the
output file.

```
ComposedPipelineBase
  └─ attach_activation_trace()
       ├─ reads env vars (once at startup)
       ├─ if off: returns None immediately
       └─ if on: walks named_modules()
            └─ registers ActivationStatHook on matching modules
                 └─ on each forward: compute stats → append JSONL
```

### Zero-overhead-when-off guarantee

- The env var check happens **once at startup** inside `attach_activation_trace()`.
- If the env var is unset or false, the function returns `None` immediately.
- No hooks are registered. No branches are added to the production forward path.
- The only cost when tracing is off is one env var lookup at pipeline
  initialization, which takes under a microsecond.

### Hooking the upstream side

The upstream reference repo isn't part of FastVideo, so it can't read FastVideo
env vars directly. Two options:

**Option 1: Inline patch** in your local clone of the upstream repo. Add
`register_forward_hook` calls in the same shape as `ActivationStatHook`. Clean
up afterward with `git stash` or `git checkout HEAD -- <file>`.

**Option 2: Wrapper script**. Write a small Python harness that imports the
upstream model, walks its `named_modules()`, and attaches hooks externally.
This is the same pattern used in
`tests/local_tests/transformers/_debug_magi_human_block_parity.py`.

The `add-model-trace` skill at `~/.config/opencode/skill/add-model-trace/`
provides a script template for this purpose.

## Future extensions (design only, not yet implemented)

### Extension 1: FX/Dynamo backend graph rewrite

**Granularity**: per-FX-node (every matmul, every add).

**Mechanism**: a `torch.compile` backend that takes the captured `GraphModule`
and inserts logger nodes after each op. Compiles into a separate artifact from
the production graph.

**Off semantics**: zero overhead. The production compile path is untouched.

**When to add**: if you need to trace inside a `torch.compile`'d graph and
Extension 0 is too coarse.

**Build cost**: roughly 1-2 days. Reference:
`torchao.quantization.pt2e._numeric_debugger`.

### Extension 2: AST source injection at import time

**Granularity**: per-line (between any two Python statements).

**Mechanism**: an importlib loader hook rewrites Python source AST at module
import time, inserting `if TRACE: dump(...)` statements. The decision is made
once at import.

**Off semantics**: zero overhead. If the env var is off at import time, source
is loaded as-is.

**When to add**: if you need per-line granularity that even FX-node-level can't
provide. This is almost never the right choice.

**Build cost**: roughly 1 week. Brittle and hard to debug.

### Extension 3: `__torch_dispatch__` / `TorchDispatchMode`

**Granularity**: per-op (every dispatcher call: matmul, add, view, etc.).

**Mechanism**: a `TorchDispatchMode` context manager that intercepts all ops at
the dispatcher level.

**Off semantics**: zero overhead. PyTorch's dispatcher only invokes mode hooks
when a mode is active.

**When on**: significant overhead. Every op pays a Python callback cost. Triton
kernels bypass it.

**When to add**: useful for quantization or dtype debugging where module-level
granularity isn't enough.

**Build cost**: roughly 1 day. Reference:
`torch.utils._python_dispatch.TorchDispatchMode`.

## Comparison with similar tools

| Tool | Pattern | FastVideo equivalent |
|---|---|---|
| SGLang `--debug-tensor-dump-output-folder` | env-gated forward hooks at startup | Extension 0 (this) |
| TransformerEngine `DumpTensors` | config-driven selective dumps | Extension 0 (env-driven) |
| HuggingFace `output_hidden_states=True` | source-level boolean gating | Not used; Extension 0 avoids model code edits |
| torchao numeric debugger | FX pass + node-level loggers | Extension 1 (future) |
| W&B `wandb.watch()` | runtime forward hooks (always on once registered) | Extension 0 has a similar mechanism, but gated off by default |

## Implementation references

- Module: `fastvideo/hooks/activation_trace.py`
- Env vars: `fastvideo/envs.py` (`FASTVIDEO_TRACE_ACTIVATIONS` and friends)
- Pipeline integration: `fastvideo/pipelines/composed_pipeline_base.py`
- Tests: `fastvideo/tests/hooks/test_activation_trace.py`
- Companion skill (for ad-hoc port investigations): `~/.config/opencode/skill/add-model-trace/`

## Changelog

| Date | Change |
|---|---|
| 2026-05-01 | Initial Extension 0 (module forward hooks) implementation. Extensions 1-3 designed but not implemented. |
