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

## Performance impact

When the master toggle is off, overhead is nil. When on, the cost is
proportional to how broadly the layer regex matches.

### What runs when tracing is on

For every match against `model.named_modules()`, FastVideo registers an
`ActivationStatHook` that runs after the module's forward returns:

1. Walks the (possibly nested) output `tuple` / `list` / `dict` and extracts
   every `torch.Tensor` leaf.
2. Computes each enabled stat on the tensor's `.detach().float()` view — the
   cast is required for bf16 inputs because some reductions are not stable
   in bf16.
3. Writes one JSON record per tensor to the line-buffered JSONL sink.

Cost is roughly `O(num_matched_modules × num_output_tensors × num_stats × tensor_numel)`
per forward, dominated by `tensor_numel` for value-stats (`abs_mean`, `sum`,
`min`, `max`, `mean`, `std`). The `shape` and `dtype` stats are O(1).

### Cost-shaping knobs

The default config (empty `FASTVIDEO_TRACE_LAYERS` is treated as `.*`, default
two stats, all steps) is intentionally blunt — useful only for a one-shot
smoke run. For real debugging, scope down:

| Knob | Effect |
|---|---|
| Tighten `FASTVIDEO_TRACE_LAYERS` to a regex matching <50 modules | Linear reduction in hook count |
| Drop unused stats from `FASTVIDEO_TRACE_STATS` (`mean`, `std`, `min`, `max`) if you only need divergence detection | One full reduction per stat saved per matched module per forward |
| Set `FASTVIDEO_TRACE_STEPS="0,15,31"` for a 32-step run | ~10x reduction vs all-steps (the hook still fires but exits early when the step doesn't match) |
| Use `shape` + `dtype` only on layers where you only care about layout | Skips tensor reductions entirely on those layers |

### Disk

Output is line-buffered (`open(..., buffering=1)`), so every record flushes
on write. A typical 32-step run that traces 40 DiT blocks with 4 stats
writes roughly 5K records — about 1 MB of JSONL. Point
`FASTVIDEO_TRACE_OUTPUT` at a fast local disk for parity runs; slow network
mounts will dominate runtime once tracing is on.

## Troubleshooting

### "I set the env var but no JSONL file appears"

Three things to check, in order:

1. **Toggle semantics.** `FASTVIDEO_TRACE_ACTIVATIONS` uses a strict
   not-equal-to-`"0"` test. Setting it to `1`, `true`, or even `""` all
   enable tracing. Only an unset variable or `FASTVIDEO_TRACE_ACTIVATIONS=0`
   disables it.
2. **Module exposure.** The hook attaches to `pipeline.modules.get("transformer")`
   at the end of `post_init`. If your pipeline does not expose a module
   under that key (e.g. a non-standard custom pipeline whose DiT is
   reachable only via `pipeline.modules["sr_transformer"]`), the trace is
   silently a no-op. Check the
   `Activation trace attached to N modules` log line at startup; if
   `N=0`, either the regex didn't match anything or the expected module
   isn't exposed.
3. **Output path failure.** The parent of `FASTVIDEO_TRACE_OUTPUT` is
   auto-created. If creation fails (permissions, read-only mount),
   `JsonlSink.__init__` raises at startup — look for an `OSError` early
   in the log.

### "My regex isn't filtering the way I expect"

`FASTVIDEO_TRACE_LAYERS` is compiled with `re.compile(spec)` and matched
with `pattern.search(name)`. Two consequences:

- `search`, not `fullmatch`. `block.layers` matches
  `transformer.block.layers.0.attn`. Anchor with `^...$` if you want
  exact matches.
- Module names use Python dot notation (`transformer.block.layers.0`),
  not slashes. The `.` in your regex is a metacharacter — escape it as
  `\.` if you want a literal dot.

Run once with `FASTVIDEO_LOGGING_LEVEL=DEBUG` to see the
`Activation trace attached to N modules (pattern=...)` line. `N` is the
ground truth for how many modules survived your regex.

### "Stats are NaN or `<error: ...>` for some layers"

Some outputs hit numerical edge cases:

- `mean` / `std` on a 0-dim tensor returns NaN.
- `abs_mean` on an empty tensor or one full of `inf` returns NaN.
- Non-tensor outputs (e.g. a Python `bool` from a verification gate) are
  silently skipped — the hook only walks `torch.Tensor` leaves.

Dropping `mean`/`std` and using `abs_mean`/`max` is more robust. For
layout-only debugging, `shape`+`dtype` never fail.

### "Tracing slows the run by 5x"

You're probably matching too broadly. An empty `FASTVIDEO_TRACE_LAYERS` is
treated as `.*` and matches every named module — for a 15B-param DiT
that's hundreds of submodules, each running stat reductions on every
forward. Tighten to a single block depth: `^transformer\.blocks\.\d+$`
typically matches a few dozen modules, which is a manageable trace.

### "Trace records are missing tensors I expect"

The hook walks `tuple` / `list` / `dict` outputs recursively but does
**not** unpack custom dataclasses or named tuples — those are silently
skipped. If your module returns
`BlockOutput(hidden_states=..., attn_logits=...)`, no records are
emitted. Workaround: either return a plain `dict` (`{"hidden_states": ..., "attn_logits": ...}`)
or attach the hook to a deeper module that already returns a raw tensor.

### "I want tracing inside a `torch.compile`'d region"

Module forward hooks run on the eager wrapper. If the entire module is
compiled, the hook sees only the wrapped op's output, not internal FX
nodes. This is by design — Extension 1 (FX backend rewrite) in
[Future extensions](#future-extensions-design-only-not-yet-implemented)
is the planned path when inside-graph granularity is required.

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

The `add-model-08-trace` skill at `.agents/skills/add-model-08-trace/`
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
- Companion skill (for ad-hoc port investigations): `.agents/skills/add-model-08-trace/`

## Changelog

| Date | Change |
|---|---|
| 2026-05-01 | Initial Extension 0 (module forward hooks) implementation. Extensions 1-3 designed but not implemented. |
