---
name: add-model-08-trace
description: Use during /add-model Phase 6 when component parity has failed and root cause requires layer-by-layer divergence analysis. Uses FastVideo activation trace first, falling back to custom hooks only for boundaries or stats the utility cannot observe.
---

# Add-Model Trace

## Manual Invocation

Load this skill when `/add-model` Phase 6 component parity has failed and the
root cause requires layer-by-layer divergence analysis. This skill is not
auto-fired. The calling subagent (DiT, VAE, encoder, or generic port skill)
loads it when its standard parity-debug loop hits a wall and cannot isolate
the divergence from end-to-end tensor comparisons alone.

Do not load this skill for first-pass parity failures. Try weight-diff and
end-to-end tensor comparison first. Load this skill only when those do not
isolate the cause.

## Goal

Find the first numerical divergence point between FastVideo's port and the
official reference, layer by layer, by instrumenting both sides at matching
tensor boundaries. The investigation must leave zero source residue in
production code when it closes.

## When To Run

After a component parity test FAILS at a bf16-noise-realistic tolerance AND
the calling subagent's first-pass debug (weight-diff, end-to-end tensor
compare) does not isolate the cause.

Required inputs before starting:

- A working FastVideo loader for the component under investigation.
- A working official loader, typically via
  `tests/local_tests/helpers/<family>_upstream.py::load_upstream_<component>`.
- Shared deterministic test inputs (same tensors on both sides).
- The component parity test file path and its current failure output.

## Primary Path: FastVideo Activation Trace

Use FastVideo's first-class activation trace before writing custom hooks:
`fastvideo/hooks/activation_trace.py`, documented in
`docs/contributing/activation_trace.md`.

Pipeline runs attach trace to the transformer during pipeline initialization.
Component-only parity harnesses may call `attach_activation_trace(model)` from
local test/debug code; do not add trace calls to production model code.

Prefix the failing parity command with a tight layer regex:

```bash
FASTVIDEO_TRACE_ACTIVATIONS=1 \
FASTVIDEO_TRACE_LAYERS="^block\.layers\.[0-9]+$" \
FASTVIDEO_TRACE_STATS="abs_mean,sum,max,shape" \
FASTVIDEO_TRACE_STEPS="0" \
FASTVIDEO_TRACE_OUTPUT="/tmp/opencode/fv_trace.jsonl" \
pytest tests/local_tests -k "parity" -v -s
```

Match the layer regex to the actual `model.named_modules()` names. Empty or
broad regexes are expensive; prefer block-level names first, then narrow to
submodules after the first divergent block is known.

## Trace Compare Contract

One JSONL file per side. FastVideo output should use `FASTVIDEO_TRACE_OUTPUT`;
the upstream harness should emit the same JSONL shape:

```json
{"module":"block.layers.0","tensor":"out","step":0,"abs_mean":0.0123,"sum":1.0,"max":0.5,"shape":[1,16,32]}
```

Compare rows by `(module, step, tensor)`. The first row whose `shape`,
`abs_mean`, or `max` diverges beyond the component tolerance is the first broken
boundary. Keep `FASTVIDEO_TRACE_LAYERS`, `FASTVIDEO_TRACE_STATS`, and
`FASTVIDEO_TRACE_STEPS` identical between sides; if row order differs, sort or
normalize before diffing.

## Drill-Down Loop

**Initial run:** trace every top-level block (`^block\.layers\.[0-9]+$` or the
family's equivalent). Identify the first block index where `abs_mean` or `max`
drifts beyond tolerance while earlier blocks match.

**Drill run:** tighten `FASTVIDEO_TRACE_LAYERS` to submodules inside the first
divergent block: attention output, MLP projections, norm outputs, modality
adapters, or other named boundaries exposed by `named_modules()`.

**Iterate:** if the first divergent operation is a free function or tensor op not
visible as an `nn.Module`, use the fallback instrumentation hierarchy below.

The loop ends when the first divergent submodule or operation is identified with
a file:line citation in the official source.

## Fallback Instrumentation Hierarchy

Use these only when activation trace cannot observe the needed boundary or
statistic.

### (1) Custom forward hooks

`module.register_forward_hook(...)` and `register_forward_pre_hook(...)`.
Always within `try/finally` with `handle.remove()`. Zero source residue.

### (2) Runtime monkey-patch

`module.attr = wrapped_func` or `cls.method = wrapped_method`, restored via
`try/finally` (save original first). Use for free functions and non-Module sites
such as activation functions (`swiglu`, `apply_rotary_emb`).

### (3) Source edits in FastVideo's own code

Only when (1) and (2) are insufficient. Track all edits within a single named
`git stash` boundary OR a temporary branch. Run `git diff` before closing the
investigation to confirm cleanup.

### (4) Source edits in official repo source

Allowed only when hook and monkey-patch approaches cannot capture the site.
For git-tracked or editable official clones, use `git diff` in the clone path to
verify cleanup. For non-editable site-packages, back up the target file before
editing and restore it before handoff.

## Hypothesis Toggles

Use env-var-gated monkey-patches to A/B test suspect implementations without
source edits. Pattern: `<FAMILY>_DEBUG_PATCH_<HYPOTHESIS>=1`.

Example from the magi-human investigation:

```
MAGI_DEBUG_PATCH_LINEAR=1
```

This patched `PackedExpertLinear.forward` to mirror upstream's
`_BF16ComputeLinear` explicit-cast pattern, isolating a dtype-cast difference
as the root cause.

Document all toggles in the script docstring. Each toggle must:

- save the original before patching;
- restore the original in a `try/finally` block;
- print a `[debug] Patched <ClassName>.<method>` line to stdout when active.

## Cleanup Gate

The calling agent MUST report `[cleanup-gate] PASS` on all five items before
handoff. Do not hand off with any item unresolved.

1. `git diff` in the FastVideo repo: empty. No stray prints, hooks, or
   monkey-patches in production code.
2. `git diff` in the official-repo clone (if used): empty. For non-editable
   site-packages installs: `diff original.py original.py.trace-backup` is
   empty OR `pip install --force-reinstall <pkg>` succeeded and the installed
   file matches the original.
3. `git stash list`: only the named investigation stash (or empty). No
   unnamed stashes left from this session.
4. No new untracked files outside `/tmp/opencode/` (logs) and the existing
   debug script directory (`tests/local_tests/transformers/` or equivalent).
5. `mypy` clean on any production files touched during the investigation.

## Escape Hatches

Escalate to the calling bucket skill when:

- A forward hook on an official module raises because of a custom `forward`
  signature or varlen handler args that the hook closure cannot satisfy. The
  bucket skill has component-specific knowledge to work around this.
- The first divergent layer is `block[0]`, meaning the divergence is in the
  adapter, modality dispatcher, coordinate embedding, or packing step before
  any block runs. Check those sites first; the bug is not in attention or MLP.
- Per-block drift is never zero anywhere across all blocks. This usually means
  the inputs are not bit-identical between sides. Verify with a state-dict
  compare (weight-diff script) AND confirm the input tensors are the same
  object or have identical values before the forward call.

## Handoff

Return to the calling subagent with:

- FastVideo trace JSONL path and upstream trace JSONL path.
- Trace settings used: `FASTVIDEO_TRACE_LAYERS`, `FASTVIDEO_TRACE_STATS`, and
  `FASTVIDEO_TRACE_STEPS`.
- The first divergent `(module, step, tensor)` row and observed drift.
- The upstream file:line citation where the divergence originates.
- Fallback hook/patch verdict if activation trace could not observe the boundary.
- Hypothesis verdict if an A/B toggle was used, for example `PATCH_LINEAR=1`.
- Cleanup-gate status: `[cleanup-gate] PASS` or a list of unresolved items.

The calling agent uses this to scope the production fix in the FastVideo
component file.

## References

- `docs/contributing/activation_trace.md` for canonical activation-trace env vars,
  JSONL output, cost model, and troubleshooting.
- `fastvideo/hooks/activation_trace.py` for the implementation and
  `attach_activation_trace(model)` entry point.
- `templates/block_trace_debug.py` in this skill directory: fallback custom-hook
  template when activation trace cannot observe the needed boundary or stat.
- `tests/local_tests/transformers/_debug_magi_human_block_parity.py` in the
  FastVideo3 repo: historical worked example for custom hook/patch debugging.
- `add-model/SKILL.md` Phase 6: the calling context for this skill.
- `add-model-03-port-dit/SKILL.md`, `add-model-04-port-vae/SKILL.md`,
  `add-model-05-port-encoder/SKILL.md`, `add-model-06-port-generic/SKILL.md`:
  bucket-specific debug language and component-specific escape-hatch knowledge.

## Changelog

| Date | Change |
|---|---|
| 2026-05-01 | Initial skill extracted from `_debug_magi_human_block_parity.py` pattern. |
