---
name: add-model-08-trace
description: Use during /add-model Phase 6 when component parity has failed and root cause requires layer-by-layer divergence analysis. Instruments both the official reference and FastVideo port with forward hooks to find the first numerical divergence point.
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

## Hard Rules: Instrumentation Hierarchy

Apply these in priority order. Use the highest-priority method that works for
the target site.

### (1) Forward hooks (PREFERRED)

`module.register_forward_hook(...)` and `register_forward_pre_hook(...)`.
Always within `try/finally` with `handle.remove()`. Zero source residue.

```python
handle = module.register_forward_hook(fn)
try:
    output = model(inputs)
finally:
    handle.remove()
```

### (2) Runtime monkey-patch (PREFERRED over source edits)

`module.attr = wrapped_func` or `cls.method = wrapped_method`, restored via
`try/finally` (save original first). Use for free functions and non-Module
sites such as activation functions (`swiglu`, `apply_rotary_emb`) that cannot
be hooked as `nn.Module` submodules.

```python
original = cls.method
cls.method = wrapped
try:
    output = model(inputs)
finally:
    cls.method = original
```

### (3) Source edits in FastVideo's own code

Only when (1) and (2) are insufficient. Track all edits within a single named
`git stash` boundary OR a temporary branch. Run `git diff` before closing the
investigation to confirm the stash or branch is clean. The cleanup gate
enforces this.

### (4) Source edits in official repo source

Allowed if EITHER:

- (a) The official repo is a git-tracked clone (e.g. `daVinci-MagiHuman/` at
  the repo root): use `git diff` in the clone path to verify cleanup.
- (b) It's installed editable (`pip install -e .`): use `git diff` in the
  editable source path to verify cleanup.

If the official repo is installed non-editable in site-packages: back up the
target file (`cp original.py original.py.trace-backup`) before editing, then
restore from backup at the end (or `pip install --force-reinstall <pkg>`).
The cleanup gate verifies via diff-against-backup or zero-diff-in-clone.

## Logging Contract

One log file per side. Paths:

```
/tmp/opencode/<family>_<component>_up_layers.log
/tmp/opencode/<family>_<component>_fv_layers.log
```

Format: one line per captured tensor, space-separated:

```
<name> <shape> <abs_mean> <sum> <min> <max>
```

Example:

```
block[00] (1,512,1024) 0.012345 6.3210 -0.4321 0.4321
```

Keep the format diff-friendly. Running `diff /tmp/opencode/x_up.log
/tmp/opencode/x_fv.log` should highlight the first divergent line directly.
Retain side-by-side stdout output alongside the per-side files for human
review.

## Drill-Down Loop

**Initial run:** attach hooks to every top-level block (`model.block.layers[i]`
or equivalent). Identify the first block index `NN` where abs_mean relative
drift exceeds 0.5% compared to the previous block.

**Drill run:** set `<FAMILY>_DEBUG_DRILL_LAYER=NN` and re-run. The script
attaches submodule hooks inside block `NN`: attention output, mlp.pre_norm,
mlp.up_gate_proj, mlp.down_proj input (via pre-hook) and output, mlp output,
attn_post_norm (if present), mlp_post_norm (if present).

**Iterate:** if the drill run points to a free function (e.g. an activation
not wrapped in an `nn.Module`), switch to a monkey-patch (method 2) to
intercept its output via the next module's pre-hook.

The loop ends when the first divergent submodule is identified with a
file:line citation in the official source.

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

- File paths to per-side logs (`/tmp/opencode/<family>_<component>_{up,fv}_layers.log`).
- The identified first divergent layer or submodule name.
- The upstream file:line citation where the divergence originates.
- Hypothesis verdict if an A/B toggle was used (e.g. "PATCH_LINEAR=1 closes
  the gap, confirming dtype-cast difference in PackedExpertLinear").
- Cleanup-gate status: `[cleanup-gate] PASS` or a list of unresolved items.

The calling agent uses this to scope the production fix in the FastVideo
component file.

## References

- `templates/block_trace_debug.py` in this skill directory: the canonical
  template this skill generalizes.
- `tests/local_tests/transformers/_debug_magi_human_block_parity.py` in the
  FastVideo3 repo: the worked magi-human example this skill was extracted from.
- `add-model/SKILL.md` Phase 6: the calling context for this skill.
- `add-model-03-port-dit/SKILL.md`, `add-model-04-port-vae/SKILL.md`,
  `add-model-05-port-encoder/SKILL.md`, `add-model-06-port-generic/SKILL.md`:
  bucket-specific debug language and component-specific escape-hatch knowledge.

## Changelog

| Date | Change |
|---|---|
| 2026-05-01 | Initial skill extracted from `_debug_magi_human_block_parity.py` pattern. |
