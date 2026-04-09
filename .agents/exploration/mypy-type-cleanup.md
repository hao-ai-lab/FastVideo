# Exploration Log: Mypy Type Cleanup

## Status: draft

## Context
This repository did not have a dedicated skill or SOP for large-scale mypy cleanup across the dynamic pipeline/config infrastructure. The task was to reduce a large batch of mypy failures by fixing shared typing bottlenecks first instead of patching individual call sites one by one.

## Progress
- [x] Read onboarding and codebase map before editing.
- [x] Identified common failure clusters: dynamic config delegation, runtime-registered pipeline/stage attributes, batch state bags, attention metadata builders, executor streaming interfaces, and a few concrete backend/config mismatches.
- [x] Applied shared typing fixes to core config, pipeline, batch, attention, executor, and platform abstractions.
- [x] Patched several concrete hotspots: flash-attn import redefinition, HCCL send/recv typing, Cosmos 2.5 tuple config literals, workflow/preprocess typing, and a few training callback/entrypoint issues.
- [x] Ran `python -m compileall fastvideo ui` after edits to confirm syntax/import validity.
- [ ] Re-run `mypy` in an environment where the tool and dependencies are available offline.

## Findings
The bulk of the mypy failures came from a small number of shared abstractions being typed more narrowly than the codebase actually uses them. FastVideo relies heavily on:

- delegated config fields via `arch_config`
- runtime-added stage attributes via `add_stage(...)`
- dynamic per-pipeline fields stored on `ForwardBatch` / `TrainingBatch`
- backend-specific attention metadata subclasses passed through generic interfaces
- executor subclasses exposing extra streaming APIs beyond the base executor

Making those patterns explicit to mypy removes a large amount of noise and makes the remaining failures more likely to be true local mismatches.

## Mistakes / Dead Ends
- `mypy` was not installed in the shell environment.
- `uv run mypy ...` could not complete in the sandbox because dependency resolution required network access and the environment could not reach PyPI.

## Proposed Standardization
Create a repo workflow or skill for typing maintenance that includes:

- an approved offline or cached `mypy` invocation
- guidance on when to use precise annotations vs. dynamic escape hatches (`Any`, `__getattr__`)
- a checklist of shared abstractions to inspect first when many files fail at once
