# Issue 1523 Handoff

## Current State
- Issue: #1523
- Title: `[ci][p1] Cache FastVideo kernel builds in Modal CI lanes`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1523
- Repository: hao-ai-lab/FastVideo
- Branch: issue/1523-fix
- Worktree: /tmp/fastvideo-worktrees/issue-1523-fix
- Current commit: 0a699ec836ea7cbfe5ec4173a800af6a8adef117
- Handoff path: .agents/handoffs/issue-1523-handoff.md
- Current stage: Stage 2 - Implement The User-Directed Fix
- Implementation begun: yes
- Last updated: 2026-07-06T07:05:36Z

## Resume Notes
- The first Stage 1 attempt created this branch/worktree and staged a handoff, but the user interrupted before the handoff was committed.
- `/tmp/fastvideo-worktrees` was wiped before resume, leaving a stale git worktree entry and no committed handoff.
- Removed only the stale `/tmp/fastvideo-worktrees/issue-1523-fix` worktree metadata, recreated the worktree for existing branch `issue/1523-fix`, fast-forwarded the clean branch from 6a32cf3a5 to current `origin/main`/`upstream/main` at 9d909f5f0, and recreated this handoff.
- No implementation changes have been made. The only intended Stage 1 file change is this handoff.

## Skill And Repository Instructions Read
- `/home/toolbox/.codex/skills/fix-issue/SKILL.md`
- `/home/toolbox/.codex/skills/fix-issue/references/handoff.md`
- `/home/toolbox/.codex/skills/fix-issue/references/stages.md`
- Root `AGENTS.md`
- `fastvideo/tests/AGENTS.md`
- `fastvideo/tests/ssim/AGENTS.md`
- Relevant lesson: `.agents/lessons/2026-05-22_dreamverse-ci-streaming-imports-need-gpu.md`

## Authentication And GitHub State
- `gh api user --jq .login` succeeded as `macthecadillac` after the sandboxed command initially hit a network error.
- Re-checked on resume: `gh api user --jq .login` -> `macthecadillac`.
- Issue state from `gh issue view 1523`:
  - State: OPEN
  - Author: Satyam-53
  - Created: 2026-07-01T02:24:02Z
  - Updated: 2026-07-01T02:24:10Z
  - Labels: enhancement, help wanted, installation, platform, type: ci, scope: inference, scope: attention, scope: kernel, scope: infra
  - Assignees: none
  - Comments: none
- Issue request:
  - Modal CI lanes still call `./build.sh` in `fastvideo/tests/modal/pr_test.py` and `fastvideo/tests/modal/ssim_test.py`.
  - This rebuilds `fastvideo-kernel` repeatedly even though Docker images prebuild it and `build.sh` can avoid torch CUDA probing when `TORCH_CUDA_ARCH_LIST` is set.
  - Desired behavior is a reusable Modal cache path for kernel builds, keyed conservatively by kernel source hash, Python version, torch version, CUDA version, and architecture list; cache hit installs/reuses the wheel, cache miss builds from source and stores the result.
- Open PR checks:
  - Ran `gh pr list -R hao-ai-lab/FastVideo --state open --limit 200 --json number,title,isDraft,url,closingIssuesReferences,headRefName,updatedAt`.
  - No open PR closes issue #1523.
  - Nearby related CI PRs exist but do not cover this issue: #1547 closes #1522, #1557 closes #1524, #1559 wires ops tests, #1560 closes #1532.
  - Targeted open-PR search for `1523 OR modal kernel cache OR fastvideo-kernel build.sh cache` returned no results.
  - Targeted duplicate issue searches for `modal kernel cache` and `fastvideo-kernel build cache` returned no results.
  - Stage 2 re-check on 2026-07-06T06:55:05Z: issue still open with no comments; no open PR closes or directly references #1523.
  - Inspected potentially related PRs:
    - #1389 `[ci] Modal: enable Flash Attention 3 on H100 perf tests with build cache` is a ready-for-review PR that adds an opt-in H100 perf entrypoint and FA3 runtime cache only. It does not cache `fastvideo-kernel` builds or touch SSIM.
    - #1449 `[feat]: add Hugging Face Kernel Hub packaging` closes #1318 and adds HF Kernel Hub packaging/workflow files; it is not Modal CI wheel caching for `pr_test.py`/`ssim_test.py`.
- There are no commenter-proposed fixes to evaluate because issue #1523 has no comments.

## Code Findings
- `fastvideo/tests/modal/pr_test.py`
  - `run_test()` calls `run_test_command(..., build_kernel=True)` by default.
  - `run_test_command()` performs dependency install before kernel build, explicitly because installing after a build can replace a just-built in-tree kernel with the pinned PyPI wheel.
  - If `build_kernel` is true, lines 117-121 still run:
    - `cd fastvideo-kernel`
    - `./build.sh`
    - `cd ..`
  - Most Modal CI lanes use `run_test()` and therefore rebuild the kernel. `run_dreamverse_app_tests()` intentionally passes `build_kernel=False`.
  - Some `pr_test.py` functions already mount `/root/data`; several build-kernel lanes do not, so a Modal-volume cache must either mount a cache volume for those lanes or fall back cleanly when no persistent cache is mounted.
- `fastvideo/tests/modal/ssim_test.py`
  - `_prepare_ssim_workspace()` installs `.[test]`, then runs `cd fastvideo-kernel && ./build.sh && cd ..` at lines 507-510.
  - SSIM partitions already mount `model_vol` at `/root/data`, so a persistent cache can be available there.
  - The SSIM orchestrator clones/prepares once per partition, then schedules subprocesses, so kernel-cache setup belongs in `_prepare_ssim_workspace()` before task scheduling.
- `fastvideo-kernel/build.sh`
  - Initializes only required submodules under `include/cutlass` and `include/tk`.
  - Installs build dependencies with `uv pip install scikit-build-core cmake ninja`.
  - If `TORCH_CUDA_ARCH_LIST` is set, it skips the torch CUDA probe and derives CMake arch/TK defaults from that env var.
  - Otherwise it probes `torch.cuda.get_device_capability(0)` and normalizes Hopper to `9.0a` and Blackwell sm_120 to `12.0a`.
  - It currently ends with `uv pip install . -v --no-build-isolation`; it does not expose a wheel-output mode for CI to cache a built wheel.
- `docker/Dockerfile`
  - Defaults `TORCH_CUDA_ARCH_LIST=9.0a`.
  - Builds `fastvideo-kernel` once during image creation with `TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ./build.sh`.
  - Then installs FastVideo editable with `--no-deps`, preserving the local kernel in the image.
- Root `pyproject.toml`
  - Runtime dependency pins `fastvideo-kernel==0.3.2`.
  - On x86_64, uv resolves to the published wheel; on aarch64, uv sources the in-tree `fastvideo-kernel` path.
  - This explains why Modal dependency install ordering matters and why skipping all build logic without a validity check risks using stale or wrong wheels.
- Current code confirms the issue is valid: Modal runners still rebuild from source unconditionally for most build-kernel lanes, and there is no shared cache helper/path.

## Merits And Scope
- The report appears valid and high-impact for CI latency: kernel builds are repeated per Modal lane/partition even when source and ABI inputs are unchanged.
- Correctness risk is stale binary reuse. The key must include source and ABI inputs, not just package version.
- Local developer builds should remain unchanged: default `fastvideo-kernel/build.sh` behavior should continue to build/install from source.
- The implementation should stay inside Modal CI/test infrastructure plus a directly used `build.sh` option if needed. Avoid broader packaging or kernel refactors.
- GPU memory impact should be negligible: this is CI setup/package installation behavior, not inference/training runtime behavior.

## Alternatives Considered
1. Minimal direct skip when a package imports successfully
   - Touches only `pr_test.py` and `ssim_test.py`.
   - Fastest to implement, but unsafe because a successful import does not prove source hash, CUDA arch, torch, or CUDA ABI match the checked-out code.
   - Not recommended.
2. Modal-volume wheel cache keyed by conservative inputs
   - Add a small shared script under `fastvideo/tests/modal/` that runs after the repo clone and dependency install.
   - Compute a cache key from the checked-out `fastvideo-kernel` tree hash, Python version, torch version, torch CUDA version, nvcc/CUDA version, platform machine, normalized `TORCH_CUDA_ARCH_LIST`, `CMAKE_ARGS`, and GPU backend.
   - On cache hit, install the cached wheel with `uv pip install <wheel>`.
   - On miss, build a wheel from the checked-out source, install it, atomically store it under the Modal volume, and log the miss/store path.
   - Use the same helper path from `pr_test.py` and `ssim_test.py`.
   - Recommended as the core fix.
3. Reuse Docker-prebuilt kernel only when a baked build-info key matches
   - Extend the Docker image build to write conservative kernel build metadata into a stable path such as `/opt/fastvideo-kernel-build-info.json`.
   - The Modal helper can compare the baked key to the current checked-out key and skip cache/build on an exact match.
   - If metadata is absent or mismatched, fall back to the Modal wheel cache.
   - Good robustness improvement, but current images may not have this metadata until rebuilt. Recommended only if it stays small and directly used by the helper.
4. Rely on uv or pip cache only
   - Low code change, but not enough because Modal jobs are separate containers and the issue asks for a reusable Modal cache path with explicit hit/miss logging and source/ABI keying.
   - Not recommended.

## Recommended Plan
1. Add a shared Modal kernel cache helper script, likely `fastvideo/tests/modal/kernel_build_cache.py`.
   - Make it runnable from inside the cloned repo rather than imported at Modal module import time, because remote Modal containers may initially mount only the entrypoint file.
   - Provide commands/modes along these lines:
     - compute and print cache metadata/key;
     - install cached wheel or build/cache/install on miss.
   - Use only standard library plus installed torch/subprocess so it works after `uv pip install -e ".[test]"`.
2. Teach `fastvideo-kernel/build.sh` a directly used wheel output mode, if necessary.
   - Keep default `./build.sh` unchanged.
   - Add a flag such as `--wheel-dir <dir>` that runs the existing setup/arch/CMAKE logic, builds a wheel with `uv build --wheel --no-build-isolation --out-dir <dir> .`, then installs the produced wheel.
   - The helper can use this to avoid duplicating build-script logic and avoid building twice.
3. Wire `pr_test.py` to call the helper instead of direct `./build.sh` when `build_kernel=True`.
   - Preserve install-before-kernel-build ordering.
   - Mount a persistent cache volume for all build-kernel lanes, or pass a cache root that falls back with a clear log if no cache volume is mounted.
   - Keep `run_dreamverse_app_tests(build_kernel=False)` unchanged.
4. Wire `ssim_test.py` to call the same helper path during `_prepare_ssim_workspace()`.
   - Reuse the same cache root/key logic.
   - Commit the Modal volume after a successful miss/store so later jobs can see the wheel.
5. Optionally add Docker build metadata if the helper can use it with a small direct check.
   - Only skip build/cache for the Docker-prebuilt package when its baked metadata exactly matches the current key.
   - If metadata is missing or mismatched, use the Modal cache path.
6. Keep logging explicit.
   - Logs should say `fastvideo-kernel cache hit`, `cache miss`, `building wheel`, `stored wheel`, `using Docker-prebuilt kernel`, or `cache unavailable; building from source`.
7. Do not change local developer build behavior.
   - Default `cd fastvideo-kernel && ./build.sh` remains a source build/install.
   - New helper is only called by Modal runners.

## Validation Plan
- Do not run local project tests, per FastVideo rules.
- Use Modal through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker` for runtime validation.
- Focused validation after implementation:
  1. Modal L40S syntax/import smoke for changed Modal scripts, e.g. `python -m py_compile fastvideo/tests/modal/pr_test.py fastvideo/tests/modal/ssim_test.py fastvideo/tests/modal/kernel_build_cache.py`.
  2. Modal L40S cache miss validation on a temporary/dedicated cache subdir: run the helper once with `--build-if-missing` and confirm it builds, installs, stores a wheel, and logs a miss.
  3. Modal L40S cache hit validation using the same cache subdir: run the helper again and confirm it installs the cached wheel without invoking source build and logs a hit.
  4. A small Modal lane smoke with `pr_test.py` if feasible, such as `run_unit_test` or a narrow command path, to confirm the PR runner calls the helper correctly.
  5. A targeted SSIM workspace smoke if feasible without running the full suite, or a narrow SSIM filter if available, to confirm `ssim_test.py` calls the same helper in workspace setup.
- Before any draft PR creation in Stage 4, run `pre-commit run --all-files` and fix all issues.
- No SSIM reference update should be needed because this changes CI setup, not model output behavior.

## Open Questions
- Whether the implementation should mount a new dedicated Modal volume for kernel wheels or use the existing `hf-model-weights` volume under a namespaced cache directory. A dedicated volume is cleaner; the existing volume needs fewer new Modal resources.
- Whether Docker-prebuilt reuse is required in the first implementation pass, or whether conservative Modal wheel caching is sufficient for the first fix and Docker metadata can follow.

## Stage 2 Implementation Notes
- Added `fastvideo/tests/modal/kernel_build_cache.py`.
  - Standalone Modal-side CLI with `install`, `write-build-info`, and `print-key` commands.
  - Computes a conservative key from cache schema, kernel package version, `fastvideo-kernel` git tree/diff/submodule status or directory hash fallback, Python platform/cache tag, torch version/CUDA/file, nvcc output, CUDA-related env, backend, normalized `TORCH_CUDA_ARCH_LIST`, and normalized `CMAKE_ARGS`.
  - `install` first checks Docker-prebuilt metadata at `/opt/fastvideo-kernel-build-info.json`; exact key match installs the baked wheel.
  - If no matching baked wheel exists, it checks a Modal cache root, installs a matching cached wheel on hit, or builds a wheel on miss and stores `metadata.json` plus the wheel atomically under the cache key.
  - Cache-miss fallback builds without storing if the cache directory is unavailable.
- Updated `fastvideo-kernel/build.sh`.
  - Default `./build.sh` behavior remains source build/install.
  - New `--wheel-dir <dir>` mode uses the existing arch/backend setup, builds a wheel with `uv build --wheel --no-build-isolation`, then installs that wheel.
- Updated `fastvideo/tests/modal/pr_test.py`.
  - Adds a dedicated Modal volume named `fastvideo-kernel-build-cache` mounted at `/root/fastvideo-kernel-cache` for every build-kernel lane.
  - Replaces direct `cd fastvideo-kernel && ./build.sh` with the cache helper after dependency install and before the test command.
  - Commits the cache volume after build-kernel lanes, warning rather than masking the test result if the commit fails.
  - Leaves DreamVerse with `build_kernel=False` and no kernel cache volume.
- Updated `fastvideo/tests/modal/ssim_test.py`.
  - Mounts the same cache volume in SSIM workers.
  - Replaces direct `./build.sh` during `_prepare_ssim_workspace()` with the shared helper.
  - Commits the cache volume after workspace setup, warning rather than masking the setup result if the commit fails.
- Updated `docker/Dockerfile`.
  - Kernel image build now uses `./build.sh --wheel-dir /opt/fastvideo-kernel-wheels`.
  - Writes `/opt/fastvideo-kernel-build-info.json` with the helper so future images can prove when the baked wheel matches the checked-out CI source/ABI key.

## Validation So Far
- Local syntax-only checks passed:
  - `python -m py_compile fastvideo/tests/modal/kernel_build_cache.py fastvideo/tests/modal/pr_test.py fastvideo/tests/modal/ssim_test.py`
  - `bash -n fastvideo-kernel/build.sh`
- No local project tests were run, per FastVideo validation rules.

## Next Steps
- Stage and inspect the complete diff including the new helper.
- Commit with GPG signing and push immediately.
- Run Modal validation through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`:
  1. Modal L40S syntax/import smoke for the changed Modal scripts and helper.
  2. Modal L40S cache miss validation with a dedicated cache root.
  3. Modal L40S cache hit validation with the same cache root.
  4. If feasible, a narrow Modal runner smoke proving `pr_test.py` calls the helper.
- If Modal exposes issues, fix, commit, push, and update this handoff.
- After implementation validation, run the required Stage 3 review/adjudication loop before presenting the draft PR message.

## Running Log
- 2026-07-06T05:23:38Z: First attempt initialized Stage 1 handoff and inspected initial GitHub/code context. User interrupted before commit; `/tmp` worktree was later lost.
- 2026-07-06T05:38:05Z: Resumed, recreated missing worktree, fast-forwarded branch to current main, re-checked GitHub state, and recreated detailed Stage 1 handoff. No implementation changes made.
- 2026-07-06T06:55:05Z: User approved the recommended approach with "Go ahead". Recreated missing `/tmp` worktree from pushed branch, re-checked issue/PR state, inspected related PRs #1389 and #1449, and moved to Stage 2.
- 2026-07-06T07:05:36Z: Implemented shared Modal wheel-cache helper, `build.sh --wheel-dir`, PR/SSIM runner wiring, and Docker-prebuilt build-info support. Local syntax-only checks passed; Modal validation still pending.
