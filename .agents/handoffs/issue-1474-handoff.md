# Issue 1474 Handoff

## Issue Snapshot
- Issue: #1474
- Title: [Bug]: Avoid GPU-to-CPU output materialization when save_video=False and return_frames=False
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1474
- State: open
- Labels: pending full issue read
- Assignees: pending full issue read
- Created/updated: pending full issue read

## Worktree And Branch
- Repo: hao-ai-lab/FastVideo
- Base checkout: /home/toolbox/FastVideo
- Worktree: /tmp/fastvideo-worktrees/issue-1474-output-materialization
- Branch: issue/1474-output-materialization
- Base ref: origin/main at 9d909f5f0 ([test]: remove dead and duplicate tests (-489 lines) (#1556))
- Handoff path: .agents/handoffs/issue-1474-handoff.md
- Current stage: Stage 2 - implementing user-approved narrow metadata-only fix
- Implementation begun: yes

## Authentication And Sandbox
- `gh api user --jq .login` was run outside the sandbox after sandboxed network failed.
- Verified GitHub identity: macthecadillac
- GitHub connector was not used.
- Fetched `origin` and `upstream`; did not fetch or use the `SolitaryThinker` remote.

## Stage 0 Discovery
- Local branch search before fetch found no branches matching `*1474*`.
- After fetching `origin` and `upstream`, `git branch --all --list '*1474*'` still found no matching branches.
- No existing issue 1474 handoff was found in a matching branch.
- Created dedicated worktree and branch:
  `git worktree add -b issue/1474-output-materialization /tmp/fastvideo-worktrees/issue-1474-output-materialization origin/main`
- Main checkout has unrelated untracked files; all issue work is isolated in the dedicated worktree.

## GitHub Context
- Full issue read with:
  `gh issue view 1474 -R hao-ai-lab/FastVideo --json number,title,state,body,labels,assignees,author,comments,createdAt,updatedAt,url,milestone`
- Issue labels: `performance`, `scope: inference`, `scope: attention`.
- Issue assignees: `Satyam-53`.
- Issue author: `Satyam-53`.
- Created: 2026-06-22T01:01:31Z.
- Updated: 2026-07-01T02:41:07Z.
- Milestone: none.
- Comments: none.
- Reporter claim: `VideoGenerator.generate_video` copies `output_batch.output` to CPU before checking whether callers requested frames. When `save_video=False` and `return_frames=False`, the copy and downstream frame materialization are discarded and add profiling/benchmark overhead.
- Reporter-proposed code location:
  ```python
  if output_batch.output.shape == samples.shape:
      samples.copy_(output_batch.output)
  else:
      samples = output_batch.output.cpu()
  ```
- Reporter edge/open questions:
  1. Whether trajectory metadata should still be returned in metadata-only mode.
  2. Whether audio outputs should be treated separately.
  3. Whether this should use only `save_video=False`/`return_frames=False` or add an explicit `materialize_output=False` flag.
  4. Whether benchmark/profile scripts should use metadata-only mode by default.
  5. Whether timing logs should include skipped/zero-duration `PostDecodeFrameProcessStage`.
  6. Whether `output_type == "latent"` should follow `return_frames` or use a separate latent return flag.
  7. Whether downstream users rely on CPU materialization side effects despite `return_frames=False`.

### Related PRs And Searches
- `gh pr list --state open --search "1474"` found no open PR explicitly referencing #1474.
- `gh pr list --state open --search "return_frames save_video"` found:
  - PR #1362 `[perf] Quantize frames to uint8 on-device before the post-decode D->H copy`
    - URL: https://github.com/hao-ai-lab/FastVideo/pull/1362
    - State: open, ready-for-review (`isDraft=false`), branch `perf/postdecode-gpu-uint8`.
    - Files: `fastvideo/entrypoints/video_generator.py`.
    - Does not list #1474 in `closingIssuesReferences`.
    - Directly overlaps. Its third patch skips `samples` preallocation/copy when `return_frames=False`, after its earlier patches switch frame building to read `output_batch.output` directly and quantize to uint8 on-device. That broader PR also changes save-to-disk behavior and pixel postprocessing.
  - PR #1496 `[misc] Clean up QAD 5090 example inference scripts`
    - URL: https://github.com/hao-ai-lab/FastVideo/pull/1496
    - State: open, ready-for-review (`isDraft=false`).
    - Search hit only; no apparent coverage for #1474.
- `gh pr list --state open --search "materialization"` found no open PRs.
- `gh pr list --state all --search "PostDecodeFrameProcessStage"` found PR #1362 plus merged docs/timing PRs #1473 and #1430. Only #1362 appears to affect this issue.
- `gh pr list --state all --search "VideoGenerator generate_video"` found several historical API/docs PRs, including #1132, #1220, #1366, #1430, #1448, but none appear to address this specific dead output materialization.
- `gh issue list --state all --search "return_frames save_video"` found #1474 and unrelated/noise issues; no duplicate specific to metadata-only output materialization.
- `gh issue list --state all --search "materialization"` found only #1474.
- Draft status of related PRs was read only and not changed.

## Code Investigation
- Root `AGENTS.md`, `fastvideo/AGENTS.md`, and `fastvideo/tests/AGENTS.md` read.
- No matching `.agents/lessons` entries for issue 1474/output materialization.
- Searches run:
  - `rg -n "return_frames|save_video|PostDecodeFrameProcessStage|output_batch|generate_video|samples\\.copy_|\\.cpu\\(\\)" fastvideo/entrypoints fastvideo/pipelines fastvideo/configs fastvideo/tests tests/local_tests`
  - `rg -n "return_frames|save_video" examples docs fastvideo/tests tests/local_tests`
  - `rg -n "_generate_single_video|PostDecodeFrameProcessStage|output_batch.output|return_frames" fastvideo/tests/entrypoints fastvideo/tests/performance fastvideo/tests/contract`
- Relevant code:
  - `fastvideo/entrypoints/video_generator.py:750-764` starts the forward thread and preallocates CPU `samples` unless `output_type == "latent"`.
  - `fastvideo/entrypoints/video_generator.py:776-782` unconditionally copies `output_batch.output` into CPU `samples` or falls back to `output_batch.output.cpu()` before checking `return_frames`.
  - `fastvideo/entrypoints/video_generator.py:802-813` unconditionally builds `frames` for non-latent/non-audio outputs. It uses `samples`, converts every frame to uint8, and calls `.cpu().numpy()`.
  - `fastvideo/entrypoints/video_generator.py:819-823` only needs `frames` when `batch.save_video and not is_latent_output`.
  - `fastvideo/entrypoints/video_generator.py:898-915` returns `samples` and `frames` only when `batch.return_frames` is true. It still returns prompt, size, generation timing, e2e latency, logging info, trajectories, audio fields, video path, and peak memory metadata independently.
  - `fastvideo/pipelines/pipeline_batch_info.py:226-242` stores `output`, trajectory fields, `save_video`, and `return_frames` on `ForwardBatch`.
  - `fastvideo/api/results.py:11-28` allows `samples` and `frames` to be `None`, preserving metadata-only result shape.
  - `fastvideo/tests/entrypoints/test_video_generator.py` mostly tests routing and path helpers; it does not currently exercise the `_generate_single_video` output materialization block.
  - `fastvideo/tests/performance/test_inference_performance_component_times.py:127-144` intentionally excludes generator bookkeeping stages such as `PostDecodeFrameProcessStage` from component metrics.

## Stage 2 Implementation
- Files changed:
  - `fastvideo/entrypoints/video_generator.py`
  - `fastvideo/tests/entrypoints/test_video_generator.py`
- Generator change:
  - Computes `is_latent_output`, `needs_frame_output`, and `needs_samples_buffer` before CPU sample allocation.
  - Skips pinned CPU sample allocation and skips the decoded tensor D->H copy when neither returned samples nor frames are needed.
  - Skips post-decode frame construction when no frames are needed.
  - Keeps existing return-frame and save-video behavior on the original `samples` path.
  - Keeps audio-only metadata output returning `audio` and `audio_sample_rate` while avoiding frame output when `return_frames=False`.
  - Keeps latent metadata output returning no `samples` when `return_frames=False`.
- Tests added:
  - metadata-only pixel output skips shape inspection, `.cpu()`, and frame-grid construction;
  - `return_frames=True` still materializes samples and frames;
  - `save_video=True, return_frames=False` still builds frames and saves;
  - audio-only metadata mode returns audio without frames;
  - latent metadata mode skips CPU materialization.
- Static checks:
  - `git diff --check` passed.
  - Manual >120-column scan found only a pre-existing line in `video_generator.py:156`, outside the patch.
- Validation status:
  - Modal L40S attempt `ap-XvjXs8VEmzo4robtq5g4fC` ran
    `pytest fastvideo/tests/entrypoints/test_video_generator.py -q` through
    `launch_l40s_job.py` from branch `interleavethinker`, with local patch applied.
  - Attempt failed before tests during `uv pip install -e '.[dev]'` because `fastvideo-kernel==0.3.2`
    was built from an sdist and could not find `cutlass/cutlass.h`.
  - Modal L40S rerun `ap-7ETeTJfHzyShAylBS7eMFb` used the same command with `--install-extra none`
    to use the dev image's existing environment. Result:
    `pytest fastvideo/tests/entrypoints/test_video_generator.py -q` passed, `25 passed in 0.33s`.

## Current Hypothesis
- The issue is valid on current `origin/main`: even in the exact no-output mode (`save_video=False`, `return_frames=False`), `_generate_single_video` still performs at least one CPU materialization of `output_batch.output`, then builds a `frames` list and drops both `samples` and `frames` from the returned result.
- The minimal behavior change should be a metadata-only path gated by `not batch.save_video and not batch.return_frames` for pixel outputs. In that mode, skip:
  - pinned CPU `samples` preallocation;
  - `samples.copy_(output_batch.output)` / `.cpu()`;
  - post-decode frame construction and `.cpu().numpy()`.
- Metadata should still be returned: `prompts`, `size`, `generation_time`, `e2e_latency`, `logging_info`, trajectory fields, audio fields if produced, `video_path=None`, and `peak_memory_mb`.
- This should not introduce an explicit `materialize_output` flag for the issue-specific fix; existing `save_video`/`return_frames` semantics already express whether pixel output is needed.
- For `output_type == "latent"`, preserve current behavior when `return_frames=True` and skip CPU latent materialization when `return_frames=False`. The result already returns latent samples only through `samples`.
- For audio-only workloads, preserve audio return behavior because audio is a primary output and currently returned independent of `return_frames`. The placeholder video tensor can be skipped when no frames are requested and no video is saved, but audio fields should remain.
- PR #1362 may supersede or conflict with this branch because it touches the same block and implements a broader optimization. If #1362 lands first, this branch should be rebased and likely reduced to a no-save/no-return postprocess skip if still needed.

## Alternatives And Recommendation
### Approach A - Minimal Metadata-Only Path For #1474
- Add local booleans such as `needs_samples = batch.return_frames` and `needs_frames = batch.return_frames or save_to_disk_candidate` after `output_batch` is available, or compute a conservative `materialize_pixel_output = batch.save_video or batch.return_frames` before allocation.
- Skip `samples` allocation/copy and skip frame postprocessing only when no samples/frames are needed.
- Keep current save-to-disk and return-frames behavior unchanged.
- Touch likely files:
  - `fastvideo/entrypoints/video_generator.py`
  - `fastvideo/tests/entrypoints/test_video_generator.py`
- Tradeoff: narrow and low-risk for #1474, but does not optimize `save_video=True, return_frames=False`; PR #1362 covers that broader case.

### Approach B - Backport/Adapt PR #1362's Broader Output Path
- Move post-decode frame conversion to read `output_batch.output` directly and quantize to uint8 on-device, then skip `samples` whenever `return_frames=False`.
- Tradeoff: larger performance win, but it duplicates/competes with open ready-for-review PR #1362 and changes pixel exactness details (GPU vs CPU cast, clamping). It needs stronger SSIM validation and maintainer coordination.

### Approach C - Add Explicit `materialize_output` Flag
- Add a new API/config flag to override output materialization independently of `save_video` and `return_frames`.
- Tradeoff: addresses reporter question 3, but adds a new public API surface for a behavior already expressible by existing flags. This is over-engineered for the current issue.

### Approach D - Wait For Or Help Land PR #1362
- Treat #1362 as the fix and avoid duplicate work.
- Tradeoff: #1362 does not currently close #1474 and has a broader scope. If maintainers want a smaller issue-specific PR, waiting does not move #1474 directly.

Recommended approach: Approach A. It directly resolves #1474 with the smallest behavioral surface, avoids taking over the broader PR #1362 optimization, and can be regression-tested with fake tensors/executor state.

## Validation Plan
- Do not run project tests locally.
- Add focused tests in `fastvideo/tests/entrypoints/test_video_generator.py`:
  - no-save/no-return pixel output should not call `.cpu()`, should not copy into a CPU `samples` buffer, should not call `torchvision.utils.make_grid`, and should return `samples is None`, `frames is None`, `video_path is None`, with metadata preserved.
  - return-frames path should still materialize/return samples and frames.
  - save-video path should still build frames and call the save path.
  - latent output with `return_frames=False` should skip CPU latent materialization and return `samples is None`; latent output with `return_frames=True` should preserve current `samples` behavior.
  - audio-only path should continue returning `audio`/`audio_sample_rate` independent of `return_frames`.
- Focused Modal L40S validation after implementation:
  - Use `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`.
  - Run a targeted entrypoint test selection if feasible, e.g. `pytest fastvideo/tests/entrypoints/test_video_generator.py -q`.
  - Run a lightweight generation smoke in metadata-only mode if a suitable existing smoke can be targeted without excessive cost.
- Before any future draft PR creation, run `pre-commit run --all-files` and fix all issues.
- No targeted Wan T2V SSIM is expected to be required for Approach A because output numerics are unchanged when output is actually saved or returned; the optimized path discards pixel outputs by contract. If Approach B is selected, SSIM validation is required because pixel conversion changes device/dtype/cast behavior.

## Recommended Implementation Plan
1. Re-check issue #1474, comments, and open PRs with `gh` before editing for Stage 2.
2. In `fastvideo/entrypoints/video_generator.py`, derive explicit booleans for output materialization:
   - whether samples are needed for the returned result (`batch.return_frames`);
   - whether frames are needed for saving or returning (`batch.save_video` for non-latent output, or `batch.return_frames`);
   - whether this is metadata-only (`not needs_samples and not needs_frames`, preserving audio output handling).
3. Use those booleans to skip CPU `samples` allocation and the `samples.copy_`/`.cpu()` block when samples are not needed.
4. Guard post-decode frame construction so it only runs when frames are needed; still record `PostDecodeFrameProcessStage` timing, likely near-zero/skipped duration, for logging continuity.
5. Keep result shape stable: `samples` and `frames` remain `None` when `return_frames=False`; metadata fields remain populated; `video_path` remains `None` when not saving.
6. Add focused fake-based tests in `fastvideo/tests/entrypoints/test_video_generator.py`. Prefer sentinel output tensors or monkeypatches that fail if `.cpu()`/copy/frame postprocess is invoked in metadata-only mode.
7. Validate on Modal, then commit with GPG signing and push.
8. Run Stage 3 review/adjudication loop on committed code before presenting a draft PR message.

## Running Log
- 2026-07-07: Started Stage 0/1 for issue 1474 using `$fix-issue`.
- 2026-07-07: Verified `gh` login as `macthecadillac`.
- 2026-07-07: Created branch/worktree and initialized this handoff. No implementation changes.
- 2026-07-07: Read issue body; no comments present.
- 2026-07-07: Checked open PRs and focused searches. PR #1362 is the only directly overlapping open PR and is ready-for-review; no draft status changed.
- 2026-07-07: Inspected `fastvideo/entrypoints/video_generator.py`, `fastvideo/pipelines/pipeline_batch_info.py`, `fastvideo/api/results.py`, and relevant tests. Current code confirms the issue.
- 2026-07-07: Completed Stage 1 recommendation. No implementation changes.
- 2026-07-07: Rechecked PR #1362 after user asked about conflicts/dependency. It remains open, `isDraft=false`, `mergeable=MERGEABLE`, `mergeStateStatus=BLOCKED`, touches only `fastvideo/entrypoints/video_generator.py`, and has no `closingIssuesReferences`.
- 2026-07-07: User approved moving to Stage 2 with recommended Approach A. Rechecked issue #1474 and related PR state before editing: issue still open with no comments; no open PR references #1474; PR #1362 remains open, `isDraft=false`, `mergeable=MERGEABLE`, `mergeStateStatus=BLOCKED`, and touches `fastvideo/entrypoints/video_generator.py`.

## Next Steps
- Implement Approach A in `fastvideo/entrypoints/video_generator.py`.
- Add focused regression tests in `fastvideo/tests/entrypoints/test_video_generator.py`.
- Validate on Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`.
- Commit with GPG signing and push.
- Run Stage 3 review/adjudication.
