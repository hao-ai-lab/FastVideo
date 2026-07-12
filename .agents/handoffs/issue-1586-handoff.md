# Issue 1586 Handoff

## Identity And Workspace

- Repository: `hao-ai-lab/FastVideo`
- Issue: #1586, `[CI] VSA training lane flaky on H200: anomalous H200 grad_norm reference + unpinned GPU (exit 1)`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1586
- Handoff: `.agents/handoffs/issue-1586-handoff.md`
- Branch: `issue/1586-vsa-h200-reference`
- Worktree: `/tmp/fastvideo-worktrees/issue-1586-vsa-h200-reference`
- Base: `upstream/main` at `970409962f358afd529b969a378174c849665837`
- Verified GitHub identity: `macthecadillac` via `gh api user --jq .login`
- Current stage: Stage 1, deep dive and plan
- Implementation begun: no; Stage 1 permits only updates to this handoff
- Started/resumed: 2026-07-12 UTC; new investigation, no prior issue branch or handoff found

## Issue Snapshot

- State: open
- Created: 2026-07-11T08:15:37Z
- Updated: 2026-07-11T08:15:45Z
- Author: `Mister-Raggs`
- Assignees: none
- Labels: `installation`, `scope: training`, `scope: inference`, `scope: attention`, `scope: kernel`, `scope: distributed`, `scope: model`
- Milestone: none
- Comments: none as of the Stage 1 read

## Stage 0 Discovery

- No local branch, cached remote ref, fork branch, or upstream branch containing `1586` was found.
- No `issue-1586-handoff.md` was found in the active checkout or existing FastVideo issue worktrees.
- No open PR directly references `1586`; broader open VSA PRs found by term search require scope evaluation.
- Current upstream main was fetched before creating this dedicated worktree.

## Reporter Claim And Proposed Fixes

The reporter says `VSA/test_training_loss_VSA.py::test_distributed_training` is flaky because a Modal declaration requests `H100:2` but may execute on H200, while the test selects a device-specific checked-in reference at runtime. The H200 `grad_norm` reference is about 0.245, versus about 1.118 on H100 and 1.237 in a recent H200 run. The assertion threshold is 0.5, so an H200 allocation fails by about 0.99.

Proposed fixes, in the reporter's order:

1. Pin the VSA lane to one reliably available GPU type.
2. Regenerate the H200 reference on an explicitly requested H200 pair.
3. Collapse H100/H200 to one tolerant reference if validation proves the metrics track.

These are hypotheses only. No implementation or Modal validation has run.

## Investigation Log

- Read the issue body and all comments (there are no comments).
- Queried the complete open PR list for issue number and VSA terms. No PR closes or directly references #1586. Open PRs #1563, #1494, #1384, and #1183 mention VSA for different backend/configuration concerns and do not appear to address the training reference flake; deeper code/history inspection is pending.
- Read the in-scope test guidance plus the CI, testing, and VSA contributor docs. No nested guidance or relevant lesson changes the plan.
- `fastvideo/tests/modal/pr_test.py:222-229` requests upgradeable `gpu="H100:2"`.
- `fastvideo/tests/training/VSA/test_training_loss_VSA.py:112-127` selects H100/H200 references by detected device and compares `avg_step_time`, `grad_norm`, `step_time`, and `train_loss`. The current H100/H200 grad norms are 1.118342604637146 and 0.24477368593215942; their 0.874 difference exceeds the 0.5 assertion threshold.
- The issue's PR #1352 provenance claim is incorrect: #1352 added audio metrics and never touched VSA training.
- Actual history: PR #900 manually changed an older VSA grad norm from 1.260593056678772 to 0.245593056678772; PR #933 introduced the H200 JSON with 0.24477368593215942 while its author said the CI tests were "really messed up"; PR #935 renamed the old L40S JSON to H100 and manually changed its metrics; release commit `9cd6a86b95` manually changed the nominal H100 values again. Both reference files have weak provenance, and the H200 value is demonstrably inconsistent with current reported behavior.
- PR #1210 added two exit-code-1 retries for `training_vsa`. This can hide intermittent allocation symptoms but cannot correct an invalid reference.
- Open PR #1410 touches VSA activation-checkpoint behavior but not these CI files or baselines. Other VSA PRs are also unrelated; no open PR covers #1586.
- Modal's official current GPU guide states that `H100` requests may be upgraded to H200 and documents `H100!` as the opt-out. Its client parser splits the count after `:`, making `H100!:2` the strict two-GPU form.
- Buildkite installs the latest Modal client when absent, so the current documented syntax applies.
- The required `interleavethinker` validation launcher offers upgradeable `H100:2`; Stage 2 validation must print/check the actual device and may need retrying for H100. A static contract test can lock production CI to `H100!:2`.

## Current Hypothesis

The issue is valid. CI asks for an upgradeable H100 pair, then selects a hardware-specific baseline after allocation. The H200 path fails current grad-norm behavior by about 0.99. The reporter's pinning direction works only with Modal's strict marker: `H100!:2`, not the existing `H100:2`.

Recommended scope is a strict H100-only lane, removal of the known-bad H200 reference/selection path, an explicit rejection of non-H100 manual runs, and focused contract coverage for the decorator. The nominal H100 file also has weak historical provenance, so it should be validated on actual H100 without changing passing values casually.

## Alternatives And Tradeoffs

1. **Strict H100 and one baseline (recommended):** deterministic hardware and one maintained baseline; potential downside is longer capacity waits.
2. **Strict declaration only:** smallest diff, but leaves a known-invalid H200 path appearing supported.
3. **Regenerate both baselines:** preserves a larger capacity pool, but keeps allocator-dependent CI and doubles baseline maintenance; requires repeated paired hardware runs.
4. **Shared Hopper baseline:** simpler, but assumes hardware-invariant numerics from insufficient evidence and still varies runtime hardware.

## Recommended Stage 2 Plan

1. Re-check issue comments and open PRs.
2. Change `run_training_tests_VSA` to `gpu="H100!:2"`.
3. Simplify the training regression to one H100 reference, reject non-H100 devices clearly, and delete the H200 JSON.
4. Add a no-import AST/text contract test that locks the VSA Modal decorator to `H100!:2`.
5. Via `interleavethinker:fastvideo/tests/modal/launch_l40s_job.py`, run the contract test on L40S and the full VSA test on an actual H100 pair, printing device identity. Retry if that launcher's upgradeable request lands on H200.
6. Keep the H100 reference unchanged if all current metrics pass. If H100 fails reproducibly, stop and report measurements before expanding scope into rebaselining.
7. Commit with GPG signing, push immediately, then run the required Stage 3 review/adjudication loop. Run `pre-commit run --all-files` before any future draft PR, and never create a non-draft PR or alter existing draft status.

## Validation Status

- Local project tests: not run; prohibited by repository instructions.
- Modal jobs: not run; prohibited during Stage 1.
- Pre-commit: not run; mandatory before any future draft PR creation.

Pass criteria: production decorator is exactly `H100!:2`; the full test reports H100 and all four metrics pass existing thresholds; no path references the deleted H200 JSON; Stage 3 validation and review gates pass.

## Next Steps

1. Re-check GitHub issue/comment/open-PR state before ending Stage 1.
2. Commit and push this Stage 1 handoff with GPG signing.
3. Await user approval or alternate Stage 2 guidance; no implementation has begun.
