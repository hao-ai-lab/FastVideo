# Issue 1571 Handoff

## Current State

- Issue: #1571, "[ci] [Feature] Flaky GPU Test Retry"
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1571
- State: OPEN
- Labels: `scope: attention`, `scope: docs`
- Assignees: none
- Author: Satyam-53
- Created: 2026-07-09T16:47:04Z
- Updated: 2026-07-09T16:47:12Z
- Repo: `hao-ai-lab/FastVideo`
- Worktree: `/tmp/fastvideo-worktrees/issue-1571-flaky-gpu-test-retry`
- Branch: `issue/1571-flaky-gpu-test-retry`
- Base: `origin/main` at `9d909f5f0 [test]: remove dead and duplicate tests (-489 lines) (#1556)`
- Handoff path: `.agents/handoffs/issue-1571-handoff.md`
- Current stage: Stage 3 - Review, Adjudicate, And Iterate
- Implementation begun: yes. User approved the recommended central Modal pytest rerun approach.
- Last updated: 2026-07-10T10:35:28Z

## Stage 0 Notes

- Read `/home/toolbox/.codex/skills/fix-issue/SKILL.md`, `references/handoff.md`, and `references/stages.md`.
- Verified `gh` identity as `macthecadillac` using `gh api user --jq .login`.
- Local main checkout has unrelated untracked work; issue work is isolated in the `/tmp` worktree above.
- `git fetch origin` and `git fetch upstream` were run outside the sandbox because sandboxed fetch could not write `.git/FETCH_HEAD`.
- No local, `origin`, or `upstream` branches containing `1571` were found after fetch.
- No existing `.agents/handoffs/issue-1571-handoff.md` was found in `/home/toolbox/FastVideo` or relevant `/tmp` worktrees.
- Created new branch/worktree `issue/1571-flaky-gpu-test-retry` from `origin/main`.

## GitHub Context

- Full issue read with `gh issue view 1571 -R hao-ai-lab/FastVideo --json ...`.
- Body summary:
  - Buildkite currently has automatic retries for agent/git failures and some `exit_status: 1` lane retries.
  - There is no pytest-level rerun support, no transient-only retry rules, and no Modal pytest commands using `--reruns` or `--only-rerun`.
  - Reporter says to inspect CI docs for workflow details.
- Comments: none.
- Commenter-proposed fixes: none.
- Duplicate/related issue search:
  - `gh search issues "Flaky GPU Test Retry" --repo hao-ai-lab/FastVideo ...` found only issue #1571.
- PR search:
  - `gh search prs 1571 --repo hao-ai-lab/FastVideo ...` found no PRs.
  - `gh search prs "Flaky GPU Test Retry" --repo hao-ai-lab/FastVideo ...` found no PRs.
  - Focused `gh pr list --search "retry OR flaky OR rerun OR pytest"` found CI/test-adjacent PRs but no direct implementation of this issue.
- Related open PRs checked:
  - #1572 `[ci]: gate the full-suite trigger on pre-commit and docs build`, ready-for-review, not draft, head `maint/ci-gate-cheap-checks`. It gates Buildkite full-suite launch behind cheap checks; it does not add pytest reruns or transient-only retry rules.
  - #1577 `[ci]: cover transformers 5.x config compat for Qwen2.5-VL (#1576)`, ready-for-review, not draft. Test coverage PR, not retry support.
  - #1561 `[ci] Stop forcing FA4 in model-load lanes`, ready-for-review, not draft. CI lane configuration for issue #1558, not retry support.
  - #1562 `[ci]: cache FastVideo kernel builds in Modal`, ready-for-review, not draft, currently needs rebase. Modal cache work, not retry support.
  - #1389 `[ci] Modal: enable Flash Attention 3 on H100 perf tests with build cache`, ready-for-review, not draft. Modal/perf infrastructure, not pytest rerun support.
  - #1494 and #1183 are attention-related but not focused on CI retry behavior.
- No PR draft status was changed.
- Stage 2 re-check on 2026-07-10T10:10:42Z:
  - Verified `gh` identity as `macthecadillac`.
  - Issue #1571 remains OPEN with no comments and no assignees.
  - Focused open PR search for issue number/title/`pytest-rerunfailures`/`--only-rerun`/reruns found no direct implementation of this issue.
  - No PR draft status was changed.

## Investigation Log

- Full issue body/comment read complete.
- Related open PR scan complete enough to establish no direct implementation exists.
- Code/workflow/docs inspection complete for Stage 1.
- Read in-scope repo instructions:
  - Root `AGENTS.md`
  - `fastvideo/tests/AGENTS.md`
- Searched:
  - `rg -n "reruns|only-rerun|pytest-rerunfailures|retry|retries|flaky|exit_status|Buildkite|buildkite|modal" .github .buildkite docs fastvideo/tests tests scripts pyproject.toml`
  - `rg -n "pytest" .github .buildkite docs fastvideo/tests/modal tests/local_tests scripts pyproject.toml`
  - `rg -n "retry|retries|rerun|flaky|--reruns|only-rerun|pytest-rerunfailures" docs .buildkite fastvideo/tests/modal pyproject.toml .github`
  - `rg -n "pytest-rerunfailures|rerunfailures|--reruns|--only-rerun" .`
- Files inspected:
  - `.buildkite/pipeline.yml`
  - `.buildkite/scripts/pr_test.sh`
  - `fastvideo/tests/modal/pr_test.py`
  - `fastvideo/tests/modal/ssim_test.py`
  - `fastvideo/tests/contract/test_ci_test_collection.py`
  - `docs/contributing/ci_architecture.md`
  - `docs/contributing/testing.md`
  - `pyproject.toml`

## Code Findings

- `.buildkite/pipeline.yml` has Buildkite-level retries for agent/git failures (`exit_status: 128`, `exit_status: -1`) across direct, fastcheck, and full-suite steps. Only a few lanes retry `exit_status: 1`: direct SSIM, direct LoRA training, direct Training VSA, full-suite SSIM, full-suite LoRA training, and full-suite Training VSA.
- Buildkite retries rerun whole lanes. They do not provide pytest-level reruns and do not distinguish assertion failures from transient GPU/infra errors.
- `.buildkite/scripts/pr_test.sh` maps `TEST_TYPE` to Modal commands. Each lane executes a Modal function, but it does not pass any retry-specific pytest arguments or `PYTEST_ADDOPTS`.
- `fastvideo/tests/modal/pr_test.py` centralizes most Modal pytest lanes through `run_test_command()`, then embeds plain `pytest` commands for encoder, VAE, transformer, training, kernel, LoRA, unit, eval, performance, API, etc. None include `--reruns` or `--only-rerun`.
- `fastvideo/tests/modal/ssim_test.py` has git retry logic for clone/fetch/submodule setup, but SSIM task execution builds plain `pytest <test_file> -vs` plus SSIM-specific args. It does not add pytest rerun args.
- `pyproject.toml` depends on `pytest` in base dependencies and the `[test]` extra, but does not include `pytest-rerunfailures`.
- `docs/contributing/ci_architecture.md` documents that all Buildkite jobs go through `pr_test.sh` and then selected Modal test commands, but it does not describe pytest-level retry behavior.
- `docs/contributing/testing.md` names the CI files and Modal entrypoints, but does not mention retry policy.
- `fastvideo/tests/contract/test_ci_test_collection.py` is a relevant static contract test for CI lane wiring. It does not currently validate retry flags, but it is a focused post-change sanity check for CI-source text integrity.

## Current Hypothesis

- The issue is valid. Current code supports the reporter's claim: retries exist at Buildkite lane level and for git checkout inside SSIM setup, but there is no pytest-level rerun support, no `pytest-rerunfailures` dependency, and no transient-only `--only-rerun` rule on Modal pytest invocations.
- A correct fix should avoid expanding broad `exit_status: 1` Buildkite retries because that reruns entire Modal lanes, burns GPU time, and can mask deterministic failures. The better scope is inside Modal pytest execution, where individual failing tests can be rerun only when the failure output matches a conservative transient regex.

## Alternatives Considered

- Approach A, minimal Buildkite-only retry expansion:
  - Touch `.buildkite/pipeline.yml` and add `exit_status: 1` retries to more lanes.
  - Pros: small YAML change.
  - Cons: not pytest-level, not transient-only, reruns whole Modal jobs, likely contrary to issue wording.
  - Recommendation: do not use except as a fallback for lanes whose failures occur outside pytest.
- Approach B, central pytest rerun support in Modal orchestrators:
  - Add `pytest-rerunfailures` to `[project.optional-dependencies].test`.
  - Add a small Modal-local helper under `fastvideo/tests/modal/` that produces conservative rerun args: `--reruns`, `--reruns-delay`, and `--only-rerun <transient regex>`.
  - Apply it from `fastvideo/tests/modal/pr_test.py` through `PYTEST_ADDOPTS` or a shell-safe wrapper in `run_test_command()`, so existing pytest command strings do not each need manual edits.
  - Apply the same args in `fastvideo/tests/modal/ssim_test.py` when building SSIM task pytest commands.
  - Update docs with the retry policy and mention that deterministic assertion failures are not retried.
  - Pros: directly addresses issue, lower GPU waste, one behavior for most Modal pytest lanes, keeps retry rules close to Modal CI code.
  - Cons: needs careful regex choice and shell/env quoting; pytest process crashes without pytest failure text may still need lane-level retry.
  - Recommendation: preferred.
- Approach C, per-lane explicit pytest flags:
  - Manually add `--reruns ... --only-rerun ...` to each pytest command string in `pr_test.py` and `ssim_test.py`.
  - Pros: very explicit command output.
  - Cons: high duplication, easy to miss lanes, harder to maintain as commands are added.
  - Recommendation: avoid unless reviewers prefer no helper.
- Approach D, mark or refactor known flaky tests:
  - Add test-specific markers or retries only around known flaky tests.
  - Pros: precise if known flakes are enumerated.
  - Cons: issue requests general CI support; no specific flaky tests or comments were provided.
  - Recommendation: not sufficient alone.

## Recommended Plan

1. Implement Approach B.
2. Add `pytest-rerunfailures` to the `test` optional dependency in `pyproject.toml` so Modal's existing `uv pip install -e ".[test]"` path installs the plugin.
3. Add a small helper in `fastvideo/tests/modal/` with:
   - a default rerun count of 2;
   - a short delay, likely 5 to 10 seconds;
   - a conservative transient-only regex for infrastructure/GPU/runtime failures, not normal assertion mismatches;
   - structured argument generation so regex quoting is not duplicated.
4. Use that helper in `fastvideo/tests/modal/pr_test.py` so commands routed through `run_test_command()` inherit the retry policy without editing every pytest string.
5. Use the same helper in `fastvideo/tests/modal/ssim_test.py` by appending the retry args to the SSIM task pytest command or injecting `PYTEST_ADDOPTS` into the subprocess environment.
6. Keep existing Buildkite retries in `.buildkite/pipeline.yml` unless implementation reveals a specific non-pytest transient failure that needs lane-level handling.
7. Update `docs/contributing/ci_architecture.md` and `docs/contributing/testing.md` with the durable behavior: Modal pytest lanes use pytest-rerunfailures, rerun only failures matching the transient regex, and deterministic failures should still fail immediately.
8. Consider adding a focused static test under `fastvideo/tests/contract/` that verifies the Modal retry helper is referenced by both `pr_test.py` and `ssim_test.py`, and that `pytest-rerunfailures` remains in the test extra. This is optional but useful because CI retry wiring is text-heavy.
9. Avoid local project test runs per repo rules.

## Validation Plan

- Stage 1 ran no validation and made no implementation changes.
- Future Stage 2 validation should use Modal, not local project tests.
- Suggested focused Modal validation after implementation:
  - Use `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`.
  - Run a cheap static contract command such as `pytest fastvideo/tests/contract/test_ci_test_collection.py -q` plus any new retry-wiring contract test.
  - Run a dry command that prints or inspects pytest args if a helper-level test is added.
- If the implementation changes only dependency/docs/Modal orchestration text, no GPU model-quality SSIM validation should be required; GPU memory impact should be effectively none except rerun attempts can extend job wall time on transient failures.
- Before any future draft PR creation, run `pre-commit run --all-files` from the issue worktree and fix all issues.
- New PRs must be draft PRs only; existing PR draft status must not be changed.

## Open Questions

- What exact transient regex should be accepted? I recommend starting conservative: CUDA/NCCL/Triton/runtime communication errors, Modal interruption/worker failures, network timeout/reset/5xx, and excluding plain assertion or numerical parity mismatches.
- Should retry behavior apply to all Modal pytest lanes by default, or only Buildkite-triggered Modal runs? I recommend all Modal CI orchestrator runs because the issue explicitly calls out Modal pytest commands, but an opt-out env var can be added if maintainers want manual runs to stay single-shot.

## Stage 1 Report Draft

- No implementation has been performed.
- Recommended approach: central pytest-rerunfailures support in Modal orchestrators with a transient-only regex, plus docs.
- Awaiting user guidance before Stage 2 implementation.

## Selected Approach

- User approved the Stage 1 recommendation on 2026-07-10.
- Implement Approach B:
  - Add `pytest-rerunfailures` to the test extra.
  - Add one Modal-local helper for retry args and transient regex.
  - Apply it centrally to `fastvideo/tests/modal/pr_test.py` and `fastvideo/tests/modal/ssim_test.py`.
  - Add focused static contract coverage.
  - Update CI/testing docs.

## Stage 2 Implementation Log

- Implemented central Modal pytest retry wiring:
  - Added `fastvideo/tests/modal/pytest_retry.py` with shared `pytest-rerunfailures` args, retry count `2`, delay `10s`, and a conservative transient-only regex.
  - Updated `fastvideo/tests/modal/pr_test.py` to compute `PYTEST_ADDOPTS` from the cloned repo's helper after checkout, so Modal does not need to mount helper files beside the entrypoint.
  - Updated `fastvideo/tests/modal/ssim_test.py` to append the same retry args to each SSIM subprocess pytest command.
  - Added `pytest-rerunfailures` to the `test` extra in `pyproject.toml`.
  - Added static contract coverage in `fastvideo/tests/contract/test_ci_pytest_reruns.py`.
  - Updated `docs/contributing/ci_architecture.md` and `docs/contributing/testing.md` with durable retry policy documentation.
- Local cleanup:
  - `uv run --with modal ... --help` was accidentally started from the repo without `--no-project`; it began resolving FastVideo dependencies and created a small `.venv`.
  - Stopped that process and removed the generated `.venv`.

## Validation Log

- Do not run project tests locally rule followed; validation was run on Modal L40S through `/tmp/fastvideo-worktrees/interleavethinker-launcher/fastvideo/tests/modal/launch_l40s_job.py`.
- Failed validation attempt 1:
  - Command:
    `uv run --no-project --with modal python -m modal run /tmp/fastvideo-worktrees/interleavethinker-launcher/fastvideo/tests/modal/launch_l40s_job.py --install-extra test --no-build-kernel --apply-local-patch --patch-paths "pyproject.toml,fastvideo/tests/modal/pr_test.py,fastvideo/tests/modal/ssim_test.py,fastvideo/tests/modal/pytest_retry.py,fastvideo/tests/contract/test_ci_pytest_reruns.py,docs/contributing/ci_architecture.md,docs/contributing/testing.md" --command "pytest fastvideo/tests/contract/test_ci_test_collection.py fastvideo/tests/contract/test_ci_pytest_reruns.py -q"`
  - Modal URL: https://modal.com/apps/hao-ai-lab/main/ap-r6oqddOSxQlgIduaTgx2Jq
  - Result: failed before tests. `uv pip install -e '.[test]'` attempted to build `fastvideo-kernel==0.3.2` from sdist and failed because `cutlass/cutlass.h` was missing from the package build context.
  - Assessment: unrelated to this CI retry patch; the targeted static tests do not require installing FastVideo or building kernels.
- Failed validation attempt 2:
  - Command used `--install-extra none` and `uv pip install pytest` in the remote command.
  - Modal URL: https://modal.com/apps/hao-ai-lab/main/ap-njpA3gOITr1UJ4QO3DdqD6
  - Result: failed before tests with `/bin/bash: line 1: uv: command not found` because `--install-extra none` skips environment sourcing that places `uv` on PATH.
  - Assessment: command setup issue only.
- Passing validation:
  - Command:
    `uv run --no-project --with modal python -m modal run /tmp/fastvideo-worktrees/interleavethinker-launcher/fastvideo/tests/modal/launch_l40s_job.py --install-extra none --no-build-kernel --apply-local-patch --patch-paths "pyproject.toml,fastvideo/tests/modal/pr_test.py,fastvideo/tests/modal/ssim_test.py,fastvideo/tests/modal/pytest_retry.py,fastvideo/tests/contract/test_ci_pytest_reruns.py,docs/contributing/ci_architecture.md,docs/contributing/testing.md" --command "source $HOME/.local/bin/env 2>/dev/null || true; source /opt/venv/bin/activate 2>/dev/null || true; uv pip install pytest && pytest fastvideo/tests/contract/test_ci_test_collection.py fastvideo/tests/contract/test_ci_pytest_reruns.py -q"`
  - Modal URL: https://modal.com/apps/hao-ai-lab/main/ap-Z7mqFKXntYOTP64dNof6m8
  - Result: passed, `6 passed in 0.04s`.
  - Modal return summary: local patch applied, `install_extra=none`, `build_kernel=False`, commit `732f9df86c5c67b6f55a7dd1e7400e1b78d9e69a`.
- Pre-commit gate:
  - Initial command from issue worktree:
    `pre-commit run --all-files`
  - Result: failed because `pre-commit` was not installed.
  - Rerun command from issue worktree:
    `uv run --no-project --with pre-commit pre-commit run --all-files`
  - Result: all hooks passed except mypy, which failed with `issue-1571-flaky-gpu-test-retry is not a valid Python package name`. This was a worktree path artifact, not a code failure.
  - Created detached validation worktree at `/tmp/fastvideo_worktrees/issue_1571_precommit` from `HEAD` to avoid the hyphenated worktree basename.
  - Passing command from detached validation worktree:
    `uv run --no-project --with pre-commit pre-commit run --all-files`
  - Passing result:
    - yapf: Passed
    - ruff: Passed
    - codespell: Passed
    - PyMarkdown: Passed
    - actionlint: Passed
    - mypy: Passed
    - Check filenames: Passed
    - suggestion: Passed
  - Detached validation worktree had no file changes after pre-commit and was removed.

## Commits And Pushes

- `732f9df86c5c67b6f55a7dd1e7400e1b78d9e69a` - signed Stage 1 handoff commit, pushed to `origin/issue/1571-flaky-gpu-test-retry`.
- `8923d8b60` - signed Stage 2 implementation commit `[ci]: add transient pytest reruns for modal tests`, pushed to `origin/issue/1571-flaky-gpu-test-retry`.
- `624ed3cff` - signed handoff-only commit recording Stage 2 validation/commit/push status, pushed to `origin/issue/1571-flaky-gpu-test-retry`.

## Stage 3 Review Loop

- Review-code sub-agent 1:
  - Agent id: `019f4b90-0944-7882-8648-8d797dc9b4d7`
  - Nickname: Copernicus
  - Prompt summary: use `$review-code` to review `macthecadillac/FastVideo` branch `issue/1571-flaky-gpu-test-retry` for issue #1571; review-only; use `gh` as `macthecadillac`; no local tests/pre-commit; report actionable findings or no findings.
  - Status: completed.
  - Findings: no actionable code findings in committed branch `issue/1571-flaky-gpu-test-retry` at `624ed3cff4f4a193e27bb1a8f5bd122bc7977e81`.
  - Issue fit: branch appears to address hao-ai-lab/FastVideo#1571 by adding pytest-level reruns, using `--only-rerun`, keeping the policy transient-focused, and wiring Modal pytest CI paths instead of broad Buildkite lane retries.
  - Related branches: `issue/1571-flaky-gpu-test-retry` only; it is the reviewed branch.
  - Validation gaps noted: existing Modal static contract validation does not prove an end-to-end Modal CI lane executes pytest with the retry options; useful follow-ups would be `pr_test.py::run_unit_test` or a targeted SSIM run on the committed branch.
  - Adjudicator/fixer: not spawned because there were no actionable findings.

## Draft PR Message

Title:

```text
[ci]: add transient pytest reruns for Modal tests
```

Body:

```markdown
## Summary

- add a shared Modal pytest retry policy using `pytest-rerunfailures`
- apply transient-only reruns to standard Modal pytest lanes through `PYTEST_ADDOPTS`
- apply the same rerun arguments to SSIM subprocess pytest commands
- document the CI retry behavior and add static contract coverage

Fixes #1571.

## Validation

- Modal L40S focused contract validation:
  - `uv run --no-project --with modal python -m modal run /tmp/fastvideo-worktrees/interleavethinker-launcher/fastvideo/tests/modal/launch_l40s_job.py --install-extra none --no-build-kernel --apply-local-patch --patch-paths "pyproject.toml,fastvideo/tests/modal/pr_test.py,fastvideo/tests/modal/ssim_test.py,fastvideo/tests/modal/pytest_retry.py,fastvideo/tests/contract/test_ci_pytest_reruns.py,docs/contributing/ci_architecture.md,docs/contributing/testing.md" --command "source $HOME/.local/bin/env 2>/dev/null || true; source /opt/venv/bin/activate 2>/dev/null || true; uv pip install pytest && pytest fastvideo/tests/contract/test_ci_test_collection.py fastvideo/tests/contract/test_ci_pytest_reruns.py -q"`
  - Result: `6 passed in 0.04s`
- `uv run --no-project --with pre-commit pre-commit run --all-files`
  - Result: passed from `/tmp/fastvideo_worktrees/issue_1571_precommit`

## Review Loop

- `review-code` sub-agent reviewed `macthecadillac/FastVideo` branch `issue/1571-flaky-gpu-test-retry` for issue #1571 and reported no actionable findings.
- No adjudicator/fixer sub-agent was spawned because there were no actionable findings to adjudicate.

## GPU Memory Impact

No direct GPU memory impact. The change can extend wall time for transient infrastructure failures by rerunning matching individual pytest failures, but it avoids broad Buildkite lane retries for deterministic failures.

# Checklist
- [x] I ran pre-commit run --all-files and fixed all issues
- [x] I added or updated tests for my changes
- [x] I updated documentation if needed
- [x] I considered GPU memory impact of my changes
```

## Remaining Risk

- End-to-end Modal CI lanes were not run through `pr_test.py::run_unit_test` or SSIM. Static contract validation verifies dependency/wiring/docs, but not a full Modal lane executing pytest with the retry args in a real suite.
- Handoff remains active and tracked. Before Stage 4 draft PR creation, transfer needed context into the PR body, remove `.agents/handoffs/issue-1571-handoff.md` with `git rm`, commit and push that deletion, and verify the branch no longer contains the handoff.

## Next Steps

- Commit and push final handoff state.
- Present Stage 3 summary and full draft PR message to the user without opening a PR.
