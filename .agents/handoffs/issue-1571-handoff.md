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
- Current stage: Stage 1 - Deep Dive And Plan
- Implementation begun: no. Stage 1 is analysis/planning only.
- Last updated: 2026-07-10T08:40:26Z

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

## Next Steps

- Commit and push this handoff-only Stage 1 state.
- Present Stage 1 report to the user and ask whether to implement the recommended approach.
