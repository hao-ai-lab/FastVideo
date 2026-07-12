# Issue 1585 Handoff

## Workload State

- Issue: `hao-ai-lab/FastVideo#1585`
- Title: `[CI] Fastcheck LoRA Training lane times out (exit 124): 15m budget doesn't cover Modal queue`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1585
- State: open
- Labels: `installation`, `performance`, `scope: training`, `scope: inference`, `scope: attention`
- Author: `Mister-Raggs`
- Assignees: none
- Created: `2026-07-11T08:05:13Z`
- Updated: `2026-07-11T08:05:32Z`
- Repository: `/home/sandbox/FastVideo`
- Worktree: `/tmp/fastvideo-worktrees/issue_1585_fastcheck_lora_timeout`
- Branch: `issue/1585-fastcheck-lora-timeout`
- Base: `upstream/main` at `970409962f358afd529b969a378174c849665837`
- Handoff: `.agents/handoffs/issue-1585-handoff.md`
- Current stage: Stage 3 complete; draft PR message ready and awaiting explicit Stage 4 direction
- Stage 1 completed: `2026-07-12T05:33:57Z`
- Implementation begun: yes; user approved Approach A on 2026-07-12

## Stage 0

- Verified `gh` identity as `macthecadillac` on 2026-07-12 UTC.
- The filesystem sandbox failed with `bwrap: Can't mount devpts on /newroot/dev/pts: Permission denied`; required reads and GitHub operations were rerun with escalated permissions.
- Fetched `origin` and `upstream` refs.
- Searched local and remote branches for `1585`; none existed.
- Searched the active checkout and mainline trees for this handoff; none existed.
- Created this dedicated worktree and branch from current `upstream/main`.
- `origin/main` was at `19a51a1fe630bcbeaf9fb6d864ad5ed3f31a3536`; current `upstream/main` was newer at `970409962f358afd529b969a378174c849665837`.

## GitHub Context

### Issue body

- The `LoRA Training Tests` fastcheck lane is terminated by its Buildkite `timeout 15m` wrapper with exit status 124.
- Reporter observed the underlying Modal function succeed in about 9 minutes, but a cold-capacity run spent about 5.5 minutes queued before function start, plus setup time.
- The Modal function timeout is also 900 seconds. Because the Buildkite wrapper starts before Modal queue/setup while the function timeout starts later, the outer timeout can fire first.
- LoRA is described as the heaviest of eight lanes with the same 15-minute step budget because it runs five training steps plus validation generation for three prompts at 50 denoising steps.
- Suggested directions are either increasing only the LoRA fastcheck step budget, for example to 25 minutes, or trimming its smoke-test validation workload.
- Reported environment: Buildkite build 4289 on 2026-07-10, failing lane `test-tube-lora-training-tests`; sibling 15-minute training lanes passed.

### Comments

- Read the only issue comment from `Mister-Raggs`, which identifies issue #1584 as a separate CI-infrastructure problem surfaced in the same runs.

### Open PR check

- Retrieved the full open PR list and filtered it for `#1585`, `lora`, `timeout`, and branch names containing `1585`; no related open or draft PR was found.
- Exact searches for `"LoRA Training Tests"`, `LoRA training`, and `timeout Modal` found no related open PR.
- An exact open-issue search for `"exit status 124"` returned only issue #1585, so no duplicate was found.
- Read issue #1584 and its only comment. It concerns independent full-clone HTTP/2 failures before tests start and does not cover the outer-timeout failure in #1585.
- Read merged PR #985, `[ci] increase ssim and lora inference test timeout`. It is relevant precedent for a lane-specific timeout increase. That PR raised the LoRA inference Buildkite and Modal timeouts to the same 20-minute value. Gemini's review warned that equal inner and outer timeouts create ambiguous failures and recommended leaving the outer wrapper extra time; #1585 demonstrates that mechanism for LoRA training.
- Read the original merged LoRA training PR #576. It added the lane, its two-L40S Modal function, the five-step training smoke test, and the validation workload. No evidence in its body or reviews indicates that the 15-minute outer timeout was selected from measured queue-inclusive runtime.
- No PR draft status was changed.

## Investigation Log

- Read root `AGENTS.md`, `fastvideo/tests/AGENTS.md`, `docs/contributing/ci_architecture.md`, and `docs/contributing/testing.md`.
- Listed `.agents/skills`; no repository skill specifically covers Buildkite timeout changes. Listed `.agents/lessons`; none concerns Buildkite/Modal timeout budgeting.
- `.buildkite/pipeline.yml` lines 337-455 classify LoRA Training Tests as a path-filtered Full Suite lane, not Fastcheck. The issue's use of "fastcheck" is terminology drift, but its referenced label and timeout match the real Full Suite lane exactly.
- `.buildkite/pipeline.yml` lines 434-448 give only this Full Suite LoRA lane `timeout 15m`. Its direct-test counterpart at lines 161-173 uses `timeout 90m`, so `/test lora-training` does not reproduce the constrained Full Suite wrapper.
- `fastvideo/tests/modal/pr_test.py` lines 208-219 defines `run_training_lora_tests` on two L40S GPUs with `timeout=900`. The Modal timeout begins when the function starts, while GNU `timeout 15m` begins before authentication, scheduling, queueing, container setup, checkout, and the function runtime.
- `.buildkite/scripts/pr_test.sh` authenticates Modal and dispatches `TEST_TYPE=training_lora` to `run_training_lora_tests`; it adds no independent timeout handling or queue allowance.
- `fastvideo/tests/training/lora/test_lora_training.py` lines 18-116 runs five training steps and enables validation. The shared validation JSON contains three 480x832, 77-frame prompts, each specifying 50 inference steps. The workload has been materially unchanged since the lane was introduced.
- Git blame attributes the Full Suite `timeout 15m` and LoRA lane to the original July 2025 CI addition. No fetched history contains a previous LoRA training timeout increase.
- The prior merged timeout fix in PR #985 changed only the affected Buildkite and Modal values without adding a static timeout contract test. Current `fastvideo/tests/contract/` has no Buildkite timeout-budget test.
- The canonical CI docs already describe LoRA Training Tests as a Full Suite lane and do not promise per-lane timeout values, so no durable documentation is currently stale.
- No code, docs, tests, GitHub state, or Modal jobs have been changed or run.

## Current Hypothesis

- Confirmed: issue #1585 is valid. The outer Buildkite budget equals the maximum inner Modal function runtime even though the outer clock includes queue and setup phases that the inner clock excludes. The reporter's observed 5.5-minute queue plus roughly 9-minute successful function leaves effectively no margin under the current 15-minute wrapper.
- The concrete defect is limited to the path-filtered Full Suite LoRA Training Tests step. Direct tests already have a 90-minute outer budget, and no evidence shows the underlying LoRA training/validation assertions are incorrect or unnecessarily expensive.

## Merits And Scope

- Impact: unrelated PRs whose changed paths trigger Full Suite training lanes can receive an exit-124 failure despite a successful Modal function. This wastes GPU capacity, rerun time, and contributor attention and can block merge aggregation.
- Severity: moderate CI reliability defect. It does not affect shipped inference/training behavior or model quality.
- Reproducibility: supported by the reporter's repeated Buildkite observations and by the code's deterministic timeout relationship. Exact queue delay is capacity-dependent, so a local or direct Modal run need not reproduce it.
- The issue labels include product scopes inferred from body keywords, but the implementation ownership boundary is CI infrastructure: `.buildkite/pipeline.yml`.
- No model, pipeline, checkpoint, or numerical behavior changes are required. GPU memory impact is none.

## Approaches Considered

### A. Targeted outer-budget increase (recommended)

- Change only the Full Suite LoRA Training Tests command in `.buildkite/pipeline.yml` from `timeout 15m` to `timeout 25m`.
- Preserve the Modal function's 900-second execution ceiling and the existing training/validation workload.
- This gives queue, authentication, checkout, and container setup roughly 10 minutes beyond the function's maximum runtime, and about 10.5 minutes of margin around the reported 9-minute successful runtime plus 5.5-minute queue.
- Tradeoff: a genuinely hung Modal client can hold the Buildkite agent 10 minutes longer before SIGTERM. The increase is isolated to one affected lane.
- Tests/docs: no product test or doc change is necessary. Validate YAML/pre-commit and, when GitHub state permits, the actual Full Suite lane. A Modal-only run cannot validate the Buildkite-side clock.

### B. Targeted increase plus a static cross-file timeout contract

- Apply Approach A and add a CPU-only contract test that parses `.buildkite/pipeline.yml` plus the `run_training_lora_tests` decorator and asserts a minimum outer-over-inner margin.
- Benefit: prevents accidental restoration of equal budgets and makes the clock relationship explicit.
- Risk: adds a bespoke parser/assertion and couples CI config layout to a test for a one-line policy. Existing timeout changes have not used such tests, and the repository strongly discourages unused scaffolding. This is defensible only if maintainers want the margin to become a durable enforced policy.

### C. Reduce LoRA validation workload

- Reduce validation prompts or inference steps in `fastvideo/tests/training/lora/test_lora_training.py` or its shared validation JSON.
- Benefit: lowers Modal runtime and GPU cost.
- Risks: weakens an intentional end-to-end regression path, changes performance characteristics used by the W&B summary comparison, and still leaves equal inner/outer timeout clocks. It treats the symptom without fixing queue budgeting.

### D. Establish a systemic timeout policy for every lane

- Audit every Buildkite wrapper and Modal decorator, then standardize all outer budgets to exceed inner timeouts by a fixed allowance.
- Benefit: addresses the same class of risk across CI.
- Risks: much broader blast radius and CI cost, with no issue evidence that all lanes are failing. This exceeds #1585 and conflicts with the requirement to avoid speculative infrastructure.

## Recommended Plan

1. Re-check issue #1585, all comments, and open PRs immediately before editing.
2. Apply Approach A: change only the path-filtered Full Suite LoRA Training Tests wrapper from 15 to 25 minutes. Do not alter the 90-minute direct-test wrapper, the Modal 900-second function timeout, or the smoke-test workload.
3. Inspect the YAML diff and parse the pipeline config to catch syntax/structure errors.
4. Use `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker` for any repository validation required by project policy. Since the implementation changes only a Buildkite-side timeout, the meaningful runtime gate is a real Full Suite LoRA lane after a PR exists; an isolated Modal job can confirm the underlying test still completes but cannot exercise queue-inclusive Buildkite timeout behavior.
5. Commit the config change and updated handoff with GPG signing, push immediately, then run the required Stage 3 independent review/adjudication loop.
6. Run `pre-commit run --all-files` before presenting draft-PR readiness and again as the mandatory Stage 4 gate before any draft PR creation.
7. Do not open a PR until the user explicitly requests Stage 4. Any new PR must be draft-only; existing PR draft status must never be changed.

## Validation Plan

- Static: confirm `.buildkite/pipeline.yml` parses successfully and that the only functional diff is the LoRA Full Suite wrapper changing from `15m` to `25m`.
- Pre-commit: run the project hook chain, including YAML/format checks, before draft-PR readiness; final mandatory command is `pre-commit run --all-files`.
- Runtime: if feasible before PR creation, use the approved Modal launcher to confirm the unchanged LoRA test path remains healthy. This is supporting evidence, not proof of the outer timeout fix.
- End-to-end: after explicit Stage 4 PR creation, run or observe the path-filtered Full Suite LoRA Training Tests lane. Pass criteria are successful completion without exit 124, with Buildkite wall time allowed to exceed 15 minutes while the Modal function remains within its own 900-second ceiling.
- Wan T2V SSIM is not relevant because no model/pipeline numerics or generated output changes.

## Open Questions

- None. The user approved the recommended minimal 25-minute outer-budget change and directed Stage 2 to begin.

## Stage 2

- Re-verified `gh` identity as `macthecadillac` before editing.
- Re-read issue #1585 and its only comment; issue state and content were unchanged.
- Re-checked the open PR list; no related issue #1585, LoRA timeout, or matching branch PR exists.
- Selected scope: change only the path-filtered Full Suite LoRA Training Tests wrapper from `timeout 15m` to `timeout 25m`.
- Explicit exclusions: no static contract test, workload reduction, Modal timeout change, direct-test timeout change, documentation change, or broad CI timeout audit.
- Implemented `.buildkite/pipeline.yml`: changed only the Full Suite LoRA Training Tests wrapper from `timeout 15m` to `timeout 25m`.
- Diff inspection confirmed the 90-minute direct-test wrapper, Modal `timeout=900`, LoRA smoke workload, tests, and docs are unchanged.
- First Modal attempt used the `interleavethinker` launcher with the local pipeline patch, app `ap-xCNC5eCHkOQk2I7qHtXgvt`. The attached client stopped the app during repository cloning, before assertions ran; this attempt provided no validation result.
- Retried detached through the `interleavethinker` launcher, app `ap-yE0Sn5gHPdlCoaTinP7dns`, FunctionCall `fc-01KXAGNTNXY8HY0165T4REYCDJ`.
- Detached command parsed `.buildkite/pipeline.yml` with PyYAML and asserted the presence of `timeout 25m` and unchanged `timeout 90m` LoRA commands after applying the 547-byte local patch.
- Result: passed; logs printed `yaml parsed` and `validated LoRA timeout entries`.
- The full LoRA GPU workload was not rerun because no runtime/model/test code changed and a Modal-only run cannot exercise the Buildkite outer clock. The definitive end-to-end check remains the Full Suite lane after PR creation.
- `pre-commit run --all-files` remains pending for the Stage 3 readiness gate.
- Signed implementation commit: `82ebb4237afa0a0eb2777c261c5d37be5e90748c` (`[ci]: extend LoRA training CI timeout (#1585)`).
- GPG verification: good signature from `Mac Lee <macthecadillac@gmail.com>`, signing subkey `C943F92E5C32D887`.
- Push: successful to `origin/issue/1585-fastcheck-lora-timeout`.

## Stage 3

- Reviewer: `/root/review_issue_1585_round1`, prompted to use `review-code` on `macthecadillac/FastVideo` branch `issue/1585-fastcheck-lora-timeout` for `hao-ai-lab/FastVideo#1585` without modifying code or GitHub state.
- Reviewed branch head: `a20a36f66` against upstream main `970409962`.
- Finding: no actionable findings. The reviewer confirmed the one-line change directly addresses #1585, preserves the inner Modal timeout/direct wrapper/workload, and is not over-engineered.
- Residual risk: the actual Buildkite Full Suite LoRA lane has not run, so queue-inclusive completion under 25 minutes is unverified; unusually long capacity queues could still exceed the new allowance.
- Adjudicator/fixer: not spawned because there were no actionable findings to adjudicate.
- Direct `pre-commit run --all-files` could not start because `pre-commit` was not installed on the host PATH.
- Fallback `uv run --no-project --with pre-commit pre-commit run --all-files` ran the full hook list. All hooks passed except mypy, which failed before analysis with `issue-1585-fastcheck-lora-timeout is not a valid Python package name`.
- The mypy failure was caused solely by the hyphenated worktree basename, not repository content.
- Moved the same worktree with `git worktree move` to `/tmp/fastvideo-worktrees/issue_1585_fastcheck_lora_timeout`; branch and files are unchanged.

- Reran `uv run --no-project --with pre-commit pre-commit run --all-files` from the underscore-only worktree.
- Result: passed. YAPF, ruff, codespell, PyMarkdown, actionlint, mypy, filename checks, and suggestion all passed.
- The Stage 3 pre-commit readiness gate is cleared. It must run again as the final Stage 4 gate before any draft PR creation.

## Draft PR Message

Title: `[ci]: extend LoRA training CI timeout`

```markdown
## Summary

- increase the path-filtered Full Suite LoRA Training Tests Buildkite wrapper from 15 to 25 minutes
- preserve the Modal function's 900-second runtime ceiling, the 90-minute direct-test wrapper, and the existing LoRA training/validation workload
- provide queue, authentication, checkout, and container-setup headroom without weakening regression coverage

Fixes #1585.

## Validation

- Modal L40S static pipeline validation through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`
  - App: https://modal.com/apps/hao-ai-lab/main/ap-yE0Sn5gHPdlCoaTinP7dns
  - applied the local `.buildkite/pipeline.yml` patch
  - parsed the pipeline with PyYAML
  - confirmed the Full Suite LoRA timeout is 25 minutes and the direct-test timeout remains 90 minutes
- `uv run --no-project --with pre-commit pre-commit run --all-files`
  - all hooks passed

## Review Loop

- a fresh `review-code` sub-agent reviewed `macthecadillac/FastVideo` branch `issue/1585-fastcheck-lora-timeout` for issue #1585
- no actionable findings were reported, so no adjudicator/fixer sub-agent was needed

## GPU Memory Impact

None. This changes only the Buildkite wall-clock allowance for one existing CI lane.

## Remaining Risk

The actual queue-inclusive Full Suite LoRA lane can only be validated after a PR exists. Capacity-dependent queue delays longer than the new 10-minute allowance could still exceed the outer timeout.

# Checklist
- [x] I ran pre-commit run --all-files and fixed all issues
- [ ] I added or updated tests for my changes
- [x] I updated documentation if needed
- [x] I considered GPU memory impact of my changes
```

## Stage 1 Persistence

- Signed handoff commit: `77fa5d0605b0d1bd11366469add5eb0e83399656` (`[ci]: investigate LoRA training timeout (#1585)`).
- GPG verification: good signature from `Mac Lee <macthecadillac@gmail.com>`, signing subkey `C943F92E5C32D887`.
- Push: successful to `origin/issue/1585-fastcheck-lora-timeout`; the branch now tracks the macthecadillac fork.
- No PR was opened.

## Next Steps

1. Rerun pre-commit after this final handoff edit, then commit with GPG signing and push.
2. Present the complete Stage 3 result and draft PR message to the user without opening a PR.
