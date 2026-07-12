# Issue 1584 Handoff

## Status

- Issue: `hao-ai-lab/FastVideo#1584`
- Title: `[CI] Modal checkout: full-clone HTTP/2 disconnects redden ~1 lane per run (exit 128)`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1584
- Stage: Stage 2, implementation
- Implementation begun: yes, 2026-07-12 UTC
- Handoff: `.agents/handoffs/issue-1584-handoff.md`
- Worktree: `/tmp/fastvideo-worktrees/issue-1584-modal-checkout`
- Branch: `issue/1584-modal-checkout`
- Base: `upstream/main` at `970409962f358afd529b969a378174c849665837`
- Current upstream: `upstream/main` at `970409962f358afd529b969a378174c849665837`
- Started: 2026-07-12 UTC

## GitHub State

- Verified `gh` identity: `macthecadillac`.
- Issue state: open.
- Labels: `scope: attention` (appears unrelated to the reported CI checkout failure; investigate).
- Assignees: none.
- Author: `Mister-Raggs`.
- Created: 2026-07-11T08:05:12Z.
- Updated: 2026-07-11T08:05:30Z.
- One comment links separate issue `#1585`; no fix is proposed in the comment.
- Matching local/remote issue branches found before creation: none.
- Matching handoffs found before creation: none.
- Related open PR review: complete as of 2026-07-12 UTC. No open PR references
  `#1584`, its title, `early EOF`, or an equivalent checkout-hardening change.
- Open PRs `#1562`, `#1581`, and `#1389` modify
  `fastvideo/tests/modal/pr_test.py`, but none changes checkout behavior.
  `#1562` restructures setup/test execution for kernel caching and adds
  `fastvideo/tests/modal/test_pr_test.py`; `#1581` adds pytest-level transient
  reruns; `#1389` adds an H100 performance entrypoint. All three are
  ready-for-review (`isDraft=false`); their draft statuses were not changed.
- Merged PR `#1202` only preserves the PR number for fork PRs and intentionally
  leaves `pr_test.py` unchanged.
- Linked issue `#1585` concerns a separate Buildkite timeout budget, not this
  checkout root cause.

## Initial Report

The report attributes intermittent pre-test Modal lane failures to the checkout in
`fastvideo/tests/modal/pr_test.py`: every lane performs a full-history clone over default HTTP/2, then fetches the PR ref, with no transfer reduction or retry. The reported signature is Git HTTP/2 cancellation followed by early EOF and invalid index-pack output.

Reporter-suggested directions, to evaluate rather than apply during Stage 1:

1. Force Git HTTP/1.1.
2. Use shallow and filtered cloning plus a shallow PR-ref fetch.
3. Increase `http.postBuffer` as secondary mitigation.
4. Add bounded retry as a backstop.

## Investigation Log

- Read the complete issue body and its sole comment through `gh`.
- Fetched `origin` and authoritative `upstream` refs.
- Read root `AGENTS.md`, `fastvideo/tests/AGENTS.md`,
  `docs/contributing/ci_architecture.md`, and
  `docs/contributing/testing.md`; searched `.agents/lessons/` and found no
  checkout-specific lesson.
- Searched the full open PR list plus GitHub PR/issue text for `1584`,
  `pr_test.py`, `early EOF`, `HTTP/2`, and `Modal checkout`.
- Inspected current `upstream/main` checkout paths:
  - `fastvideo/tests/modal/pr_test.py:94-158` performs one unfiltered,
    full-history `git clone`, then an unshallow PR-ref fetch or direct checkout,
    with no retry and default Git HTTP behavior.
  - `fastvideo/tests/modal/ssim_test.py:456-508` has a three-attempt shell
    `git_retry` for clone/fetch/submodules, but neither shallow/filter options
    nor HTTP/1.1.
  - `fastvideo/tests/modal/launch_l40s_job.py:280-315` retries clone three times,
    cleans the destination between attempts, and forces clone HTTP/1.1, but
    remains a separate launcher-specific implementation.
- Quantified transfer pressure: the local shared Git object packs total about
  504 MiB; the `upstream/main` working tree is about 167.5 MiB, of which assets
  are about 134.1 MiB. Four current assets alone are roughly 23-31 MiB each.
  Shallow history therefore removes substantial avoidable transfer, while a
  blobless clone still downloads current-tree blobs when checkout materializes
  the full worktree.
- Inspected historical checkout-hardening commits on the former PR `#1471`
  branch: `33ce65c11`, `532b09fb7`, `1d08d28ec`, and removal commit
  `00f7be9be`. The sequence added a blobless no-checkout clone and shallow
  PR/direct checkout retries, fixed a composed-shell syntax error, extended
  retry to direct checkouts, then was explicitly removed as unrelated bundled
  CI work. PR discussion says the rewrite should be a separate PR; no review
  rejected the goal. The historical version did not retry the initial clone,
  force HTTP/1.1, or use a shallow initial clone, so it does not fully resolve
  the reported failure.
- Inspected overlapping open-PR patches. A future implementation should rebase
  onto current `upstream/main` first and keep checkout construction isolated
  enough to resolve likely `pr_test.py` conflicts cleanly.
- No code, documentation, GitHub state, or Modal jobs have been changed or run.

## Merit Assessment

- Confirmed defect: yes. The exact vulnerable command exists on current
  `upstream/main`, runs once per Buildkite/Modal lane, and fails before tests.
- Reporter evidence: a canonical Git transport failure signature from build
  4288, recurrence on rerun, and 16 sibling lanes passing in the same run.
- Impact: broad CI reliability across unrelated PRs; about 18 independent
  transfers amplify even a modest per-clone failure rate.
- Scope: CI infrastructure in `fastvideo/tests/modal/pr_test.py`, not attention.
  The current `scope: attention` issue label appears misclassified, but Stage 1
  made no label change.
- The HTTP/1.1, shallow/filter, and bounded-retry directions are supported by
  current code and existing in-repo patterns. Increasing `http.postBuffer` is
  not recommended: it controls buffered HTTP POST/upload behavior and does not
  address a clone response stream dropping mid-transfer.

## Candidate Approaches

### A. Minimal transport mitigation

Force HTTP/1.1 for the current clone and add a three-attempt clone retry with
destination cleanup. This directly targets the observed failure with the
smallest patch, but preserves the unnecessary full-history transfer and leaves
the PR fetch/direct checkout as single-attempt network/materialization steps.

### B. Targeted hardened checkout (recommended)

Keep the existing `run_test_command` ownership boundary, but generate a
three-attempt setup that:

1. clones with repository-local HTTP/1.1, `--depth=1`, `--filter=blob:none`,
   and `--no-checkout`, cleaning `/FastVideo` before clone retries;
2. shallow-fetches the exact PR ref or direct commit with bounded retries;
3. detach-checks out `FETCH_HEAD` with bounded retries so lazy blob transfer is
   also covered;
4. updates submodules using the repository's HTTP/1.1 setting;
5. validates/quotes Buildkite-derived values rather than interpolating raw
   strings into shell.

Add focused command-construction/behavior tests, including `bash -n` coverage
for PR/direct and kernel/no-kernel compositions because the historical patch
regressed that exact boundary. This materially reduces the transfer, targets
HTTP/2, and adds a bounded backstop without creating shared infrastructure.

### C. Shared Modal checkout utility

Extract one checkout helper for `pr_test.py`, `ssim_test.py`, and
`launch_l40s_job.py`. This would unify policy, but changes three independently
executed Modal entrypoints and expands the regression surface beyond the
reported failing path. Current implementations have different workspace reuse
and execution needs, so this is not justified for `#1584` without a broader
request.

### D. Sparse checkout

Exclude large assets per lane in addition to shallow/filter cloning. This can
reduce the unavoidable 134 MiB current-tree asset transfer, but every lane has
different file needs and some tests consume tracked assets. The configuration
and maintenance cost are too high for the current defect.

## Recommended Plan

1. Rebase the issue branch onto current `upstream/main` before implementation.
2. Implement Approach B only in `fastvideo/tests/modal/pr_test.py`, preserving
   fork PR refs and direct-commit behavior from PR `#1202`.
3. Add focused tests under `fastvideo/tests/modal/` (or extend
   `test_pr_test.py` if open PR `#1562` has merged) for retry bounds, cleanup,
   HTTP/1.1, shallow/blobless flags, exact PR/direct target fetch, detached
   checkout, quoting/validation, and syntactically valid composed shell.
4. Documentation impact is expected to be none: the durable CI architecture
   and lane ownership do not change. Add a short durable note to
   `docs/contributing/ci_architecture.md` only if implementation introduces a
   user-relevant checkout policy that operators need for diagnosis.
5. Validate through the `interleavethinker` copy of
   `fastvideo/tests/modal/launch_l40s_job.py`: run the focused tests on Modal
   L40S, then a checkout/setup smoke that exercises both PR-ref and direct
   commit command variants without running an expensive model suite.
6. Run `pre-commit run --all-files` as the mandatory pre-PR gate. Do not run
   project tests locally.
7. Commit with GPG signing and push immediately, then run the required Stage 3
   independent review/adjudication loop.
8. If later explicitly asked to create a PR, re-check GitHub state, retire this
   handoff, create only a draft PR, add `macthecadillac` to the issue assignees
   without changing PR assignees, and never alter an existing PR's draft state.

## Validation Pass Criteria

- Both PR-ref and direct-commit paths fetch and detach-checkout the exact target.
- A simulated transient failure succeeds on a later attempt; three failures
  preserve a nonzero exit; partial clone state cannot poison a retry.
- Generated/composed shell passes syntax validation for all relevant variants.
- Modal L40S focused tests pass through the mandated launcher.
- A Modal checkout smoke shows shallow/blobless HTTP/1.1 commands complete and
  the checked-out SHA matches the requested ref/commit.
- `pre-commit run --all-files` passes before any draft PR is created.

## Open Questions

- None. The user approved the recommended targeted hardened checkout
  (Approach B) on 2026-07-12 UTC.

## Stage 2 Log

- Re-checked the issue body, all comments, and the full open PR list before
  editing. GitHub state and overlap findings were unchanged.
- Rebased onto `upstream/main` at `970409962f358afd529b969a378174c849665837`.
- Rewritten signed handoff commit: `69cff397b`; force-with-lease push succeeded.
- Implemented the approved checkout hardening in
  `fastvideo/tests/modal/pr_test.py`:
  - structured subprocess arguments replace shell-interpolated Git commands;
  - clone uses HTTP/1.1, depth 1, blob filtering, and no initial checkout;
  - clone/fetch/checkout/submodule commands receive three bounded attempts;
  - clone retries remove partial destination state before each attempt;
  - PR refs and validated direct commit SHAs are fetched shallowly and checked
    out detached.
- Added `fastvideo/tests/modal/test_pr_test.py` with network-free coverage for
  clone recovery, bounded exhaustion, cleanup, PR/direct target selection,
  invalid Buildkite metadata, command flags, integration ordering, and shell
  syntax for kernel/no-kernel setup variants.
- Documentation remains unchanged because CI ownership and user-facing
  operation are unchanged.
- Modal validation attempts:
  1. Run `ap-h1n0YPwdKyecf9Tum5s916` failed before pytest because
     `--install-extra test` rebuilt the published `fastvideo-kernel==0.3.2`
     source distribution, whose build could not find CUTLASS headers. This is
     unrelated to the patch.
  2. Run `ap-68WngBzUp8FpWiUyn6nuwB` failed before pytest because the
     no-install command did not source the container environment and therefore
     could not find `uv`.
  3. Run `ap-zxlDxnUGusJblcrZ9BLvy8` reached the focused suite: 10 tests
     passed and one test-only assertion failed because an empty string is
     contained in every Python string. Production checkout cases passed.
- Corrected the no-install assertion to check explicitly that
  `uv pip install -e` is absent.
- Focused Modal L40S rerun `ap-aN9NR47J2GNaavv3KaLcWH`: passed,
  `11 passed in 0.06s`.
- Live Modal L40S transfer smoke `ap-iuuYuWaVdIXfQahs54XpnU`: passed using
  the exact production clone/fetch/checkout flags.
  - Direct commit path checked out
    `69cff397bce2699b1c7d25c018da7c07716eed36` from the pushed fork.
  - PR path fetched `refs/pull/1581/head` from
    `hao-ai-lab/FastVideo` and checked out
    `603c7b0d1c662f3d17df749ccec332568297a839`, matching `ls-remote`.
- Stage 2 validation is complete. No GPU model execution was needed, and the
  change has no GPU memory impact.
- Changed-file pre-commit command:
  `uv run --no-project --with pre-commit pre-commit run --files .agents/handoffs/issue-1584-handoff.md fastvideo/tests/modal/pr_test.py fastvideo/tests/modal/test_pr_test.py`.
  Result: filename check passed; all other hooks reported no files to check,
  consistent with the repository's deliberate `fastvideo/tests/` and handoff
  exclusions.
