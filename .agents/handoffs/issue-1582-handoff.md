# Issue 1582 Handoff

## Target And Workspace

- Issue: #1582, `[ci] Perf CI local tests cleanup`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1582
- Repository: `hao-ai-lab/FastVideo`
- Branch: `issue/1582-perf-ci-local-tests-cleanup`
- Worktree: `/tmp/fastvideo-worktrees/issue-1582-perf-ci-local-tests-cleanup`
- Handoff: `.agents/handoffs/issue-1582-handoff.md`
- Started: `2026-07-13T03:13:17Z`
- Last updated: `2026-07-13T03:28:26Z`
- Current stage: Stage 1 complete, awaiting user guidance
- Implementation begun: no

## Stage 0

- Verified GitHub identity with `gh api user --jq .login`: `macthecadillac`.
- Initial `git fetch origin --prune` failed because the host SSH config contains the unsupported Linux option `UseKeychain`; no authentication change was attempted.
- Refreshed authoritative code successfully with `git fetch upstream --prune` over HTTPS.
- No local, `macthecadillac/FastVideo`, or `hao-ai-lab/FastVideo` branch containing `1582` was found.
- No existing `.agents/handoffs/issue-1582-handoff.md` or open PR for #1582 was found.
- Created this dedicated branch/worktree from `upstream/main` at `970409962`.

## GitHub Snapshot

- State: open
- Author: `Satyam-53`
- Assignee: `macthecadillac`
- Labels: none
- Created: `2026-07-11T01:27:22Z`
- Updated: `2026-07-11T11:58:11Z`
- Request: move only performance-CI behavior-correctness tests from `fastvideo/tests/performance` to `tests/local_tests/performance`, leaving tests directly used by PR gating or manual `pr_test.py` runs in place.
- Related ticket cited in the body: #1527.
- Comment by `hiSandog`: proposes an explicit pytest marker or manifest, intentional gate selection in `pr_test.py`, and a CI assertion preventing lost performance-gate discovery. This is a hypothesis to evaluate, not an approved implementation.

## Parent And Related Work

- #1582 has no formal GitHub parent issue (`GET /issues/1582/parent` returned 404).
- #1527 is an open Phase 1 performance-regression tracking epic. Its formal children are #1529-#1536.
- Child states observed at investigation start: #1529 closed, #1530 closed, #1531 closed, #1532 open, #1533 closed, #1534 open, #1535 open, #1536 open.
- No open PR directly references #1582 or its title.
- #1529 / PR #1544 (merged) added config-validation behavior tests and modified the actual benchmark test.
- #1530 / PR #1546 (merged) added identity helpers/tests plus comparator and dashboard behavior coverage in the same directory.
- #1531 / PR #1551 (merged) added result-schema behavior coverage and expanded comparator/dashboard tests in the same directory.
- #1533 / PR #1545 (merged) expanded comparator/dashboard policy behavior tests in the same directory.
- #1532 has active ready-for-review PR #1560. It modifies `test_compare_baseline_policy.py`, adds `test_inference_performance_identity.py`, and changes gate/helper scripts. Its latest direct performance check is failing. Draft status was observed only and not changed.
- #1534 is deliberately parked until v2 cohort history accumulates; it has no open PR and is not a practical prerequisite for cleanup.
- #1535 has no open PR; its remaining legacy-dashboard work is likely to add or modify dashboard behavior tests.
- #1536 has no open PR and is documentation-focused.
- #1573 is a conceptual Phase 2 roadmap with no formal children. It does not make #1582 a formal dependency.

## Investigation Log

- Stage 1 is analysis-only. No code, test, documentation, issue, label, PR, or Modal changes have been made.
- Read root `AGENTS.md`, `fastvideo/tests/AGENTS.md`, the performance/CI/testing contributor guides, and `tests/local_tests/README.md`.
- No performance-specific durable lesson exists under `.agents/lessons/`.
- `run_performance_tests` invokes `pytest ./fastvideo/tests/performance -vs`, then `compare_baseline.py` and `dashboard.py`; directory collection therefore runs the GPU gate and every CPU behavior test together.
- `run_unit_test` explicitly lists `fastvideo/tests/...` paths and does not collect `tests/local_tests`.
- The Buildkite performance filter watches `fastvideo/tests/performance/**`; no CI filter watches `tests/local_tests/**`.
- `tests/local_tests/README.md` says that tree is skipped in CI. A plain move there removes behavior tests from performance and unit CI, although local/root pytest discovery can still run them.
- The marker/manifest comment is technically plausible but broader than a filesystem move. A marker alone does not change collection; CI routing and a discovery-contract test would also be needed.
- The only actual parameterized benchmark gate is `test_inference_performance.py::test_inference_performance`. Other current `test_*.py` modules validate config parsing, comparator policy, dashboard API/Plotly/service behavior, identity/fingerprinting, component-time extraction, and result schema.
- Current main contains 91 behavior-test functions and one benchmark-test function in the performance directory. There is one benchmark JSON config, so the gate currently has one parameter case.
- PR #1560 adds three producer/identity behavior tests and 584 lines of comparator-policy test coverage in the same files/surface. It therefore creates an ordering conflict, not a feature prerequisite.
- Duplicate search found no other issue or open PR for this cleanup.

## Current Hypothesis

- There is no hard runtime or feature dependency on completing all #1527 children. There is a concrete sequencing conflict with active PR #1560/#1532 because it changes and adds files #1582 would classify or move. Waiting for every child is unjustified: #1534 is parked and #1535/#1536 can adopt the new layout after cleanup.

## Merits And Scope

- The report is valid. CPU behavior tests are coupled to an expensive two-L40S performance lane solely through directory discovery.
- A behavior-test failure currently prevents PR/direct runs from reaching baseline comparison and artifact generation, even when the measured benchmark itself is healthy.
- The move has no model, generated-media, runtime-performance, or GPU-memory behavior impact. It changes validation ownership and discovery.
- The issue body is ambiguous about retained CI coverage. The requested destination is documented as local-only and skipped in CI, while the commenter proposes preserving an executable CI boundary.

## Approaches

1. Literal minimal move: after #1560 settles, move all behavior-only test modules to `tests/local_tests/performance/`, keep only `test_inference_performance.py` and the scripts/helpers in `fastvideo/tests/performance/`, and make `pr_test.py` invoke the exact benchmark file. This matches the requested path and cleanly removes behavior tests from the performance lane, but drops their current CI coverage.
2. Move and retain CI coverage: perform the same move and exact benchmark-file selection, then explicitly collect `tests/local_tests/performance/` in the unit lane and add its Buildkite path filter. Update `tests/local_tests/README.md` to document this narrow exception. This preserves regression coverage at lower GPU cost but makes the nominally local-only tree partially CI-backed.
3. Marker/manifest system: add a performance-gate marker or manifest and make all lanes select by it. This is more extensible, but a marker alone does not solve routing and the current suite has only one gate file. It adds unnecessary machinery unless maintainers expect multiple mixed-purpose gate files soon.
4. Wait for every #1527 child and clean up once: minimizes repeated file moves, but #1534 is intentionally parked for weeks and #1535/#1536 have no active PR. This unnecessarily blocks a useful CI cleanup.

## Recommendation

- Do not wait for all #1527 sub-issues.
- Sequence #1582 immediately after #1532/PR #1560 lands, or explicitly base/rebase the cleanup on #1560 if maintainers want it sooner. This avoids losing or misplacing #1560's new behavior tests and reduces merge conflicts.
- Prefer approach 2 if maintainers intend comparator/identity/dashboard regressions to stay CI-protected. Prefer approach 1 only if the explicit goal is to make them genuinely local-only.
- Use an exact file path in `run_performance_tests` instead of a new marker/manifest: `pytest ./fastvideo/tests/performance/test_inference_performance.py -vs`. The current gate boundary is one file, so this is direct and durable.
- Reclassify the directory again after rebasing because remaining #1527 work may have added tests. Move only behavior modules; keep `test_inference_performance.py`, `compare_baseline.py`, `dashboard.py`, `identity.py`, and runtime scripts used by the gate.
- Update `tests/local_tests/README.md`, `docs/contributing/testing.md`, `docs/contributing/performance_benchmarks.md`, and relevant test-layout guidance so commands and CI status match the chosen approach.

## Validation And Next Steps

- No tests or Modal jobs run in Stage 1.
- Future implementation, if approved, must use Modal via `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`; local tests are prohibited.
- A future draft PR requires `pre-commit run --all-files`, a signed commit/push cycle, Stage 3 review/adjudication, explicit user approval for PR creation, and preservation of any existing PR draft status.
- Planned Modal validation: run the moved behavior suite; collect the exact gate file and confirm only benchmark config cases are selected; run the real two-L40S benchmark path and verify raw results, comparator summary, normalized artifacts, and dashboard behavior remain intact.
- If approach 2 is selected, also validate the production unit-lane command collects the moved suite and add contract/path-filter coverage showing a PR touching only `tests/local_tests/performance/**` triggers that lane.
- Documentation validation must confirm local commands no longer imply that directory-wide performance pytest mixes behavior tests with benchmarks.
- GPU memory impact: none expected; the actual benchmark still requests two L40S GPUs, while behavior tests move to a separate or local path.
- Mandatory future PR gate: `pre-commit run --all-files`; not run during Stage 1. The gate is not cleared.
- Open decision: should moved performance behavior tests remain CI-backed through the unit lane (approach 2), or become truly local-only as the destination README currently specifies (approach 1)?

## Durability

- Stage 1 findings were committed with GPG signing in `1d5a69596` and pushed to `origin/issue/1582-perf-ci-local-tests-cleanup`.
- Push verified at `2026-07-13T03:35:35Z`. No PR was created and no GitHub issue state was modified.
