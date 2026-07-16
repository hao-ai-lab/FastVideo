# Issue 1535 Handoff

## Current State

- Issue: #1535, "[ci] Dashboard Grouping And Legacy V1"
- Issue URL: https://github.com/hao-ai-lab/FastVideo/issues/1535
- Repo: hao-ai-lab/FastVideo
- Worktree: /tmp/fastvideo-worktrees/issue-1535-ci-dashboard-grouping-legacy-v1
- Branch: issue/1535-ci-dashboard-grouping-legacy-v1
- Base: upstream/main at 9d909f5f0457ac91f489d5fc8000931f042b72ce
- Handoff path: .agents/handoffs/issue-1535-handoff.md
- Current stage: Stage 1 complete - awaiting user guidance
- Implementation begun: no
- Last updated: 2026-07-16 10:05:54 UTC

## Authentication And Sandbox Notes

- Verified `gh` identity with `gh api user --jq .login`: `macthecadillac`.
- Fetched `origin` (macthecadillac fork) and `upstream` (hao-ai-lab/FastVideo).
- The checkout also has a `SolitaryThinker` remote, but it was not used.
- `git fetch origin --prune` completed with known_hosts cross-device-link warnings but exit code 0.

## Stage 0 Resume Or Start Notes

- Searched local and fetched refs for branches containing `1535`: no matches.
- Checked active checkout for `.agents/handoffs/issue-1535-handoff.md`: none.
- Created dedicated worktree and branch:
  - `git worktree add -b issue/1535-ci-dashboard-grouping-legacy-v1 /tmp/fastvideo-worktrees/issue-1535-ci-dashboard-grouping-legacy-v1 upstream/main`
- Root `AGENTS.md` read in the issue worktree before writing this handoff.
- Searched `.agents/lessons` for `dashboard|grouping|legacy|ci|v1|performance`; only model-port and Dreamverse CI lessons matched, with no directly applicable lesson identified yet.

### 2026-07-16 Resume Refresh

- Re-fetched `origin` and `upstream`; verified the local issue branch matches
  `origin/issue/1535-ci-dashboard-grouping-legacy-v1` exactly.
- Recreated the missing dedicated worktree at
  `/tmp/fastvideo-worktrees/issue-1535-ci-dashboard-grouping-legacy-v1` and
  resumed from this tracked handoff at Stage 1 complete / awaiting guidance.
- Re-verified `gh` identity as `macthecadillac`.
- Re-read issue #1535 and its only comment. The issue remains open, is now
  assigned to `macthecadillac`, and has no new commenter-proposed fix.
- Re-checked open PRs and searched for direct references to #1535; no open PR
  currently covers or closes the issue, and no PR exists for this branch.
- Dependency status changed materially since the original investigation:
  - #1546 merged on 2026-07-07 as `1ee11e08d` and closed #1530.
  - #1551 merged on 2026-07-09 as `afb4f7d3c` and closed #1531.
  - #1560 merged on 2026-07-15 as `a25385614` and closed #1532.
- Therefore the prior wait-or-stack question is obsolete. Stage 2 can rebase
  this branch onto current `upstream/main` and implement only the legacy/v1
  dashboard gap after confirming the exact behavior of the merged stack.

## GitHub Context

- Issue metadata from full `gh issue view 1535`:
  - Number: 1535
  - State: OPEN
  - Title: [ci] Dashboard Grouping And Legacy V1
  - URL: https://github.com/hao-ai-lab/FastVideo/issues/1535
  - Author: Satyam-53
  - Created: 2026-07-02T22:39:07Z
  - Updated: 2026-07-05T22:54:26Z
  - Labels: `scope: attention`, `scope: docs`
  - Assignees: none
  - Milestone: none
- Issue body summary:
  - Update performance dashboard so v2 records group by comparable identity dimensions while v1 records remain visible as historical context.
  - Desired grouping dimensions: workload, variant, benchmark version, hardware profile, software profile, and recipe fingerprint or label where useful.
  - V1 records must remain visible but not be aggregated with v2 records unless explicitly migrated.
  - Acceptance criteria: separate variants/software cohorts; do not merge v1/v2 in regression views; keep historical v1 charts readable; make calibration/mismatch/regression statuses clear; tests cover grouping by variant and software profile.
  - Relevant files listed by issue: `fastvideo/tests/performance/dashboard.py`, `fastvideo/performance_dashboard/`, dashboard API/service tests, and performance benchmark docs.
  - Parent RFC: #1374.
- Issue comments reviewed:
  - SolitaryThinker, 2026-07-05T22:54:26Z: says the owner-approved epic plan is set; this phase is first-refusal for `macthecadillac`; v2 cohort grouping already shipped with #1546, leaving the legacy/v1 grouping story. Proposed scope: group pre-v2 records under `(model_id, gpu_type)` buckets in the dashboard, visually separated from v2 cohorts, no comparator changes. Records without identity fields should skip rolling comparison per owner policy.
  - Evaluation of proposed fix: current `upstream/main` does not have the #1546 dashboard cohort grouping. The #1546 branch adds cohort grouping by model/GPU plus the v2 identity fields, and legacy records fall into the same model/GPU group with empty identity fields displayed as `legacy`. It does not yet incorporate #1532's explicit comparator status vocabulary in the dashboard. The comment's "no comparator changes" direction looks right for #1535 if stacked after the #1530/#1531/#1532 work is reconciled.
- Parent/related issues read:
  - #1374 RFC recommends exact comparable identity: `workload_id`, `variant_id`, `benchmark_version`, `hardware_profile_id`, `software_profile_id`, `recipe_fingerprint`; v1 records remain historical and are not migrated immediately.
  - #1530 covers deterministic recipe fingerprints and hardware/software profile IDs. Issue OPEN, assigned to `macthecadillac`.
  - #1531 covers v2 normalized record emission and explicitly says v1 records stay readable for historical charts. Issue OPEN, assigned to `macthecadillac`.
  - #1532 covers exact-identity comparator statuses. Issue OPEN, assigned to `macthecadillac`.
- Related PRs checked from open PR list and `gh pr view`:
  - #1546 `[ci]: add performance fingerprint cohorts`, branch `issue/1530-fingerprinting-cohorts`, ready-for-review (`isDraft=false`), state OPEN, mergeStateStatus BLOCKED, closes #1530. It touches dashboard service, frontend, Plotly dashboard, docs, identity helpers, and tests. It groups dashboard records by full comparison cohort and labels missing cohort values as `legacy`.
  - #1551 `[ci]: emit v2 performance result schema`, branch `issue/1531-v2-result-emission`, ready-for-review (`isDraft=false`), state OPEN, mergeStateStatus DIRTY, closes #1531, depends on #1546. It preserves v2 fields in normalized records while keeping legacy v1 records readable.
  - #1560 `[ci] Add exact identity performance statuses`, branch `issue/1532-exact-identity-statuses`, ready-for-review (`isDraft=false`), state OPEN, mergeStateStatus BLOCKED, closes #1532. It introduces `comparison_status` values `PASS`, `REGRESSION`, `CALIBRATION_NEEDED`, `RECIPE_MISMATCH`, and `INFRA_ERROR`, with `comparison_status_reason`. It does not include the #1546 dashboard changes (`origin/issue/1530-fingerprinting-cohorts` is not an ancestor of `origin/issue/1532-exact-identity-statuses`).
  - Search for PRs directly referencing `1535`: no direct matches.
  - Search for `performance dashboard grouping`: #1551, closed #1310, and merged #1292 appeared; no PR closes #1535.
  - No PR draft status was changed.

## Investigation Notes

- No code or docs implementation changes have been made.
- Root `AGENTS.md` read. No directory-specific `AGENTS.md` exists under `fastvideo/tests/performance`, `fastvideo/performance_dashboard`, or `apps/performance_dashboard`.
- Current `upstream/main` code findings:
  - `fastvideo/tests/performance/dashboard.py` groups the Plotly dashboard only by `["model_id", "gpu_type"]` at lines 36-40, titles charts as `model | gpu | metric`, and hover data only includes `config_id` and `commit_sha`.
  - `fastvideo/performance_dashboard/service.py` groups API summaries/trends only by `(model_id, gpu_type)` at lines 80-86, recomputes latest summaries within that group at lines 97-169, and trend groups only expose `model_id`, `gpu_type`, and points at lines 172-194.
  - `apps/performance_dashboard/frontend/src/App.tsx` labels summary count as `model/GPU groups` at line 422, keys rows by `model_id-gpu_type` at line 449, and trend cards by `model_id-gpu_type-metric` at line 498; it has no visible cohort column or comparator status reason.
  - `fastvideo/tests/performance/test_dashboard_service.py` only asserts legacy grouping today; there are no tests on `upstream/main` that records differing only by `variant_id` or `software_profile_id` become separate dashboard cohorts.
  - `docs/contributing/performance_benchmarks.md` still says exact identity comparison and dashboard regrouping are follow-ups at lines 228-231, compatibility with legacy records is limited to missing component metrics and baseline eligibility at lines 305-311, and Plotly dashboard grouping is `(model_id, gpu_type)` at lines 349-350.
- Related-branch findings:
  - #1546 adds `COMPARISON_COHORT_KEYS` to Plotly and service grouping, preserves legacy records by filling missing cohort columns with empty strings, and displays those empty values as `legacy`. It adds `test_dashboard_plotly.py` and API/service tests for variant/software separation.
  - #1546 frontend has `cohortTitle` and `cohortDetail` helpers that display missing cohort values as `legacy`; it changes the Latest Status panel to `comparison cohorts`.
  - #1532 comparator has explicit statuses and treats v2 records without a comparable baseline as `CALIBRATION_NEEDED`; legacy records without v2 identity still use model/GPU lookup and can be `PASS` with `No legacy baseline...`.
  - Because #1546 and #1532 are not stacked on one another, #1535 should avoid independently reimplementing both. It should either wait for the stack to merge/rebase or be based on a reconciled stack branch.

## Merits And Scope

- The issue is valid on current `upstream/main`: dashboard views and recomputed baselines can still aggregate different benchmark identities if v2 fields appear in normalized records because the dashboard only keys on model/GPU.
- The user-facing risk is dashboard ambiguity, not core model behavior. A viewer could see a single trend or summary for mixed variants/software cohorts and misread calibration, mismatch, or regression states.
- The issue comment narrows the remaining scope: after v2 grouping, do the legacy/v1 story. That should mean preserving pre-v2 rows as one explicit legacy cohort per `(model_id, gpu_type)`, visually separated from v2 cohorts, without changing comparator policy.
- GPU memory impact: none expected; dashboard/data-transform only.
- Documentation impact: update contributor docs and dashboard README only if the final implementation changes visible dashboard behavior or endpoint payloads.

## Alternatives Considered

1. Minimal `upstream/main` fix:
   - Port #1546-style dashboard grouping helpers directly to this issue branch and add tests for variant/software profile split and legacy default handling.
   - Pros: addresses the original issue quickly on current `main`.
   - Cons: duplicates active #1546/#1551 work and risks conflicts; does not incorporate #1532 status fields unless this branch also grows comparator/status surface.

2. Legacy-only patch stacked after #1546:
   - Keep #1546's full cohort grouping, then improve the legacy cohort display and coverage: one explicit `Legacy v1` or equivalent cohort label for missing identity fields under each `(model_id, gpu_type)`, ensure v1 and v2 records never share a group, and make legacy trend cards readable.
   - Pros: matches SolitaryThinker's comment and avoids comparator changes.
   - Cons: depends on #1546 merging or this issue branch being rebased onto that PR branch.

3. Full reconciled follow-up after #1530/#1531/#1532:
   - Base #1535 on the reconciled performance identity/status stack. Add dashboard support for explicit `comparison_status` and `comparison_status_reason` from #1532, while preserving #1546's cohort grouping and making legacy/v1 grouping visibly distinct.
   - Pros: best satisfies every #1535 acceptance criterion, including clear calibration/mismatch/regression labels.
   - Cons: waits on or actively reconciles currently non-linear open PR branches.

4. Comparator changes in #1535:
   - Change baseline/comparator behavior here so legacy records skip rolling comparison and v2 records use exact identity.
   - Rejected for now: #1532 already owns comparator status behavior, and the issue comment explicitly says no comparator changes.

## Recommended Plan

- Prefer Alternative 3 if the user wants the issue fully closed, because the acceptance criteria include status clarity and #1532 owns the canonical status fields. The implementation should be small and dashboard-focused once the stack is reconciled:
  1. Rebase or recreate the issue branch on the latest reconciled branch containing #1546 dashboard grouping, #1551 v2 normalized fields, and #1560 comparator statuses. If those PRs merge first, rebase onto updated `upstream/main`.
  2. In `fastvideo/performance_dashboard/service.py`, preserve cohort grouping by full comparison identity but add stable, explicit display metadata for legacy cohorts and comparator status fields:
     - `comparison_status`
     - `comparison_status_reason`
     - a dashboard-facing `cohort_label` or equivalent derived label if the existing frontend helpers are too implicit.
  3. Keep legacy records grouped by `(model_id, gpu_type)` with all v2 identity fields empty; label them as legacy/v1 and never allow them to share a cohort key with v2 records.
  4. In `apps/performance_dashboard/frontend/src/api.ts` and `App.tsx`, render cohort identity plus comparator status/reason without treating legacy empty fields as unknown errors.
  5. In `fastvideo/tests/performance/dashboard.py`, keep Plotly grouping by full comparison cohort and label missing identity fields as legacy/v1. Include comparator status/reason in hover/skipped-metric context if those fields exist.
  6. Add focused tests:
     - service/API test: same `model_id`/`gpu_type` but different `variant_id` and `software_profile_id` produce separate groups.
     - service/API test: v1 record with no identity fields and v2 record with identity fields under same model/GPU produce separate groups.
     - service/API or frontend data-shape test: `comparison_status` and `comparison_status_reason` propagate to summary/trend payloads.
     - Plotly test: legacy group gets a readable legacy/v1 cohort label and v2 group title includes workload/variant/version plus hardware/software/recipe detail.
  7. Update docs to replace the current "dashboard regrouping is follow-up" wording with the actual grouping behavior and legacy/v1 treatment.
- If the user wants a lower-risk immediate patch before #1530/#1532 merge, use Alternative 2 and stack on `origin/issue/1530-fingerprinting-cohorts`, but clearly document that comparator status labels will be completed after #1532 lands.

## Validation Plan

- Do not run local project tests. Validation should run on Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`.
- Targeted test command for eventual Stage 2:
  - `pytest fastvideo/tests/performance/test_dashboard_service.py fastvideo/tests/performance/test_dashboard_api.py fastvideo/tests/performance/test_dashboard_plotly.py -q`
- If frontend payload/types are touched:
  - run frontend build in the appropriate environment (`apps/performance_dashboard/frontend`, `npm run build`) if dependencies are available in Modal or an approved environment.
- Before any draft PR in Stage 4:
  - `pre-commit run --all-files`
- No Wan SSIM or GPU model validation should be needed because this is dashboard/CI reporting code only.

## Open Questions

- Should Stage 2 wait for #1546/#1551/#1560 to merge/rebase, or should this issue branch stack on one of those open PR branches now?
- Preferred legacy label: `legacy`, `Legacy v1`, or another owner-approved wording?

## Next Steps

1. Present Stage 1 report to the user.
2. Wait for user guidance before implementation. No implementation has been performed.

## 2026-07-16 Refreshed Stage 1 Conclusion

This section supersedes the earlier alternatives, recommendation, open
questions, and next steps where dependency state has changed. The historical
notes above remain useful for reconstructing why the branch originally waited.

### Current Upstream Revision And Searches

- Inspected upstream/main at
  1c04ace57351f7340e8d1a2e8b8f62180856ed16 after fetching upstream.
- Read the current root guidance, fastvideo/AGENTS.md, and
  fastvideo/tests/AGENTS.md; no narrower guidance exists for the dashboard
  service or React app.
- Re-read apps/performance_dashboard/README.md and the schemas, legacy
  compatibility, CI integration, and troubleshooting sections of
  docs/contributing/performance_benchmarks.md.
- Inspected current implementations and focused tests in:
  - fastvideo/performance_dashboard/service.py
  - fastvideo/performance_dashboard/api.py
  - fastvideo/tests/performance/dashboard.py
  - fastvideo/tests/performance/test_dashboard_service.py
  - fastvideo/tests/performance/test_dashboard_api.py
  - fastvideo/tests/performance/test_dashboard_plotly.py
  - apps/performance_dashboard/frontend/src/api.ts
  - apps/performance_dashboard/frontend/src/App.tsx
- Searched all current and closed PRs/issues for 1535, Legacy v1, legacy/v1,
  and dashboard comparison-status work. No overlapping PR or duplicate issue
  was found.
- Read merged PR #1560's body, comments, and reviews to confirm its dashboard
  changes were cohort-parity work and did not intentionally add status display.

### Refreshed Code Findings

- Full v2 cohort grouping is already merged and correct in both dashboard
  paths. The service's comparison_cohort_key and Plotly dashboard's
  _dashboard_frame use all six identity fields for complete v2 records,
  independent of display-only model_id/gpu_type.
- Legacy records with no complete v2 identity are keyed separately under
  (model_id, gpu_type) and cannot merge into a complete v2 cohort. Partial v2
  identities also remain scoped by model/GPU and their present identity values,
  preventing unrelated invalid records from coalescing.
- Existing tests cover full-cohort separation, variant/benchmark-version
  separation, software/recipe separation, display-name changes, legacy
  model/GPU grouping, and v1/v2 separation including benchmark version zero.
- The merged comparator emits canonical comparison_status values PASS,
  REGRESSION, CALIBRATION_NEEDED, RECIPE_MISMATCH, and INFRA_ERROR, plus
  comparison_status_reason. Legacy records skip rolling comparison with
  baseline_status=skipped_missing_identity; partial v2 identities are invalid
  and become infrastructure errors.
- No dashboard service, API, Plotly, or React code reads or displays
  comparison_status, comparison_status_reason, or baseline_status. Summary and
  trend transforms currently drop all three fields. The React UI shows only
  stored boolean success and a locally recomputed pass/fail value, so
  calibration, recipe mismatch, and infrastructure failure are visually
  indistinguishable despite the issue's explicit status-label criterion.
- True legacy cohorts currently render as legacy / legacy / legacy with
  recipe legacy | legacy | legacy. They are separated correctly, but this is
  less readable than one explicit Legacy v1 label and does not clearly
  distinguish old v1 history from an invalid partial-v2 record.
- The issue is therefore still valid but narrower than originally reported:
  grouping behavior is already present; explicit legacy presentation and
  canonical status visibility remain.
- This is dashboard/data presentation only. No model behavior, GPU memory,
  performance benchmark semantics, or comparator policy should change.

### Refreshed Alternatives

1. Close as already implemented
   - Treat merged #1546/#1560 grouping and the repeated legacy placeholders as
     sufficient.
   - Lowest code risk, but leaves the explicit status-label criterion unmet and
     historical v1 presentation ambiguous. Not recommended.
2. Legacy-label-only patch
   - Render all-empty historical cohorts as Legacy v1 in Plotly and React, with
     focused v1/v2 separation and label tests.
   - Matches the issue comment's narrow wording, but still hides the canonical
     calibration/mismatch/regression statuses required by the issue body.
3. Scoped dashboard completion (recommended)
   - Keep the merged grouping and comparator unchanged; add explicit cohort
     presentation metadata plus pass-through of the canonical stored status,
     reason, and baseline status to dashboard summaries/trends and both visual
     surfaces.
   - Slightly broader than a label-only patch, but it directly closes every
     remaining acceptance criterion without duplicating or altering comparator
     logic.

### Refreshed Recommended Plan

1. Rebase the issue branch onto current upstream/main with signed rewritten
   commits, preserving the tracked handoff and the branch's dedicated worktree.
2. In fastvideo/performance_dashboard/service.py, expose dashboard metadata
   derived from each normalized record:
   - a stable cohort kind/display label that distinguishes complete v2,
     historical v1, and invalid/incomplete v2 presentation without changing
     the existing grouping key;
   - result_schema_version, baseline_status, comparison_status, and
     comparison_status_reason, with neutral empty defaults for old historical
     records that predate those fields.
   Propagate it to latest-summary rows, trend groups where appropriate, and
   every trend point.
3. In apps/performance_dashboard/frontend/src/api.ts and App.tsx, type and
   render the canonical status. Use one explicit Legacy v1 cohort label for
   true v1 history, keep partial-v2 identity visible as invalid/incomplete
   rather than mislabeling it as v1, and show status reasons in the latest row
   and trend point detail without removing the existing stored/recomputed
   context.
4. In fastvideo/tests/performance/dashboard.py, use the same explicit legacy
   title and include canonical status/reason in Plotly hover data when present,
   while defaulting missing fields so old records remain renderable.
5. Add focused service/API/Plotly coverage for:
   - a true v1 record and a complete v2 record sharing model/GPU remain separate;
   - true v1 output is labeled Legacy v1;
   - invalid partial-v2 data is not labeled as v1;
   - status, reason, and baseline status survive summary/trend transforms and
     API serialization;
   - old historical records missing status fields still render safely.
   Retain the already-merged variant and software-profile separation tests.
6. Update apps/performance_dashboard/README.md and the durable dashboard/legacy
   paragraphs in docs/contributing/performance_benchmarks.md to state the
   visible v1 treatment and canonical status behavior.

### Refreshed Validation Plan

- Do not run project tests locally.
- Run focused validation on Modal L40S through
  fastvideo/tests/modal/launch_l40s_job.py from branch interleavethinker:
  pytest fastvideo/tests/performance/test_dashboard_service.py fastvideo/tests/performance/test_dashboard_api.py fastvideo/tests/performance/test_dashboard_plotly.py -q.
- Run the React production build in an approved environment after frontend type
  changes: npm run build from apps/performance_dashboard/frontend.
- Inspect the dashboard at desktop and mobile widths to confirm the added status
  and legacy labels do not overflow the existing dense table/cards.
- Run pre-commit run --all-files before presenting PR readiness and again as
  the mandatory Stage 4 pre-PR gate. Fix every failure before any draft PR.
- Pass criteria: all focused tests pass; frontend build succeeds; v1/v2 are
  separate; true v1 is explicitly labeled; canonical status/reason are visible;
  historical records without new fields remain renderable.
- No SSIM or model GPU run is warranted; GPU memory impact is none.
- Any new PR must be draft-only and existing PR draft status must never be
  changed.

### Refreshed Open Questions And Next Decision

- No dependency or technical question currently blocks Stage 2.
- Recommended wording is Legacy v1; it is explicit, durable, and matches the
  issue title and documentation terminology.
- Await user approval of Alternative 3, selection of another alternative, or
  changes to the plan. No implementation has been performed.

## Commits And Pushes

- `56f9e3e092838bc18137209110232b0b995cabf6` - signed handoff-only commit `[misc]: add issue 1535 handoff`, pushed to `origin/issue/1535-ci-dashboard-grouping-legacy-v1`.
- `aecb30397` - signed handoff status update `[misc]: record issue 1535 handoff status`, pushed to `origin/issue/1535-ci-dashboard-grouping-legacy-v1`.
- Handoff remains active and should not be removed until Stage 4 immediately before creating any draft PR.
