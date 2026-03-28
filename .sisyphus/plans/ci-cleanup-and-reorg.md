# CI Infrastructure Cleanup & Reorganization

## TL;DR

> **Quick Summary**: Remove obsolete RunPod CI infrastructure and dead workflows, fix broken Mergify automation, and rename all GitHub Actions workflows with a consistent category-prefix naming convention.
> 
> **Deliverables**:
> - Dead files removed (6 files + 1 directory)
> - Dead references cleaned (.pre-commit-config, test comments, docs)
> - Mergify config updated with Buildkite failure notification rules
> - All 11 remaining workflows renamed with `ci-` / `publish-` / `infra-` / `community-` / `_template-` prefixes
> - All cross-references updated atomically
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1 → Task 3 → Task 4 → F1-F4

---

## Context

### Original Request
User wants to clean up the CI infrastructure: remove obsolete RunPod GPU testing, fix Mergify auto-labeling and CI failure notifications, and reorganize workflow files for clarity.

### Interview Summary
**Key Decisions**:
- RunPod GPU testing is CONFIRMED OBSOLETE — remove entirely
- `go` label replaced by `ready` — delete `go`
- `test.yml` superseded by Buildkite — delete
- `vsa-publish.yml` references deleted paths — delete
- RunPod developer docs: KEEP but rewrite (remove CI content)
- `docker/rocm7_1.Dockerfile`: KEEP (ROCm plans exist)
- `.buildkite/scripts/pre_commit.sh`: KEEP
- Workflow rename scheme: approved (ci- / publish- / infra- / community- / _template-)

**Research Findings**:
- 15 recent PRs checked via `gh` CLI: ZERO have Mergify auto-labels
- Mergify "Configuration changed" check runs on config-modifying PRs, but NO Mergify checks appear on regular PRs → app likely inactive/expired
- Buildkite check names on PRs: `buildkite/ci`, `buildkite/ci/microscope-*`, `buildkite/ci/*-tests`
- `.pre-commit-config.yaml` line 22 references non-existent `sta-publish.yml`
- `docs.yml` lines 10, 17 self-reference `.github/workflows/docs.yml` — must update on rename
- `docs/README.md:39` references `docs.yml` — must update on rename

### Metis Review
**Identified Gaps (addressed)**:
- Mergify dashboard check must be a HARD PREREQUISITE before config changes (reordered in plan)
- All workflow renames must be a single atomic commit (enforced in Task 3)
- `docs.yml` self-referencing path triggers must be updated in rename commit (included in Task 3)
- `docs/README.md:39` reference must be updated (included in Task 3)
- 6 test file RunPod comments VALIDATED — they exist (Metis false negative corrected)

---

## Work Objectives

### Core Objective
Remove all dead CI infrastructure, restore Mergify automation, and establish a clear naming convention for workflow files.

### Concrete Deliverables
- 6 files deleted, `.github/scripts/` directory deleted
- `.pre-commit-config.yaml` cleaned of dead exclusions
- 6 test files cleaned of RunPod comments
- RunPod developer docs rewritten as pure dev guide
- Mergify config augmented with Buildkite CI failure notification rules
- 11 workflow files renamed with category prefixes
- All cross-references (4 `uses:` paths, pre-commit exclusions, docs.yml self-triggers, docs/README.md) updated

### Definition of Done
- [ ] `ls .github/scripts/` → directory does not exist
- [ ] `ls .github/workflows/runpod-test.yml .github/workflows/pr-test.yml .github/workflows/test.yml .github/workflows/vsa-publish.yml 2>&1` → all "No such file"
- [ ] `grep -r "runpod" .github/workflows/` → no output
- [ ] `grep "sta-publish" .pre-commit-config.yaml` → no output
- [ ] `ls .github/workflows/*.yml | grep -cE "(ci-|publish-|infra-|community-|_template-)"` → 11
- [ ] `grep "buildkite" .github/mergify.yml` → matches exist
- [ ] `pre-commit run --all-files` → exits 0

### Must Have
- Zero dangling cross-references after every commit
- Workflow internal `name:` fields UNCHANGED (Mergify and branch protection depend on them)
- Atomic rename commit — no intermediate broken states

### Must NOT Have (Guardrails)
- DO NOT change any workflow `name:` field — only filenames change
- DO NOT refactor or "improve" existing Mergify rules — only add missing notification rules
- DO NOT touch `.buildkite/pipeline.yml` or `trigger-full-suite.yml` behavior
- DO NOT touch `pre-commit.yml`'s `workflow_call:` definition (may be used externally)
- DO NOT add new workflow features during the rename
- DO NOT split Part 3 renames across multiple commits
- DO NOT add Mergify rules before confirming the app is active (dead code risk)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pre-commit hooks, pytest)
- **Automated tests**: None needed — this is infrastructure file management
- **Framework**: pre-commit (for lint/format verification)

### QA Policy
Every task includes agent-executed QA scenarios using Bash commands.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **File operations**: Use Bash — verify existence/non-existence, grep for references
- **Config validation**: Use Bash — `python -c "import yaml; ..."`, `pre-commit run`
- **Cross-reference integrity**: Use Bash — grep for stale references

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start immediately — cleanup):
├── Task 1: Delete obsolete files + clean references [quick]
│
Wave 2 (After Wave 1 — parallel tasks):
├── Task 2: Rewrite RunPod developer docs [writing]
├── Task 3: Rename all workflows + update cross-refs (ATOMIC commit) [deep]
│
Wave 3 (After user confirms Mergify dashboard — may run parallel with Wave 2):
├── Task 4: Update Mergify config with Buildkite failure rules [quick]
│
Manual Tasks (user action required — any time):
├── Task M1: Delete GitHub secrets/environment/label
├── Task M2: Check Mergify dashboard + re-authorize if needed
│
Wave FINAL (After ALL tasks):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Cross-reference integrity check (deep)
├── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | — | 2, 3 |
| 2 | 1 | FINAL |
| 3 | 1 | FINAL |
| 4 | M2 (manual gate) | FINAL |
| M1 | — | — |
| M2 | — | 4 |

### Agent Dispatch Summary

- **Wave 1**: 1 task → `quick`
- **Wave 2**: 2 tasks → `writing`, `deep`
- **Wave 3**: 1 task → `quick`
- **FINAL**: 4 tasks → `oracle`, `unspecified-high`, `deep`, `deep`

---

## TODOs

- [ ] 1. Delete obsolete files and clean dead references

  **What to do**:
  - Delete workflow files:
    - `.github/workflows/runpod-test.yml`
    - `.github/workflows/pr-test.yml`
    - `.github/workflows/test.yml`
    - `.github/workflows/vsa-publish.yml`
  - Delete scripts and directory:
    - `.github/scripts/runpod_api.py`
    - `.github/scripts/runpod_cleanup.py`
    - `.github/scripts/` directory (empty after above deletions)
  - Clean `.pre-commit-config.yaml` — remove these exclusion lines:
    - Line referencing `.github/workflows/sta-publish.yml` (file does not exist — phantom reference)
    - Line referencing `.github/workflows/vsa-publish.yml` (file being deleted)
  - Clean RunPod comments from 6 test files — remove the trailing comment `# store in the large /workspace disk on Runpod` from:
    - `fastvideo/tests/encoders/test_llama_encoder.py:25`
    - `fastvideo/tests/encoders/test_clip_encoder.py:25`
    - `fastvideo/tests/vaes/test_wan_vae.py:24`
    - `fastvideo/tests/vaes/test_hunyuan_vae.py:26`
    - `fastvideo/tests/transformers/test_hunyuanvideo.py:28`
    - `fastvideo/tests/transformers/test_wanvideo.py:27`
  - Important: Only remove the COMMENT portion (` # store in the large /workspace disk on Runpod`). The actual code (`local_dir=os.path.join(...)`) must stay intact.

  **Must NOT do**:
  - DO NOT delete any other workflow files
  - DO NOT modify any code logic — only delete files and remove comments/exclusions
  - DO NOT touch `.buildkite/` or Buildkite-related files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward file deletions and small edits — no complex logic
  - **Skills**: []
    - No special skills needed — basic file operations

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation task)
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None (start immediately)

  **References**:

  **Pattern References**:
  - `.pre-commit-config.yaml:19-26` — Exclusion block where `sta-publish.yml` (line ~22) and `vsa-publish.yml` (line ~23) are listed. Remove only these two lines; keep `fastvideo-publish.yml` and `build-image-template.yml` exclusions.

  **File References**:
  - `fastvideo/tests/encoders/test_llama_encoder.py:25` — Comment reads: `local_dir=os.path.join("data", BASE_MODEL_PATH) # store in the large /workspace disk on Runpod`. Remove only ` # store in the large /workspace disk on Runpod`.
  - Same pattern in all 6 test files listed above.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All obsolete files deleted
    Tool: Bash
    Preconditions: Repository checked out at current HEAD
    Steps:
      1. Run: test -f .github/workflows/runpod-test.yml && echo "EXISTS" || echo "GONE"
      2. Run: test -f .github/workflows/pr-test.yml && echo "EXISTS" || echo "GONE"
      3. Run: test -f .github/workflows/test.yml && echo "EXISTS" || echo "GONE"
      4. Run: test -f .github/workflows/vsa-publish.yml && echo "EXISTS" || echo "GONE"
      5. Run: test -f .github/scripts/runpod_api.py && echo "EXISTS" || echo "GONE"
      6. Run: test -f .github/scripts/runpod_cleanup.py && echo "EXISTS" || echo "GONE"
      7. Run: test -d .github/scripts && echo "EXISTS" || echo "GONE"
    Expected Result: All 7 checks print "GONE"
    Failure Indicators: Any check prints "EXISTS"
    Evidence: .sisyphus/evidence/task-1-files-deleted.txt

  Scenario: Dead pre-commit exclusions removed
    Tool: Bash
    Preconditions: .pre-commit-config.yaml exists
    Steps:
      1. Run: grep "sta-publish" .pre-commit-config.yaml | wc -l
      2. Run: grep "vsa-publish" .pre-commit-config.yaml | wc -l
      3. Run: grep "fastvideo-publish" .pre-commit-config.yaml | wc -l
    Expected Result: Steps 1 and 2 return 0; Step 3 returns 1 (still present)
    Failure Indicators: Steps 1 or 2 return non-zero
    Evidence: .sisyphus/evidence/task-1-precommit-clean.txt

  Scenario: RunPod comments removed from test files
    Tool: Bash
    Preconditions: Test files exist
    Steps:
      1. Run: grep -r "workspace disk on Runpod" fastvideo/tests/ | wc -l
      2. Run: grep "local_dir=os.path.join" fastvideo/tests/encoders/test_llama_encoder.py | wc -l
    Expected Result: Step 1 returns 0; Step 2 returns 1 (code intact, comment gone)
    Failure Indicators: Step 1 returns non-zero OR Step 2 returns 0
    Evidence: .sisyphus/evidence/task-1-comments-clean.txt

  Scenario: No dangling references to deleted files
    Tool: Bash
    Preconditions: Deletions completed
    Steps:
      1. Run: grep -r "runpod-test\.yml\|runpod_api\|runpod_cleanup" .github/ | wc -l
      2. Run: grep -r "test\.yml" .github/workflows/ | grep -v "_template\|_test\|precommit\|pre-commit\|full-suite\|trigger\|docs\|build\|publish\|kernel\|comfyui\|welcome\|labeler\|stale" | wc -l
    Expected Result: Both return 0
    Failure Indicators: Any returns non-zero
    Evidence: .sisyphus/evidence/task-1-no-dangling-refs.txt
  ```

  **Commit**: YES
  - Message: `[cleanup]: remove obsolete RunPod CI and dead workflows`
  - Files: `.github/workflows/runpod-test.yml`, `.github/workflows/pr-test.yml`, `.github/workflows/test.yml`, `.github/workflows/vsa-publish.yml`, `.github/scripts/runpod_api.py`, `.github/scripts/runpod_cleanup.py`, `.pre-commit-config.yaml`, 6 test files
  - Pre-commit: `pre-commit run --all-files`

- [ ] 2. Rewrite RunPod developer docs as pure development guide

  **What to do**:
  - Edit `docs/contributing/developer_env/runpod.md`:
    - Remove any CI/CD pipeline content (automated testing, GitHub Actions references)
    - Keep: How to set up a RunPod instance for FastVideo development
    - Keep: GPU selection guidance, SSH access, environment setup
    - Update: References to match current project setup (uv, Python 3.12, current Docker images)
  - `mkdocs.yml` navigation entry stays (docs are being kept)
  - RunPod screenshots in `docs/assets/images/runpod_*.png` stay (review during rewrite — update if outdated)

  **Must NOT do**:
  - DO NOT delete the RunPod docs page or mkdocs nav entry
  - DO NOT delete RunPod screenshots without replacing them
  - DO NOT add new sections about other cloud providers

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation rewrite requiring understanding of developer workflows
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: FINAL
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `docs/contributing/developer_env/runpod.md` — Current RunPod guide, read fully before rewriting
  - `docs/assets/images/runpod_*.png` — 6 screenshots, check if they match current RunPod UI

  **External References**:
  - Current Docker images: `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest`
  - Install command: `uv pip install -e .[dev]`

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: RunPod docs rewritten without CI content
    Tool: Bash
    Preconditions: docs/contributing/developer_env/runpod.md exists
    Steps:
      1. Run: grep -ci "github actions\|CI pipeline\|automated test\|pr-test\|runpod-test\.yml" docs/contributing/developer_env/runpod.md
      2. Run: grep -ci "development\|setup\|GPU\|ssh" docs/contributing/developer_env/runpod.md
      3. Run: test -f docs/contributing/developer_env/runpod.md && echo "EXISTS"
    Expected Result: Step 1 returns 0; Step 2 returns >0; Step 3 prints "EXISTS"
    Failure Indicators: Step 1 returns non-zero (CI content remains)
    Evidence: .sisyphus/evidence/task-2-docs-rewritten.txt

  Scenario: mkdocs navigation intact
    Tool: Bash
    Preconditions: mkdocs.yml exists
    Steps:
      1. Run: grep -c "runpod" mkdocs.yml
    Expected Result: Returns >=1 (navigation entry preserved)
    Failure Indicators: Returns 0 (entry accidentally removed)
    Evidence: .sisyphus/evidence/task-2-mkdocs-nav.txt
  ```

  **Commit**: YES
  - Message: `[docs]: rewrite RunPod guide as developer environment reference`
  - Files: `docs/contributing/developer_env/runpod.md`
  - Pre-commit: N/A

- [ ] 3. Rename all workflow files with category prefixes (ATOMIC commit)

  **What to do**:
  ALL of the following must be in a SINGLE commit. No intermediate states.

  **Step A — Rename files** (use `git mv`):
  | Current | New |
  |---------|-----|
  | `pre-commit.yml` | `ci-precommit.yml` |
  | `trigger-full-suite.yml` | `ci-trigger-full-suite.yml` |
  | `docs.yml` | `infra-docs.yml` |
  | `build-image.yml` | `infra-build-image.yml` |
  | `build-image-template.yml` | `_template-build-image.yml` |
  | `fastvideo-publish.yml` | `publish-fastvideo.yml` |
  | `fastvideo-kernel-publish.yml` | `publish-kernel.yml` |
  | `publish-comfyui.yml` | `publish-comfyui.yml` (no change) |
  | `welcome.yml` | `community-welcome.yml` |
  | `issue-labeler.yml` | `community-issue-labeler.yml` |
  | `stale.yml` | `community-stale.yml` |

  Note: `publish-comfyui.yml` already has the correct prefix — no rename needed. So 10 actual renames.

  **Step B — Update cross-references**:
  1. **`infra-build-image.yml`** (was `build-image.yml`): Update 4 `uses:` references from `./.github/workflows/build-image-template.yml` to `./.github/workflows/_template-build-image.yml`
  2. **`infra-docs.yml`** (was `docs.yml`): Update self-referencing path triggers on lines 10 and 17 from `.github/workflows/docs.yml` to `.github/workflows/infra-docs.yml`
  3. **`.pre-commit-config.yaml`**: Update exclusion entries:
     - `.github/workflows/fastvideo-publish.yml` → `.github/workflows/publish-fastvideo.yml`
     - `.github/workflows/build-image-template.yml` → `.github/workflows/_template-build-image.yml`
  4. **`docs/README.md:39`**: Update reference from `.github/workflows/docs.yml` to `.github/workflows/infra-docs.yml`

  **Step C — Verify** (DO NOT change):
  - Every workflow file's internal `name:` field must remain EXACTLY as-is
  - `ci-precommit.yml`'s `workflow_call:` trigger must remain
  - No workflow behavior changes — pure file rename

  **Must NOT do**:
  - DO NOT change any `name:` field inside workflow files
  - DO NOT split renames across multiple commits
  - DO NOT rename `publish-comfyui.yml` (already correct prefix)
  - DO NOT remove `workflow_call:` from `ci-precommit.yml`
  - DO NOT add new features or fix bugs in workflows during rename

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Atomic multi-file operation requiring careful cross-reference tracking. One mistake breaks CI.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2)
  - **Blocks**: FINAL
  - **Blocked By**: Task 1

  **References**:

  **File References (read BEFORE renaming)**:
  - `.github/workflows/build-image.yml` — Contains 4 `uses:` lines referencing `build-image-template.yml`. Search for `uses: ./.github/workflows/build-image-template.yml` and replace with `uses: ./.github/workflows/_template-build-image.yml`
  - `.github/workflows/docs.yml:10,17` — Self-referencing path triggers: `.github/workflows/docs.yml`. Replace with `.github/workflows/infra-docs.yml`
  - `.pre-commit-config.yaml:21,24` — Exclusion lines referencing `fastvideo-publish.yml` and `build-image-template.yml`. Update to new names.
  - `docs/README.md:39` — Text: "via the `.github/workflows/docs.yml` workflow". Update to `infra-docs.yml`.

  **WHY Each Reference Matters**:
  - `build-image.yml` → `_template-build-image.yml`: If `uses:` path is wrong, Docker image builds break entirely
  - `docs.yml` self-trigger: If path is wrong, docs workflow won't re-trigger when its own file is modified
  - `.pre-commit-config.yaml`: If exclusion paths are wrong, pre-commit will lint workflow files that should be excluded (may cause false failures)
  - `docs/README.md`: Stale documentation reference (low impact but should be correct)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All workflows renamed with correct prefixes
    Tool: Bash
    Preconditions: Task 1 completed (dead files already removed)
    Steps:
      1. Run: ls .github/workflows/*.yml | sort
      2. Run: ls .github/workflows/*.yml | wc -l
      3. Run: ls .github/workflows/*.yml | grep -cE "^.github/workflows/(ci-|publish-|infra-|community-|_template-)"
    Expected Result: Step 2 and Step 3 both return 11. Step 1 shows all files with correct prefixes.
    Failure Indicators: Step 2 ≠ Step 3 (some files lack prefix), or total ≠ 11
    Evidence: .sisyphus/evidence/task-3-rename-listing.txt

  Scenario: No old filenames remain
    Tool: Bash
    Preconditions: Renames completed
    Steps:
      1. Run: ls .github/workflows/ | grep -E "^(pre-commit|trigger-full-suite|docs|build-image|fastvideo-publish|fastvideo-kernel-publish|welcome|issue-labeler|stale)\.yml$" | wc -l
    Expected Result: Returns 0 (no old names remain)
    Failure Indicators: Returns non-zero
    Evidence: .sisyphus/evidence/task-3-no-old-names.txt

  Scenario: Cross-references point to existing files
    Tool: Bash
    Preconditions: Renames and cross-ref updates completed
    Steps:
      1. Run: grep -r "uses:.*\.github/workflows/" .github/workflows/ | grep -oP '\.github/workflows/[^ @"]+' | sort -u | while IFS= read -r ref; do test -f "$ref" && echo "OK: $ref" || echo "BROKEN: $ref"; done
      2. Run: grep "\.github/workflows/" .pre-commit-config.yaml | grep -oP '\.github/workflows/[^ "]+' | while IFS= read -r ref; do test -f "$ref" && echo "OK: $ref" || echo "BROKEN: $ref"; done
    Expected Result: All lines show "OK:", zero "BROKEN:"
    Failure Indicators: Any line shows "BROKEN:"
    Evidence: .sisyphus/evidence/task-3-cross-refs-valid.txt

  Scenario: Workflow name fields unchanged
    Tool: Bash
    Preconditions: Renames completed
    Steps:
      1. Run: grep "^name:" .github/workflows/ci-precommit.yml
      2. Run: grep "^name:" .github/workflows/infra-docs.yml
      3. Run: grep "^name:" .github/workflows/infra-build-image.yml
    Expected Result: Step 1 contains "pre-commit", Step 2 contains "Deploy Documentation", Step 3 matches original name
    Failure Indicators: Any name field was changed from its original value
    Evidence: .sisyphus/evidence/task-3-names-unchanged.txt

  Scenario: docs.yml self-trigger updated
    Tool: Bash
    Preconditions: infra-docs.yml exists
    Steps:
      1. Run: grep "infra-docs.yml" .github/workflows/infra-docs.yml | wc -l
      2. Run: grep "docs.yml" .github/workflows/infra-docs.yml | wc -l
    Expected Result: Step 1 returns 2 (lines 10, 17); Step 2 returns 0 (old name gone)
    Failure Indicators: Step 1 < 2 or Step 2 > 0
    Evidence: .sisyphus/evidence/task-3-docs-self-trigger.txt
  ```

  **Commit**: YES
  - Message: `[refactor]: rename workflow files with category prefixes`
  - Files: 10 renames + `.pre-commit-config.yaml` + `infra-build-image.yml` + `infra-docs.yml` + `docs/README.md`
  - Pre-commit: `pre-commit run --all-files`

- [ ] 4. Add Buildkite CI failure notification rules to Mergify

  **What to do**:
  - **PREREQUISITE**: Task M2 must be completed first (user confirms Mergify dashboard status)
  - Edit `.github/mergify.yml` to add a new rule after the existing "comment on pre-commit failure" section:

  Add a rule that comments on PRs when Buildkite CI tests fail:
  ```yaml
  - name: comment on Buildkite CI failure
    conditions:
      - or:
        - check-failure~=buildkite/ci
      - -closed
    actions:
      comment:
        message: |
          ## Buildkite CI tests failed

          Hi @{{author}}, some CI tests have failed. Please check the [Buildkite build](https://buildkite.com/fastvideo/ci) for details.

          Common causes:
          - **Test failures**: Check test output for assertion errors
          - **Import errors**: Make sure new dependencies are in `pyproject.toml`
          - **GPU memory**: Some tests require specific GPU types (L40S, H100)

          If the failure looks unrelated to your changes, add a comment explaining why.
  ```

  - Also verify the existing `check-failure~=pre-commit` rule syntax is correct
  - Ensure YAML is valid after changes

  **Must NOT do**:
  - DO NOT modify existing Mergify rules (auto-labeling, auto-merge, conflict detection)
  - DO NOT "improve" or refactor the existing rule structure
  - DO NOT add this rule if Mergify dashboard confirms the app is inactive/expired (user will gate this)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file edit, adding one YAML block
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (independent of Tasks 2, 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: FINAL
  - **Blocked By**: Task M2 (manual gate — user confirms Mergify is active)

  **References**:

  **Pattern References**:
  - `.github/mergify.yml:170-195` — Existing "comment on pre-commit failure" rule. Follow this exact pattern structure for the new rule.

  **Data References (from live PR analysis)**:
  - Buildkite check names seen on recent PRs:
    - `buildkite/ci` (umbrella status)
    - `buildkite/ci/microscope-unit-tests`, `buildkite/ci/microscope-transformer-tests`, etc.
    - `buildkite/ci/ssim-tests`, `buildkite/ci/training-tests`, `buildkite/ci/performance-tests`, etc.
  - The regex `buildkite/ci` will match all of these.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Mergify config is valid YAML with Buildkite rule
    Tool: Bash
    Preconditions: .github/mergify.yml exists
    Steps:
      1. Run: python3 -c "import yaml; yaml.safe_load(open('.github/mergify.yml'))"
      2. Run: grep -c "buildkite" .github/mergify.yml
      3. Run: grep -c "comment on Buildkite" .github/mergify.yml
    Expected Result: Step 1 exits 0 (valid YAML); Step 2 returns >=1; Step 3 returns 1
    Failure Indicators: Step 1 fails (invalid YAML) or Step 2 returns 0
    Evidence: .sisyphus/evidence/task-4-mergify-valid.txt

  Scenario: Existing Mergify rules unchanged
    Tool: Bash
    Preconditions: .github/mergify.yml modified
    Steps:
      1. Run: grep -c "label attention changes" .github/mergify.yml
      2. Run: grep -c "auto-merge when ready" .github/mergify.yml
      3. Run: grep -c "comment on pre-commit failure" .github/mergify.yml
    Expected Result: All return 1 (existing rules intact)
    Failure Indicators: Any returns 0
    Evidence: .sisyphus/evidence/task-4-existing-rules-intact.txt
  ```

  **Commit**: YES
  - Message: `[fix]: add Buildkite CI failure notification to Mergify config`
  - Files: `.github/mergify.yml`
  - Pre-commit: `python3 -c "import yaml; yaml.safe_load(open('.github/mergify.yml'))"`

- [ ] M1. **Manual: Delete GitHub secrets, environment, and label** (USER ACTION)

  > This task CANNOT be automated — requires GitHub repository admin access.

  **What to do**:
  1. Go to GitHub repo Settings → Secrets and variables → Actions → Repository secrets
     - Delete: `RUNPOD_API_KEY`
     - Delete: `RUNPOD_PRIVATE_KEY`
     - **DO NOT delete**: `WANDB_API_KEY`, `BUILDKITE_API_TOKEN`, `REGISTRY_ACCESS_TOKEN`
  2. Go to Settings → Environments
     - Delete: `runpod-runners`
  3. Go to repo Labels page (Issues → Labels, or use `gh label delete go --yes`)
     - Delete: `go` label (replaced by `ready`)

  **Verification**: Run `gh api repos/hao-ai-lab/FastVideo/labels --jq '.[].name' | grep "^go$"` → should return nothing.

  **Parallelization**: Can be done any time, independent of other tasks.

- [ ] M2. **Manual: Check Mergify dashboard and re-authorize** (USER ACTION — GATES Task 4)

  > This task CANNOT be automated — requires Mergify dashboard access.

  **What to do**:
  1. Go to https://dashboard.mergify.com/ → Select the FastVideo repository
  2. Check the event log — are recent PRs showing up?
  3. Check subscription status — is the plan active? Any usage limits hit?
  4. Check GitHub App permissions — does Mergify have `issues: write`, `pull-requests: write`, `contents: read` permissions?
  5. If issues found:
     - Re-authorize the GitHub App: GitHub Settings → Integrations → Mergify → Configure
     - Upgrade plan if free tier limits were reached
     - Check webhook delivery: GitHub Settings → Webhooks → verify Mergify webhook has recent successful deliveries
  6. After fixing, test: create a trivial PR modifying `fastvideo/attention/backends/base.py` → verify `attention` label is applied within 120 seconds

  **Report back**: Tell the agent executing Task 4 whether to proceed (Mergify active) or skip (Mergify broken, needs more investigation).

  **Parallelization**: Should be started early. GATES Task 4 — do not add Mergify rules until this is confirmed.

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan. For each "Must Have": verify implementation exists. For each "Must NOT Have": search for forbidden patterns. Check `.sisyphus/evidence/` files exist. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pre-commit run --all-files`. Check YAML validity of all workflow files and mergify.yml. Verify no orphaned files in `.github/workflows/`. Check for any `TODO` or `FIXME` left behind.
  Output: `Pre-commit [PASS/FAIL] | YAML [N valid/N total] | VERDICT`

- [ ] F3. **Cross-Reference Integrity Check** — `deep`
  For every `uses: ./.github/workflows/` reference: verify target file exists. For every `.pre-commit-config.yaml` exclusion: verify file exists. For every path trigger in workflow files: verify path format is correct. `grep -r "runpod\|sta-publish\|vsa-publish\|test\.yml" .github/` should return nothing.
  Output: `References [N/N valid] | Stale refs [CLEAN/N issues] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify ONLY planned changes were made: no workflow `name:` fields changed, no Buildkite files touched, no new features added, no pre-commit `workflow_call:` removed. Check git diff covers exactly the planned scope.
  Output: `Scope [CLEAN/N deviations] | VERDICT`

---

## Commit Strategy

| Commit | Message | Files | Pre-commit |
|--------|---------|-------|------------|
| 1 | `[cleanup]: remove obsolete RunPod CI and dead workflows` | Deletions + .pre-commit-config.yaml + 6 test files | `pre-commit run --all-files` |
| 2 | `[docs]: rewrite RunPod guide as developer environment reference` | docs/contributing/developer_env/runpod.md | N/A |
| 3 | `[refactor]: rename workflow files with category prefixes` | 11 renames + cross-ref updates | `pre-commit run --all-files` |
| 4 | `[fix]: add Buildkite CI failure notification to Mergify` | .github/mergify.yml | YAML validation |

---

## Success Criteria

### Verification Commands
```bash
# Part 1: Cleanup
! test -d .github/scripts                                    # directory gone
! test -f .github/workflows/runpod-test.yml                  # deleted
! test -f .github/workflows/pr-test.yml                      # deleted
! test -f .github/workflows/test.yml                         # deleted
! test -f .github/workflows/vsa-publish.yml                  # deleted
grep -r "runpod" .github/workflows/ | wc -l                  # → 0
grep "sta-publish" .pre-commit-config.yaml | wc -l           # → 0
grep -r "workspace disk on Runpod" fastvideo/tests/ | wc -l  # → 0

# Part 2: Mergify
python3 -c "import yaml; yaml.safe_load(open('.github/mergify.yml'))"  # valid YAML
grep -c "buildkite" .github/mergify.yml                      # → >0

# Part 3: Rename
ls .github/workflows/*.yml | wc -l                           # → 11
ls .github/workflows/*.yml | grep -cE "^.github/workflows/(ci-|publish-|infra-|community-|_template-)"  # → 11
# Cross-reference integrity
grep -r "uses:.*\.github/workflows/" .github/workflows/ | grep -v "^#" | while IFS= read -r line; do
  ref=$(echo "$line" | grep -oP '\.github/workflows/[^ @"]+')
  [ -f "$ref" ] || echo "BROKEN: $ref"
done  # → no output
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] `pre-commit run --all-files` passes
- [ ] Zero dangling cross-references
- [ ] All workflow `name:` fields unchanged from before rename
