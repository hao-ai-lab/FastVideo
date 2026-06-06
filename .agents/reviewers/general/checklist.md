# General reviewer checklist

## Public API

- [ ] User-facing renames / removals have a deprecation shim or migration note.
- [ ] New CLI flags documented in `docs/` or the PR body.
- [ ] OpenAI-compatible serving behavior matches OpenAI spec (or divergence
      is justified).
- [ ] `SamplingParams` / `ServeConfig` / `VideoGenerator` defaults unchanged
      unless called out.

## Pipeline stages (shared)

- [ ] Cross-model test evidence provided when `pipelines/stages/` is edited
      (at least 2 pipelines).
- [ ] New stages are reusable — accept generic inputs, not model-specific
      dicts.

## CI

- [ ] No secret hard-coded in workflow yaml.
- [ ] `pull_request_target` usage is gated (if newly used).
- [ ] Minimal `permissions:` block declared.
- [ ] `concurrency:` block declared for long workflows.
- [ ] Action versions pinned to SHA (match repo convention).

## Docs

- [ ] New pages added to `mkdocs.yml` nav.
- [ ] Example commands match current CLI / API.
- [ ] Internal links are relative.

## UI

- [ ] Auth / ACL preserved.
- [ ] No unescaped HTML from run metadata.

## Dependencies

- [ ] License compatible (MIT / BSD / Apache-2.0 OK).
- [ ] Version pins consistent with the rest of the file.
- [ ] Removed deps no longer imported anywhere.

## Cross-cutting (check on every PR)

- [ ] No secret strings in diff.
- [ ] No large binaries committed directly.
- [ ] License header on new source files (if neighbors have one).
- [ ] Debug / scratch scripts not committed to repo root.
- [ ] `.gitignore` updated for any new generated-file pattern.

## PR template compliance

- [ ] Title matches the mergify regex (from
      `.github/mergify.yml:merge_protections.success_conditions`):

      ```
      (?i)^\[(feat|feature|bugfix|fix|refactor|perf|ci|doc|docs|misc|chore|kernel|new.?model)\]
      ```

      Common misses: `[test]`, `[tests]`, `[wip]`, `[draft]` — none are valid.
- [ ] Description is not the template placeholder.
- [ ] Test Plan + Test Results sections populated (even for non-code PRs —
      "N/A because docs-only" is acceptable).

## Merge-protection anomalies (always check on a merged PR)

- [ ] If the PR is already merged, confirm the required checks
      (`pre-commit`, `fastcheck-passed`, `full-suite-passed`) were SUCCESS at
      merge, or that the PR body explicitly acknowledges an expected failure
      (see `../shared/repo-conventions.md` → "Expected failure" workflow).
- [ ] Title matched the mergify regex at merge time. If not, the `/merge`
      queue should have rejected it — note the anomaly.
