<!--
PR TITLE: Must start with a type tag, e.g.:
  [feat] Add new model       [bugfix] Fix VAE tiling      [refactor] Restructure pipeline
  [perf] Optimize kernel     [ci] Update tests             [docs] Add guide
  [misc] Cleanup configs     [new-model] Port Flux2

MERGE WORKFLOW:
  1. Ensure pre-commit passes and you have at least 1 approval
  2. Comment /merge (or add the "ready" label) to enter the Merge Queue
  3. Full Test Suite runs automatically on a staging branch → auto-merge on success

ON-DEMAND TESTING (write access required):
  /test full       — Full Test Suite        /test ssim        — SSIM regression
  /test training   — Training pipeline      /test encoder     — Encoder tests
  /test transformer — Transformer tests     /test vae         — VAE tests
  /test kernel     — CUDA kernel tests      /test unit        — Unit tests
  See docs/contributing/pull_requests.md for all 17 test commands
-->

## Purpose

<!-- What does this PR do? Link the related issue if applicable. -->

Fixes #

## Changes

<!-- Describe your changes concisely. What approach did you take? -->

-

## Test Plan

<!-- How did you verify your changes? Paste exact commands and output. -->

```bash
# Commands you ran
```

## Test Results

<!-- Paste test output, before/after comparisons, or SSIM scores for model changes. -->

<details>
<summary>Test output</summary>

```
# Paste output here
```

</details>

## Checklist

- [ ] I ran `pre-commit run --all-files` and fixed all issues
- [ ] I added or updated tests for my changes
- [ ] I updated documentation if needed
- [ ] I considered GPU memory impact of my changes

**For model/pipeline changes, also check:**
- [ ] I verified SSIM regression tests pass
- [ ] I updated the support matrix if adding a new model
