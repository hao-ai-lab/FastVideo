---
name: ci-runner
description: Work on the Slurm CI runner lanes - add or migrate a CI lane to the ci-runner Buildkite queue, keep Modal/Slurm test selections drift-free via shared lane scripts, and flip a lane's authoritative routing per migration wave.
---

# Slurm CI runner lanes

FastVideo's `ci-runner` Buildkite queue dispatches lanes to one exclusive
Slurm tray via a host-owned driver (`/opt/fastvideo-ci-runner/run-ci`;
`run-unit` for the legacy unit step). The host side is a private operator
bundle, deliberately not in this repository (site-specific paths/endpoints;
see `docs/contributing/ci_architecture.md` and, historically,
`HANDOFF-ci-runner.md`). This skill covers the REPO half only.

## Invariants (contract-tested — keep them true)

- Every Slurm lane's test selection lives in a shared script under
  `.buildkite/scripts/` (`unit_test.sh`, `lanes/<lane>.sh`) executed by BOTH
  the Modal launcher (`fastvideo/tests/modal/pr_test.py`) and the tray, so
  selections cannot drift. Pinned by
  `fastvideo/tests/modal/test_pr_test.py` (shared-script tests).
- Each Slurm step in `.buildkite/pipeline.yml` pins: a unique `key`, an `if`
  on `TEST_SCOPE`/`TEST_TYPE`, the driver command string, 90-minute timeout,
  step-level `TEST_TYPE` env, and `queue: "ci-runner"`. Pinned byte-for-byte
  by `fastvideo/tests/contract/test_ci_test_collection.py`.
- The host policy fail-closes anything not matching a known lane tuple, so a
  repo-side lane change is inert until the operator adds the matching lane
  row on the host. Coordinate both halves.

## Adding a lane (repo half)

1. Extract the Modal payload into `.buildkite/scripts/lanes/<lane>.sh`
   (payload only; backend-specific env such as HF_HOME/auth stays in the
   backend wrapper), `chmod +x`.
2. Repoint `run_<lane>_tests` in `pr_test.py` at `bash <that script>`.
3. Add the pipeline step (copy an existing `*-ci` block), the `/test
   <name>-ci` entry in `.github/workflows/ci-slash-commands.yml` (VALID +
   MAP), and extend the two contract tests above.
4. `pre-commit run --files <changed>` and run the contract tests.

## Migration waves / flipping

New lanes start as ADDITIVE `/test <name>-ci` routes with Modal
authoritative. Flipping a lane = repointing its original pipeline step to the
ci-runner queue/driver (+ widening its host-side scope tuple, e.g. adding the
Full Suite pair) in one commit; rollback is the revert. Hardware-sensitive
lanes (ssim, performance, golden-gate's bitwise fingerprints) must be
re-baselined on the tray's GPU architecture before they may gate merges.
