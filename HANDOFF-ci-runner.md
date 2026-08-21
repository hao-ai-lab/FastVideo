# CI runner handoff

State: 2026-08-20

Branch: `infra/ci-runner`

Base inspected: `origin/main` at `00338aa9ca9161fc2e080d482bbac5836ebbfe0e`

No pull request is open. Do not open one without fresh author approval.

## Delivered on this branch

- `/test unit-ci` creates a direct Buildkite job with
  `TEST_TYPE=unit_test_ci`.
- `/merge`, `/test full`, and other Full Suite triggers include the same unit
  step without replacing any existing Modal lane.
- Buildkite sends that step to queue `ci-runner` and invokes only the
  host-owned `/opt/fastvideo-ci-runner/run-unit` dispatcher.
- The ARM64 CUDA 13 / SM100 image build and source-aware kernel cache are in
  the same branch.
- Modal and the tray runner share `.buildkite/scripts/unit_test.sh`, so their
  test selections cannot drift.

The `/merge` workflow must keep its explicit Buildkite API request. Its
`ready` label is written with `GITHUB_TOKEN`; GitHub intentionally does not
start another workflow from that generated `labeled` event. See
[GitHub's token trigger contract](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs).

## Execution path

```text
/merge or /test unit-ci
  -> immutable PR head SHA in Buildkite
  -> unit-ci step on queue ci-runner
  -> trusted host dispatcher
  -> SSH to the Slurm login plane
  -> one exclusive four-GPU sbatch job
  -> isolated enroot workspace
  -> .buildkite/scripts/unit_test.sh
  -> numeric ci.EXIT returned to Buildkite and GitHub
```

The private host policy accepts only these scope pairs:

| Trigger | `TEST_SCOPE` | `FULL_SUITE` | `TEST_TYPE` |
|---|---|---|---|
| `/test unit-ci` | `direct` | `false` | `unit_test_ci` |
| Full Suite | `full` | `true` | `unit_test_ci` |

All other commands, queues, repositories, pipelines, scopes, plugins,
artifact paths, retry modes, and shell-injection variables fail closed.

## Verified evidence

- Focused repository suite: `54 passed`.
- Private dispatcher tests: `13 passed`.
- Earlier tray proof with the same image/runtime: Slurm job `2441`, kernel
  cache hit, `901 collected / 901 passed`, `ci.EXIT=0`.
- Tested image digest:
  `sha256:12fe5a28817e89ddb1cdde7201226db9aa9ca2e541122d94e409f4ee2c8432e8`.

The earlier tray proof predates the Full Suite condition. Treat the new
`/merge` path as unverified until the Buildkite canaries below pass.

## Deployment order

1. Create or verify the dedicated `ci-runner` queue and a scoped, expiring,
   IP-restricted agent token.
2. On the persistent dispatcher host, install checksum-verified Buildkite
   Agent `3.137.0` under a dedicated non-root account.
3. Install the updated root-owned policy hook before enabling this branch. An
   old hook rejects Full Suite jobs by design.
4. Install the dispatcher scripts under `/opt/fastvideo-ci-runner/`; configure
   the dedicated SSH key, pinned known-hosts file, local registry/ledger, and
   digest-specific image path outside the repository checkout.
5. Confirm the tested image is readable from the Slurm login plane before
   allocating a tray. Never import an image while holding GPU capacity.
6. Start one Buildkite agent on queue `ci-runner` and record its agent ID.

The agent intentionally uses `spawn=1`, so tray jobs serialize globally.
Buildkite's 90-minute job timeout begins only after an agent accepts the job;
an offline queue can otherwise leave Full Suite pending until Buildkite's
separate scheduled-job expiry. Do not enable the Full Suite route before the
agent is online.

The private operator bundle contains the exact file-placement table, checksum,
hooks, driver tests, and start command. It is intentionally not committed to
this public repository because it contains site-specific paths and endpoints.

## Required canaries

Record the Buildkite build ID, Slurm job ID, pytest collection count, and final
exit status for each applicable run.

1. `/test unit-ci` reaches the dispatcher and returns the real test status.
2. A Full Suite build schedules the same `unit-ci` step and its result gates
   `full-suite-passed`.
3. Modified command, queue, pipeline, scope pair, plugin, artifact path,
   `BASH_ENV`, or Git configuration is rejected before checkout.
4. Canceling the Buildkite job sends `TERM`; the dispatcher cancels only its
   durable numeric Slurm job ID and closes its registry entry.

## Rollback and remaining gates

- Immediate disable: pause queue `ci-runner` or stop its dedicated agent, then
  cancel any scheduled runner jobs or revert the Full Suite condition so they
  do not remain pending. Existing Modal/default lanes remain unchanged.
- The protected dispatcher host still must be identified and provisioned.
- The API bearer token used for host discovery was exposed in a chat transcript
  and must be rotated; no token value belongs in this repository or handoff.
- PR #1720 is closed because its head branch was renamed. A replacement PR
  requires fresh author approval.
