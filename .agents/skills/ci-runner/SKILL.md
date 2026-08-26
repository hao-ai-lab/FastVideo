---
name: ci-runner
description: Work on FastVideo's Slurm-only, change-aware GPU CI lanes, static Buildkite graph, trusted ci-runner policy, lane scripts, and GB200 validation.
---

# Slinky Slurm CI lanes

FastVideo's `ci-runner` Buildkite queue is the control plane for all active
GPU CI. A private host-owned dispatcher leases GPUs from the Slinky Slurm tray
and runs the immutable PR SHA inside an isolated Enroot container. Buildkite
pipeline upload and Slurm submission occur on the login plane; every test
payload executes on Slurm compute.

The files under `fastvideo/tests/modal/` and `.buildkite/scripts/pr_test.sh`
are dormant rollback code. Never add an active Buildkite or slash-command
route to them. `pr_test.sh` must continue to reject Buildkite invocations.

The private operator bundle is deliberately outside this repository because
it contains site paths and credentials. See
`docs/contributing/ci_architecture.md`; this skill covers the repository half
and the coordination contract with that bundle.

## Invariants

- `.buildkite/pipeline.yml` contains exactly one static step for every active
  GPU lane. Each step pins a unique key and label, a 90-minute timeout, the
  trusted `/opt/fastvideo-ci-runner/run-ci` command (`run-unit` is the one
  compatibility wrapper), step-level internal `TEST_TYPE`, and
  `queue: "ci-runner"`.
- Active CI contains no `pr_test.sh` command, Modal invocation, default queue,
  Buildkite plugin, `soft_fail`, or job-controlled artifact glob.
- The six Fastcheck lanes use `:microscope:` labels. Full-Suite-only lanes use
  `:test_tube:` or `:bar_chart:` so direct reruns update the right aggregate.
- SSIM and vanilla training request all four GPUs. Keep both in the
  `fastvideo/slinky/whole-tray` Buildkite concurrency group with a limit of one
  so the second job does not consume an agent or command timeout while waiting
  for the same tray.
- `/test full` schedules all twenty lanes. `/merge`, `ready`, and new pushes to
  ready PRs use the trusted base-branch planner in
  `.github/scripts/plan_merge_ci.py`: automatic Fastcheck remains the universal
  six-lane baseline, and the merge build adds only path-relevant integration
  lanes. Unknown source/build paths fail closed to all fourteen additive lanes.
  The trusted uploader still normalizes and validates the complete static graph
  before Buildkite evaluates its plan conditions.
- Focused merge builds may pass allowlisted golden-gate and SSIM test basenames.
  The private host validates the lane plan and basenames before staging them,
  and the in-container scripts validate them again. Direct `/test ssim`,
  explicit `/test full`, and the weekly main-branch schedule run the complete
  SSIM matrix.
- The trusted uploader serves exactly three entry pipelines:
  `pr-fastcheck` for automatic PR builds, `ci` for slash-command/ready-label
  API builds, and `fastvideo-performance-lane` for the weekly schedule. Keep
  incoming GitHub webhook processing disabled on `ci` so it cannot duplicate
  `pr-fastcheck` on every PR update.
- Test payloads live in `.buildkite/scripts/unit_test.sh` or executable
  `.buildkite/scripts/lanes/<lane>.sh`. Backend policy (GPU count, extras,
  secrets, kernel build, artifacts) stays in the agent-owned lane table.
- Tests must preserve an inherited `MASTER_PORT`. Packed containers share the
  tray network namespace, so the private runner assigns a distinct port range
  per GPU lease and the SSIM scheduler assigns task offsets within its range.
- The ARM64 runner image includes the pinned FA4 CuTe overlay validated on
  GB200. Keep SSIM at `FASTVIDEO_FA4=1` because its references were seeded with
  FA4; keep lanes with FA2 baselines at `FASTVIDEO_FA4=0`. A runner image change
  must revalidate both the FA4 import and an actual GB200 forward kernel.
- `fastvideo/tests/ssim/ci_runner.py` is the active four-GPU SSIM scheduler.
  New SSIM files are discovered through `REQUIRED_GPUS` and
  `*_MODEL_TO_PARAMS`; do not wire them through the dormant Modal scheduler.
- The host policy fail-closes unknown tuples. A repository-side lane change is
  inert until the operator updates the private lane table and uploader policy
  in the same rollout.

## Adding or changing a lane

1. Read the closest `AGENTS.md` and the domain-specific testing guide.
2. Add or update the executable lane payload under `.buildkite/scripts/`.
   Keep it deterministic and free of host-specific paths or credential fetches.
3. Add the static pipeline step and canonical `/test <name>` mapping. Keep the
   `<name>-ci` alias only when compatibility requires it.
4. Add its source/test path ownership to `.github/scripts/plan_merge_ci.py`.
   Prefer the narrowest correctness-preserving lane set; leave unknown paths
   fail-closed. Extend `fastvideo/tests/contract/test_ci_test_collection.py`,
   `test_merge_ci_plan.py`, and focused CPU-only scheduler/policy tests.
5. Coordinate the private lane row: GPU count (1-4), wall time, script, scope
   pairs, step key, command, HF cache/token, tracking mode, extras, attention
   backend policy, kernel policy, and artifact relay. Active training lanes
   keep W&B offline and do not stage a W&B credential.
6. Update the trusted pipeline-uploader schema. A mismatch must reject the
   pipeline rather than silently skip a lane.
7. Run `pre-commit run --files <changed paths>`, the planner's representative
   diff matrix, contract tests, private driver tests, and a real GB200 canary.
   Multi-GPU, hardware-reference, training, performance, and SSIM changes need
   their own target-hardware evidence.

## Rollback

Rollback the Slurm routing/configuration change or pause the `ci-runner` queue.
Do not silently reactivate Modal. A manual Modal experiment requires the
explicit local opt-in documented in `ci_architecture.md`; returning it to
production CI needs a separate reviewed decision.
