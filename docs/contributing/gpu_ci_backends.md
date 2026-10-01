# Selectable GPU CI Backends

GPU CI can use the existing Slurm dispatcher, Modal, or GB200 Kubernetes Jobs
in the `vllm` namespace. GitHub and Buildkite remain the trigger and reporting
systems. `vllm` names the cluster namespace here; tests run FastVideo's existing
lane scripts rather than a vLLM inference server.

The default remains Slurm. The existing `.buildkite/pipeline.yml`, private
Slurm uploader, and dormant Modal launchers are preserved. Deploying the new
trusted uploader is an operator step; merging these files does not change the
live Buildkite pipelines or create cluster resources. See
[CI/CD Architecture](ci_architecture.md) for the existing installation.

## Execution and Limits

```text
GitHub PR / slash command / schedule
  -> trusted Buildkite bootstrap
     -> slurm: existing validated graph and private dispatcher
     -> modal or vllm: one trusted suite coordinator
        -> shared PR admission and GPU reservations
        -> isolated GPU worker per existing lane
        -> lane results, logs, and backend-specific suite status
```

All new Modal and `vllm` coordinators share one admission database. It enforces:

- At most two distinct PRs with admitted GPU work.
- At most four reserved GPUs per PR across its concurrent builds and lanes.
- At most eight reserved GPUs across both backends combined.
- Additional PRs wait without creating GPU workers.

A PR is identified by repository and PR number, so Fastcheck, merge builds,
manual reruns, and separate attempts share its allowance. Its slot remains
occupied between lanes until all admitted builds for that PR finish. A
non-PR build, including a schedule on `main`, consumes its own slot and the
same GPU budget. The preserved Slurm dispatcher has its existing independent
limits; this new admission database does not control legacy Slurm jobs.

Lanes retain their existing one-, two-, or four-GPU requirements. Four-GPU
SSIM and training lanes wait for that PR's other lanes to release capacity.
Selected integration lanes wait for the golden gate and are skipped if it
fails. CPU-only GitHub checks are unaffected. Pending admission is bounded by
`queue_timeout_seconds`, initially six hours. The coordinator's Buildkite
command timeout is eight hours, including admission and lane waits after the
command starts; time waiting for a Buildkite agent is outside that timeout.

Reservations persist before a worker is created. Cancellation releases them
only after worker termination is confirmed. A lost coordinator, uncertain
create response, or unavailable backend retains its reservations until
recovery confirms cleanup. There is no heartbeat expiry that silently frees
GPUs while an old worker could still be running.

## Operator Installation

Use an operator-reviewed immutable checkout under
`/opt/fastvideo-gpu-ci/source`. Install the supplied `scripts/gpu_ci/run` and
`scripts/gpu_ci/upload` wrappers as `/opt/fastvideo-gpu-ci/run` and
`/opt/fastvideo-gpu-ci/upload`. They invoke the reviewed Python entrypoint
with isolated import mode. Install the supplied files instead of generating
shell wrappers from build metadata:

```bash
install -m 755 /opt/fastvideo-gpu-ci/source/scripts/gpu_ci/run /opt/fastvideo-gpu-ci/run
install -m 755 /opt/fastvideo-gpu-ci/source/scripts/gpu_ci/upload /opt/fastvideo-gpu-ci/upload
```

The controller needs Python 3.10 or newer, PyYAML,
the Buildkite agent, and `kubectl`; Modal additionally needs the reviewed
Modal SDK and a controller-side Modal credential.

Copy [the configuration example](../../scripts/gpu_ci/config.example.json)
to `/etc/fastvideo-gpu-ci.json`. Keep the installation and configuration
operator-owned and unwritable by workers. Configure the actual legacy
`slurm_uploader` command, replace each enabled backend's image placeholder
with a reviewed registry digest, and leave `default_backend` as `slurm`
during canaries. The checked-in placeholders deliberately cannot start jobs.

The permanent dispatcher must reach the Kubernetes API independently of a
developer laptop, SSH tunnel, or Tailscale session. The existing CPU development
pod is useful for investigating access, but is not a production CI service.
Use a dedicated controller identity and a separate `gpu-ci-dispatch` Buildkite
queue. The trusted bootstrap also needs access to the existing Slurm uploader
while that backend remains enabled.

Run all coordinators on **one host**, with `state_path` and `artifacts_dir` on
its persistent local disk. The implementation uses SQLite transactions and
local file locks. Do not place the database or locks on Lustre/NFS, run
independent database copies, or scale controller replicas across nodes. Those
configurations would invalidate the global limits. Multiple Buildkite agent
processes on the same host can share the installation and ledger; enough
agents are needed for concurrent suite coordinators and queued work.

Configure agent-owned hooks to skip repository checkout and reject commands
other than the trusted uploader and coordinator. Disable repository hooks
and plugins on this queue. Do not use a PR checkout to load `scripts/gpu_ci`,
its configuration, or its worker entrypoint. Build metadata is validated by
the dispatcher; arbitrary environment variables are not forwarded into GPU
workers. Buildkite, Kubernetes, and Modal control-plane credentials stay on
the dispatcher.

For Kubernetes, provision namespace-scoped permissions to create/get/delete
Jobs and read Pods and their logs. Worker Pods disable service-account token
mounting and request the exact GPU count on ARM64 GB200 nodes. The image must
include the expected `/opt/venv` runtime, CUDA/SM100 support, and the reviewed
FA4 dependencies. Use a separate AMD64 image digest for Modal. The Modal
profile preserves H100 for encoder, custom-kernel, and VSA lanes, and L40S
for the remaining lanes. Its reviewed image must support both SM89 and
SM90a. Attention settings are selected per backend and lane to preserve the
existing Modal and GB200 FA4 profiles; do not replace them with one global
attention override. Validate the images against their respective hardware
before enabling merge gates.

Optional Kubernetes configuration includes `context`, `hf_secret`,
`hf_secret_key`, `cache_pvc`, `cache_subpath`, `artifacts_pvc`, and
`artifacts_subpath`. Only provide a read-only Hugging Face credential when
private/gated downloads require it. PR workers are untrusted and can access
any credential supplied to them, so never provide a reference-publication
or other write-capable token. Prefer a pre-populated read-only cache.

Use dedicated CI PVCs and relative CI subpaths. The personal
`lustre-pvc-vllm` and `nfs-pvc-vllm` claims are rejected. The cache is mounted
read-only; mutable references and locks stay inside the worker. The artifact
init container prepares each Job's dedicated artifact subdirectory before
the worker mounts it. An
operator quota or admission policy scoped to CI can provide a second GPU
ceiling; do not apply an eight-GPU quota to the shared `vllm` namespace if it
would also cap other users' work.

## Wire Every Buildkite Entry Pipeline

Set each pipeline's operator-owned bootstrap command to
`/opt/fastvideo-gpu-ci/upload`. Apply this to all three entry pipelines:

| Pipeline | Trigger and scope |
|---|---|
| `pr-fastcheck` | Automatic PR webhook; `TEST_SCOPE=fastcheck` or unset. |
| `ci` | Existing API triggers for Fastcheck, full, merge, direct, and scheduled SSIM. Keep its incoming PR webhook disabled to avoid duplicate builds. |
| `fastvideo-performance-lane` | Existing schedule; `TEST_SCOPE=direct`, `TEST_TYPE=performance`. |

The wrapper resolves `CI_GPU_BACKEND` from the build environment, falling
back to `default_backend` in the trusted configuration. Allowed values are
exactly `slurm`, `modal`, and `vllm`; unknown values fail. For Slurm, upload
delegates to the unchanged private uploader. For Modal or `vllm`, it uploads
one fixed `/opt/fastvideo-gpu-ci/run` command on the dedicated queue, with
the selected backend and scope pinned in step environment.

Set `CI_GPU_BACKEND=vllm` in the Buildkite build environment for a rack canary,
or `modal` for a Modal canary. The existing API build payload can carry the
same string in its `env` object. Automatic PR webhook builds inherit the
operator's configured default unless pipeline/build configuration explicitly
overrides it. Do not put backend selection inside a PR-controlled command.

Keep the existing exact `BUILDKITE_COMMIT`, repository, PR identity, and
`TEST_SCOPE` metadata. Direct runs also need an allowlisted `TEST_TYPE`.
Merge runs need the trusted base-branch planner's `MERGE_TEST_PLAN`,
`MERGE_GOLDEN_TESTS`, and `MERGE_SSIM_TESTS`. Full/direct/scheduled quality
runs keep their complete matrices. The worker fetches and verifies the
immutable commit before installing dependencies or invoking a lane script.

The new Modal adapter uses bounded one-to-four-GPU sandboxes and the same
lane scripts as Kubernetes. It does not reactivate `pr_test.sh` or the old
Modal SSIM fan-out, which cannot enforce this shared four-GPU-per-PR budget.
The legacy files remain available for their existing manual workflows.

## Statuses and Default-Backend Cutover

The selected adapter reports separate suite contexts:

| Scope | GitHub context |
|---|---|
| Fastcheck | `gpu-ci/<backend>/fastcheck-passed` |
| Merge or explicit full suite | `gpu-ci/<backend>/full-suite-passed` |
| Direct lane | `gpu-ci/<backend>/direct-test-completed` |
| Scheduled SSIM | `gpu-ci/<backend>/scheduled-ssim-passed` |

The repository variable `CI_GPU_BACKEND` controls which new backend can
publish the existing required `fastcheck-passed` and `full-suite-passed`
contexts. Empty or `slurm` preserves the existing status behavior. `modal`
or `vllm` enables `ci-gpu-backend-status.yml`, which reads the latest statuses
for that one backend and mirrors both required contexts. A missing result
becomes pending; failure and error remain failures. It never combines one
backend's Fastcheck with another backend's full-suite result. Late status
events re-read current state instead of replaying stale event payloads.

Direct tests are diagnostic and do not promote a whole suite on the new
backends. Rerun the matching Fastcheck or merge/full suite to clear its gate.
Per-build tests on the other new backend remain separate diagnostics. The
legacy direct-test aggregation workflow is disabled while a new backend is
selected. These workflows retain the existing trust assumption that only
authorized status-writing integrations can publish CI status contexts.

For production cutover, drain existing Slurm builds: its preserved pipeline
still emits canonical status contexts. The new uploader rejects a Slurm
override when `default_backend` is `modal` or `vllm`, preventing later legacy
builds from overwriting the promoted backend's checks. Ensure no old bootstrap
bypasses the new uploader. Synchronize the operator `default_backend` and
the GitHub repository variable, then clear or rerun required checks for all
open PRs. Old green contexts do not become new-backend validation merely
because a setting changed. Run both Fastcheck and the merge/full suite on
the promoted backend before allowing merge. Apply the same drain and rerun
procedure when rolling back to Slurm.

## Validation and Recovery

Inspect the rendered opt-in pipeline without submitting a build:

```bash
CI_GPU_BACKEND=vllm TEST_SCOPE=fastcheck \
  /opt/fastvideo-gpu-ci/venv/bin/python -I \
  /opt/fastvideo-gpu-ci/source/scripts/gpu_ci/entrypoint.py render \
  --config /etc/fastvideo-gpu-ci.json
```

Validate a one-GPU lane, then a two-/four-GPU lane and the complete Fastcheck
suite. Exercise two PRs plus a third waiter, concurrent builds of the same
PR, cancellation, retries, and coordinator restart. Confirm observed GPU
reservations never exceed two PRs, four per PR, or eight in total. A GPU
reservation includes a pending worker, so unavailable nodes cannot cause
the controller to submit more work than its allowance.

Run SSIM, training, and performance canaries separately. References must
match the effective GPU/runtime/attention backend; do not silently reuse
L40S performance results as GB200 baselines or reseed references as part of
routine CI. Workers keep W&B offline and disable reference publication.

Inspect the persistent ledger and recover an abandoned attempt with:

```bash
/opt/fastvideo-gpu-ci/venv/bin/python -I \
  /opt/fastvideo-gpu-ci/source/scripts/gpu_ci/entrypoint.py status \
  --config /etc/fastvideo-gpu-ci.json

/opt/fastvideo-gpu-ci/venv/bin/python -I \
  /opt/fastvideo-gpu-ci/source/scripts/gpu_ci/entrypoint.py recover \
  --config /etc/fastvideo-gpu-ci.json --build-id BUILD_ID.JOB_ID
```

Use the exact ledger ID from `status`; each Buildkite retry has a distinct
job ID. Recovery refuses a live coordinator, persists cancellation, stops
owned resources, and releases reservations only after confirming termination.
If creation or cleanup is ambiguous, investigate the recorded handle on its
backend and retain the reservation until the outcome is known. Do not delete
the database or manually zero counters to unblock the queue.

The dispatcher uploads controller logs, per-lane numeric results, request
metadata, and the suite summary to Buildkite. These are the sources for its
exit status. Generated videos and JUnit files stay in the worker unless a
dedicated Kubernetes artifact PVC is configured; automatic publication of
those worker files and Modal worker artifacts is not implemented. This is
a deployment limitation to account for before replacing existing artifact
review workflows.
