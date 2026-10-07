---
date: 2026-09-29
experiment: Wan-VACE multi-GPU inference on shared Slurm nodes
category: infrastructure
severity: important
---

# Spawn workers fail with ENOENT in `SemLock._rebuild`

## What Happened

On Slurm nodes, standard `VideoGenerator` runs with the `mp` executor sometimes
failed during worker startup. A spawn child raised `FileNotFoundError` from
`multiprocessing.synchronize.SemLock._rebuild` while unpickling its arguments,
even though the parent still held its Queue objects. Failures showed up on
several nodes and looked intermittent.

## Root Cause

The cluster's Slurm epilog `80-epilog-cleanup-shm-tmp.sh` runs
`find /dev/shm -maxdepth 2 -user "$SLURM_JOB_USER" -delete` at the end of
**every** job. When one of a user's jobs ends, the epilog deletes that user's
`/dev/shm` files on the node, including those that the user's *other* running
jobs are still using. Python's POSIX named semaphores (`/dev/shm/sem.mp-*`) are
among them. If the deletion lands between queue creation and a spawn child
unpickling it, the child fails with ENOENT.

Evidence:

- **Timing.** Every recorded deletion burst during a probe job matched, to the
  second, the end of another job by the same user on the same node
  (`sacct -u $USER -N <node>` end times).
- **Controlled reproduction.**
  - Job A held a spawn `Lock` semaphore and a marker file in `/dev/shm`.
  - Job B, on the same node, started and then ended.
  - Job B's start deleted nothing.
  - Within 1 s of job B's end, both of job A's objects were gone, and job A's
    finalizer then hit the same ENOENT.
- **Still unexplained.** One historical failure has no same-user job
  end in its window, so another deleter cannot be ruled out for it. logind
  `RemoveIPC` was the earlier hypothesis; it is not needed to explain the other
  cases.

## Fix / Workaround

- **FastVideo hardening.** Standard inference no longer creates the two
  streaming queues it never uses. `FastVideoArgs.enable_streaming_ipc_queues`
  defaults to `False`, and `StreamingVideoGenerator` sets it to `True`. This
  removes the exposure for standard inference only.
- **Still exposed.** Streaming queues, NCCL shared-memory segments, and
  DataLoader shared memory on the same node can still be deleted by the epilog.
- **Cluster fix (administrators).** Clean `/dev/shm` only when the user has no
  other job on the node, or give each job a private `/dev/shm` with
  `JobContainerType=job_container/tmpfs`.
- **User workaround.** Don't co-locate your own jobs on one node
  (`--exclusive=user`), or avoid ending short jobs next to long-running ones.

## Prevention

- Before blaming an application for `/dev/shm` ENOENT on Slurm, read the
  node-local prolog/epilog hooks (`/cm/local/apps/slurm/var/{prologs,epilogs}`).
  Then correlate the failure window with `sacct -u $USER -N <node>` job end
  times.
- Do not pass IPC primitives to spawn workers unless the worker needs them.
- `fastvideo/tests/worker/test_multiproc_executor.py` checks that standard
  workers survive semaphore removal during spawn.
