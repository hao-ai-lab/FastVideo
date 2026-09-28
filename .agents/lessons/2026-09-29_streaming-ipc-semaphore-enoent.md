---
date: 2026-09-29
experiment: Wan-VACE multi-GPU inference on shared Slurm nodes
category: infrastructure
severity: important
---

# Spawn workers fail with ENOENT in `SemLock._rebuild`

## What Happened

Standard `VideoGenerator` runs with the `mp` executor sometimes failed during
worker startup: a spawn child raised `FileNotFoundError` from
`multiprocessing.synchronize.SemLock._rebuild` while unpickling its arguments.
The failures appeared on several Slurm nodes and were intermittent. The parent
process still held its Queue objects.

## Root Cause

`MultiprocExecutor` always created two streaming `multiprocessing.Queue`s and
passed them to every worker. They are backed by six POSIX named semaphores
(`/dev/shm/sem.*`). If something deletes those names before a spawn child
deserializes them, `SemLock._rebuild` fails with ENOENT. A test can reproduce
this by holding the child at the import barrier and unlinking the semaphores.

The external deleter was not identified. RemoveIPC and session teardown are
still hypotheses. Also note that these are POSIX named semaphores, not
System V ones.

## Fix / Workaround

The queues are needed only for streaming, so standard inference no longer
creates them. `FastVideoArgs.enable_streaming_ipc_queues` defaults to `False`,
and `StreamingVideoGenerator` sets it to `True` before workers spawn.
`MultiprocExecutor.enable_streaming()` raises when the queues are missing.
This removes the exposure for standard inference. It does not stop an external
process from deleting IPC objects, so streaming runs remain exposed.

## Prevention

- Do not pass IPC primitives to spawn workers unless the worker needs them.
- `fastvideo/tests/worker/test_multiproc_executor.py` checks that standard
  workers survive semaphore removal during spawn.
- Treat a passing stress run as evidence for the mitigation only, not as a
  confirmed root cause. `inotify` names the deleted object, not the deleter.
  Blame a process only after its unlink call is observed to succeed.
- Make failed tests fail the job: a trailing `|| echo` in a Slurm script hides
  non-zero exits.
