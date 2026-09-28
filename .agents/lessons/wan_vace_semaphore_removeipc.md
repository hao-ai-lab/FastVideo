# Wan-VACE worker semaphore ENOENT (RemoveIPC + unnecessary streaming queues)

## Symptom

Multiprocessing `SemLock._rebuild` fails with ENOENT during Wan-VACE worker startup on
Slurm GPU nodes. Six named semaphores under `/dev/shm/sem.mp-*` (two streaming
`ctx.Queue()` × three locks each) vanish ~57–60s after creation, before slow-import
workers rebuild them.

## Root cause (2026-09-28)

1. **Primary (environment)**: systemd logind **RemoveIPC=yes** (default when unset) removes
   all IPC objects for a UID when the user's last **logind** session ends. Batch jobs often
   have **no logind session** (`/run/user/$UID` absent, `loginctl list-sessions` → none).
   A concurrent SSH/`srun --pty` logout on the same node can delete semaphores still needed
   by the batch worker. Instrumented runs showed DELETE via inotify with **no** Python
   `sem_unlink` / audit-hook events (jobs 946472, 943386).

2. **Contributing (application)**: `MultiprocExecutor` always created streaming IPC queues
   even for standard `VideoGenerator` inference (Wan-VACE), spawning six avoidable named
   semaphores pickled into workers. Fixed in `feb9b289` via
   `FastVideoArgs.enable_streaming_ipc_queues` (default `False`; `StreamingVideoGenerator`
   sets `True` before worker spawn).

Failures were **not** limited to a single node (`dgx-3nc-05-24`, `dgx-3nc-07-34`, others).

## What we ruled in

- **Orderly shutdown**: strace job 946649 — eight expected `sem_unlink` from
  `semlock_finalizer` only; no rebuild failures.
- **In-process Python unlink before failure (946472)**: no matching diag/shim record before
  simultaneous disappearance.

## Mitigations

| Audience | Action |
|----------|--------|
| Wan-VACE / standard inference | Use build with `enable_streaming_ipc_queues=False` (default) — no code change needed beyond `feb9b289`. |
| Streaming queue mode (MatrixGame2) | Prefer `loginctl enable-linger $USER` or cluster `RemoveIPC=no`; see `.codex-local/reports/VACE_ADMIN_AUDIT_HANDOFF_A4_20260928.md`. |
| Admins | Audit epilog/prolog if inotify DELETE correlates with root cleanup processes (ps snapshot in `watch_shm_deletes.py`). |

## Diagnostics (`.codex-local/`)

- `watch_shm_deletes.py` — CREATE/DELETE + ps snapshot on DELETE
- `shm_unlink_shim.so` — log all `sem_unlink`/`shm_unlink`; filter `unlink` paths
- `vace_ipc_audit.py` — `sys.addaudithook` for `/dev/shm` removals
- `sample_logind_sessions.py` — logind session + RemoveIPC config sampling
- `repro_spawn_queue_semaphores.py` — minimal CPU repro (spawn Queue + delayed workers)
- `run_vace_semaphore_local_loop.py` — local Wan-VACE SP2 session loop

Semaphore stress loops set `VACE_SKIP_FROZEN_IDENTITY=1` so frozen scope recipes can run
after intentional production fixes.

## References

- `.codex-local/reports/VACE_SEMAPHORE_ROOT_CAUSE_20260928.md`
- `.codex-local/reports/VACE_REMOVEIPC_A1_20260928.md`
- `.codex-local/reports/VACE_ADMIN_AUDIT_HANDOFF_A4_20260928.md`
- `docs/inference/wan_vace.md` (Known Gaps)
