# Wan-VACE worker semaphore ENOENT (RemoveIPC hypothesis)

## Symptom

Multiprocessing `SemLock._rebuild` fails with ENOENT during Wan-VACE worker startup on
Slurm GPU nodes. Named semaphores under `/dev/shm/sem.*` vanish between creation and reuse.

## What we ruled in

- **Orderly shutdown path**: strace job 946649 on `dgx-3nc-12-24` shows eight expected
  `sem_unlink` calls from Python `semlock_finalizer` only; no rebuild failures.
- **In-process Python unlink before failure (946472)**: `vace_sem_diag` did not record a
  matching successful unlink of the six failing names before rebuild.

## Leading hypothesis (unproven on fault node)

systemd **RemoveIPC=yes** (default when uncommented in `logind.conf`) removes all IPC objects
for a UID when the user's last **logind** session ends. A concurrent SSH/`srun --pty` session
ending on the same node can delete semaphores still needed by a batch worker.

## What not to do yet

- Do not change Queue ownership, resource_tracker, or model code based on a single non-repro
  success run (946649).
- Do not treat strace/ptrace campaigns as performance evidence.

## Next diagnostic steps

1. On the **fault node** (`dgx-3nc-05-24`), confirm `RemoveIPC` and correlate logind journal
   with the 14:33:50 UTC failure window.
2. Run worker under `.codex-local/scripts/watch_shm_deletes.py` (inotify) and
   `LD_PRELOAD=.../shm_unlink_shim.so` together; if inotify fires but shim is silent, suspect
   external/systemd deletion (A4 admin audit).
3. Environment mitigations if confirmed: `loginctl enable-linger` for the batch UID, or cluster
   `RemoveIPC=no` — not application code changes.

## References

- `.codex-local/reports/VACE_SEMAPHORE_DIAGNOSTIC_20260928.md` (job 946472)
- `.codex-local/reports/VACE_SYSCALL_DIAGNOSTIC_20260928.md` (job 946649)
- `.codex-local/reports/VACE_REMOVEIPC_A1_20260928.md`
