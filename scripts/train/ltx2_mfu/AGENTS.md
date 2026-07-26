# LTX-2 MFU tracker

This directory is the resumable lab notebook for PR #1630. Preserve historical scripts and measured results; add new attempts rather than rewriting old evidence.

- Read `README.md` and `REPORT.md` before running a gate.
- Use the benchmark contract in `README.md`. Label any precision, batch, topology, or source change as a different baseline.
- Compare performance only within one healthy allocation, preferably A/B/A. Record slowest-rank step time after warmup.
- Never commit credentials, checkpoints, videos, raw logs, profiler traces, copied third-party source, or generated binaries. Record W&B/job IDs, source versions, and hashes instead.
- Put end-to-end harnesses in `harness/`, launch snapshots in `runners/`, focused experiments in `probes/`, and compact conclusions in `REPORT.md` or `reports/`.
- Update `README.md` with the current stopping point and `REPORT.md` with accepted and rejected results before handing off.
- Treat files below as frozen experiment snapshots unless a rerun explicitly supersedes one; production changes belong in the normal package directories and should later be extracted into focused PRs.
