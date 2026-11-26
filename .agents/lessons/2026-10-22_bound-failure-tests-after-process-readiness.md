---
date: 2026-10-22
experiment: CPU worker lifecycle and real-signal regression tests
category: infrastructure
severity: important
---

# Start failure-response deadlines after process readiness

## What Happened

Real two-rank Gloo and spawn-worker signal tests passed in isolation but
failed intermittently in a complete unit run on a shared filesystem. A
30-second subprocess deadline expired before the intended failure was
injected. Sometimes the first rank instead exhausted Gloo's three-second
timeout while its peer was still importing FastVideo.

## Root Cause

Importing PyTorch and FastVideo in a fresh interpreter is preparation, not
failure-response latency. The original tests counted both against one
deadline. Two rank imports can finish several seconds apart, so even a
correctly bounded collective can fail before the scenario under test starts.
Killing only torchrun after a parent-side timeout can also leave its ranks
alive because those processes have their own sessions.

## Fix / Workaround

The external-launcher smoke now has two parent-controlled handshakes: both
ranks reach process-group initialization, then both report executor readiness.
One bounded preparation budget covers both phases. Only after preparation
does the parent release the operation, keeping the existing failure-response
budget and real Gloo timeout. The signal probe separately acknowledges entry
after spawn imports; its signal-point, exit and traceback bounds remain.
The spawned execute-forward response probe does the same before its first
request, so each request keeps its original response deadline.

Timeout cleanup snapshots and reaps descendant ranks as well as the launcher.
Harness tests use a fake clock to prove that slow preparation does not consume
the operation budget, that either deadline still fails, that descendants are
cleaned up, and that startup exits retain diagnostics without releasing faults.

The same boundary applies to subprocess API clients. Load the real SDK and
acknowledge readiness before starting the existing request/poll/download
deadline. Run the shipped client unchanged after the parent releases it,
and reap its process group on timeout. Cover startup and request deadlines
separately with a fake clock, then exercise the real client/server routes.

## Prevention

- Keep preparation, rendezvous and injected-failure timings distinguishable.
- Gate fault injection on every participant's readiness; do not increase a
  process group's timeout to accommodate imports.
- Keep a separate finite preparation limit; missing readiness must still fail.
- Exercise the real two-process and real-signal paths after harness changes.
- Relevant regressions live in `fastvideo/tests/worker/test_external_launcher_executor.py`,
  `fastvideo/tests/worker/test_signal_terminated_worker_is_reported.py` and
  `fastvideo/tests/worker/test_multiproc_executor.py`.
