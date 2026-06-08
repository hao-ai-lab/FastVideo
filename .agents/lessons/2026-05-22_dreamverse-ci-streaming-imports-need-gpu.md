---
date: 2026-05-22
experiment: PR #1386 DreamVerse app CI backend tests
category: infrastructure
severity: important
---

# DreamVerse App CI Streaming Imports Need GPU

## What Happened

DreamVerse app CI backend pytest collection imports FastVideo streaming surfaces.
When those tests run in a CPU-only Modal environment, collection can fail before
any app assertions run with Triton reporting:

```text
RuntimeError: 0 active drivers
```

## Root Cause

Some streaming import paths can import `fastvideo_kernel` at module import time.
Triton then probes for an active GPU driver during pytest collection. A CPU-only
Modal container has no active driver, so the failure appears as an import-time
collection error rather than a DreamVerse app behavior failure.

## Fix / Workaround

For PR #1386, use a surgical CI fix: allocate a GPU to
`run_dreamverse_app_tests`. Do not refactor core streaming/kernel imports just to
unstick this app CI path.

Keep `build_kernel=False` for this job. The DreamVerse app backend test imports
streaming surfaces but does not need to rebuild or exercise custom kernels.

## Prevention

When adding or modifying DreamVerse app CI jobs that import FastVideo streaming
modules, make the GPU requirement explicit if the import graph may touch
`fastvideo_kernel`. Prefer small CI resource fixes for app test collection issues
unless the product code genuinely requires lazy import cleanup.
