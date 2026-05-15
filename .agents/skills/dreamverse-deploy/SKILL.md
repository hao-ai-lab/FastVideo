---
name: dreamverse-deploy
description: Use when redeploying the migrated Dreamverse app backend and frontend on a chosen local GPU; tears down existing ports, launches services, and waits for readiness checks.
---

# dreamverse-deploy — redeploy migrated Dreamverse on a chosen GPU

**Scope:** project (lives in this repo at `.agents/skills/dreamverse-deploy/`)

**When to use:** you want to (re)launch the migrated `apps/dreamverse/` backend
+ frontend on this dev node, pinned to a specific physical GPU. Tears down
any existing deploy on the same ports first, then boots fresh and waits for
both `/readyz` and the FE root to return 200.

**Pairs with:** [`integration-plan.md`](../../memory/dreamverse-integration/integration-plan.md)
"Local GPU4 verification hook" + [`decisions-log.md D-19`](../../memory/dreamverse-integration/decisions-log.md#d-19).

## Prerequisites

- Working tree on a branch that has `apps/dreamverse/` (e.g. `will/dreamverse-monorepo`)
- Local conda env at `~/miniconda3/envs/fv-main/` with `flashinfer-python`,
  `cerebras-cloud-sdk`, `openai` installed (override the default path with
  `DREAMVERSE_PYTHON=/path/to/python`)
- `~/.env` exporting `CEREBRAS_API_KEY`, `GROQ_API_KEY`, etc.
- pnpm installed at `/home/william5lin/.local/share/pnpm/pnpm` (or in `$PATH`)
- `gcc-13` + `g++-13` at `/usr/bin/` (workaround for nvcc gcc-15 rejection)
- **Recommended:** native ffmpeg env file at `apps/dreamverse/scripts/ffmpeg-env.sh`
  (built once via `bash apps/dreamverse/scripts/install_native_ffmpeg.sh`).
  When present, the deploy sources it inside the backend setsid block so the
  worker spawns ffmpeg from `$HOME/opt/ffmpeg-native/bin/ffmpeg` (LTO + libx264
  + native arch) instead of the system `/usr/bin/ffmpeg`. When missing, the
  deploy falls back to system ffmpeg with a warning. Set
  `DREAMVERSE_REQUIRE_NATIVE_FFMPEG=true` to make the missing env file a hard
  failure.

If any required prereq is missing, the script fails fast with a clear message.

## Usage

```bash
# Deploy on GPU 4 with default ports (backend 8009, FE 5274) — torch.compile
# and warmup are both OFF by default so first-segment cold start is ~45s
# instead of ~3-4min.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 4

# Deploy on GPU 6 with custom ports
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 6 8089 5275

# Deploy on GPU 0 with warmup enabled
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --warmup 0

# Deploy with torch.compile enabled (max-autotune; first segment ~3-4min,
# subsequent segments save ~3s — only worth it for benchmarking)
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --torch-compile 4

# Deploy with both warmup AND torch.compile enabled
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --warmup --torch-compile 4

# Flags can appear before, between, or after positional args
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 4 8089 5275 --warmup
```

### Arguments

| Position | Name | Default | Notes |
|---|---|---|---|
| 1 | `GPU` | (required) | Physical GPU index, e.g. `4` |
| 2 | `BACKEND_PORT` | `8009` | TCP port for the FastAPI server |
| 3 | `FRONTEND_PORT` | `5274` | TCP port for the Next.js dev server |

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--warmup` / `--no-warmup` | off | Run GPU warmup at boot (~minutes). Overrides `DREAMVERSE_WARMUP` |
| `--torch-compile` / `--no-torch-compile` | off | Enable max-autotune `torch.compile`. First segment ~3-4min when on, ~45s when off. Overrides `DREAMVERSE_TORCH_COMPILE` |
| `--nvenc` / `--no-nvenc` | off | Use `h264_nvenc` hardware encoder instead of `libx264` software. Eliminates ~1100ms/segment of CPU encoding cost (raises realtime ratio from ~0.78x → ≥1.0x, eliminating inter-segment buffer-drain stutter). Requires native ffmpeg built with `--enable-nvenc` (the install script's default since the NVENC update). Hard-fails up-front if the binary is missing or lacks NVENC. Overrides `DREAMVERSE_NVENC` |
| `-h` / `--help` | — | Show usage |

Flags can appear in any position relative to the positional args. Explicit flag values always win over env-var defaults.

### Environment variables (used when no flag is given)

| Var | Default | Purpose |
|---|---|---|
| `DREAMVERSE_WARMUP` | `false` | Same as `--warmup`/`--no-warmup`. Flag takes precedence |
| `DREAMVERSE_TORCH_COMPILE` | `false` | Same as `--torch-compile`/`--no-torch-compile`. Flag takes precedence |
| `DREAMVERSE_NVENC` | `false` | Same as `--nvenc`/`--no-nvenc`. Flag takes precedence |
| `DREAMVERSE_PYTHON` | `~/miniconda3/envs/fv-main/bin/python` | Conda env python used for prereq probes (flashinfer import). The wrapper at `apps/dreamverse/scripts/dreamverse-server` still resolves python via the `.venv` symlink, which points at the same interpreter on this dev node |
| `DREAMVERSE_REPO_ROOT` | git rev-parse | Repo root override |
| `DREAMVERSE_LOG_DIR` | `/tmp/opencode/dreamverse-deploy` | Where to write `backend.log` / `frontend.log` |
| `DREAMVERSE_REQUIRE_NATIVE_FFMPEG` | `false` | If `true`, fail when `$HOME/opt/ffmpeg-native/bin/ffmpeg` is absent |

## What it does

1. Validates prereqs.
2. Kills any process on the target backend/frontend ports + waits for the
   target GPU to release memory (allows up to 30s for cleanup).
3. Sources `~/.env`.
4. Exports the env recipe required for boot:
   - `CUDA_VISIBLE_DEVICES=<gpu>`
   - `FASTVIDEO_ENABLE_DEVTOOLS=1`
   - `FASTVIDEO_ENABLE_STARTUP_WARMUP=<DREAMVERSE_WARMUP>`
   - `FASTVIDEO_GPU_COUNT=1`
   - `ENABLE_TORCH_COMPILE=<0|1 derived from DREAMVERSE_TORCH_COMPILE>`
   - `CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13`
   - `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13 -allow-unsupported-compiler"`
   - `FASTVIDEO_FFMPEG_BIN=$HOME/opt/ffmpeg-native/bin/ffmpeg` +
     `FASTVIDEO_VIDEO_CODEC=libx264` (when the native binary exists)
5. Launches the backend via `apps/dreamverse/scripts/dreamverse-server` in a
   detached `setsid` session, captures PID.
6. Polls `/readyz` until 200 (max 5 min).
7. Launches the frontend via `pnpm run dev:devtools` in a detached session,
   captures PID.
8. Polls FE `/` until 200 (max 60s).
9. Prints URLs, PIDs, and log paths.

## What it does NOT do

- Does not modify `~/.env` or the FastVideo `.venv`.
- Does not push code or commit anything.
- Does not run Playwright. Use the e2e wrapper separately:
  ```bash
  cd apps/dreamverse/web
  PLAYWRIGHT_SKIP_WEBSERVER=1 BACKEND_URL=http://127.0.0.1:8009 \
    PLAYWRIGHT_BASE_URL=http://127.0.0.1:5274 \
    NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 \
    pnpm exec playwright test
  ```
  The fast suite (8 specs, ~5s) runs by default; the long-running
  two-segment audio-continuation spec is gated behind
  `PLAYWRIGHT_LONG_RUNNING=1` (see below).

## Long-running e2e (paired with `--warmup --torch-compile`)

[`apps/dreamverse/web/e2e/long-running-segments.spec.ts`](../../../apps/dreamverse/web/e2e/long-running-segments.spec.ts)
drives a real two-segment session through the FE, captures every WS
frame, and asserts segments 1 AND 2 both reach `media_segment_complete`
with at least one binary fMP4 chunk per segment — the canonical
regression guard against the D-20 BrokenPipe pattern documented in
[`decisions-log.md D-20`](../../memory/dreamverse-integration/decisions-log.md#d-20).
Skipped by default. Enable with:

```bash
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh \
    --warmup --torch-compile 4

cd apps/dreamverse/web
PLAYWRIGHT_SKIP_WEBSERVER=1 \
  BACKEND_URL=http://127.0.0.1:8009 \
  PLAYWRIGHT_BASE_URL=http://127.0.0.1:5274 \
  NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 \
  PLAYWRIGHT_LONG_RUNNING=1 \
  pnpm exec playwright test e2e/long-running-segments.spec.ts
```

Expected runtime: ~7-9 minutes on a B200 (torch.compile max-autotune
warm-up dominates the cold start; per-test timeout is 900s). The spec
hard-fails on any WS `error`/`step_error` frame so the BrokenPipe
regression surfaces with the actual ffmpeg/audio diagnostics rather
than an opaque "test timed out".

## Teardown

Stop both services without redeploying:

```bash
# Stop services on default ports (port-pattern based)
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --stop

# Stop AND nuke any process holding GPU N
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --stop 4
```

The redeploy path (`<GPU>` mode) automatically nukes any process holding the
target GPU before launching — including orphan `multiproc_executor` worker
subprocesses left over from a parent backend that was killed without grace.
This was the failure mode of an earlier naive port-only kill: parent dies,
children survive, GPU stays full, next deploy OOMs.

## Notes

- The wrapper at `apps/dreamverse/scripts/dreamverse-server` is what makes
  the migrated `apps/dreamverse/server/main.py` run instead of the legacy
  conda-installed Dreamverse — see [decisions-log.md D-19](../../memory/dreamverse-integration/decisions-log.md#d-19) for why
  this matters.
- The B200 / sm_100a NVCC flags are mandatory on this dev node because the
  conda toolchain ships gcc-15, which nvcc rejects. If you're on a machine
  with a supported native gcc, those exports are still safe (no-op when the
  paths don't exist; the script verifies them upfront).
