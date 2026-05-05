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
- FastVideo `.venv` populated with `flashinfer-python`, `cerebras-cloud-sdk`, `openai`
- `~/.env` exporting `CEREBRAS_API_KEY`, `GROQ_API_KEY`, etc.
- pnpm installed at `/home/william5lin/.local/share/pnpm/pnpm` (or in `$PATH`)
- `gcc-13` + `g++-13` at `/usr/bin/` (workaround for nvcc gcc-15 rejection)

If any prereq is missing, the script fails fast with a clear message.

## Usage

```bash
# Deploy on GPU 4 with default ports (backend 8009, FE 5274)
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 4

# Deploy on GPU 6 with custom ports
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 6 8089 5275

# Deploy on GPU 0 with warmup enabled (default disabled for fast iter)
DREAMVERSE_WARMUP=true ./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 0
```

### Arguments

| Position | Name | Default | Notes |
|---|---|---|---|
| 1 | `GPU` | (required) | Physical GPU index, e.g. `4` |
| 2 | `BACKEND_PORT` | `8009` | TCP port for the FastAPI server |
| 3 | `FRONTEND_PORT` | `5274` | TCP port for the Next.js dev server |

### Environment variables (override defaults)

| Var | Default | Purpose |
|---|---|---|
| `DREAMVERSE_WARMUP` | `false` | If `true`, runs the GPU warmup at boot (~minutes) |
| `DREAMVERSE_REPO_ROOT` | git rev-parse | Repo root override |
| `DREAMVERSE_LOG_DIR` | `/tmp/opencode/dreamverse-deploy` | Where to write `backend.log` / `frontend.log` |

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
   - `CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13`
   - `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13 -allow-unsupported-compiler"`
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
