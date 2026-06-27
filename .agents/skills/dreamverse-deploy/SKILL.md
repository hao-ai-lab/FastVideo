---
name: dreamverse-deploy
description: Redeploy the local Dreamverse backend and frontend on a chosen physical GPU, wait for readiness, inspect logs, or stop the managed stack. Use for local Dreamverse GPU development and Playwright verification; do not use for Modal or production image deployment.
---

# Dreamverse Deploy

Use the bundled script instead of reconstructing the process lifecycle by
hand. It delegates service startup to the maintained launchers under
`apps/dreamverse/scripts/launch/` and keeps transient PIDs and logs under the
gitignored `.agents/tmp/` directory.

## Prerequisites

- A Linux GPU host with `nvidia-smi`, `setsid`, `curl`, and either `lsof` or
  `fuser`.
- `dreamverse-server` installed from this checkout; if missing, run
  `uv pip install -e ".[dreamverse]"`.
- `npm` available (or set `NPM=/path/to/npm`). The frontend launcher runs
  `npm ci` when `node_modules/` is absent.
- Provider credentials in the environment or `~/.env` when prompt rewriting
  needs them.

Native ffmpeg is optional. Build it with
`apps/dreamverse/scripts/install_native_ffmpeg.sh`; the deploy script sources
the generated `apps/dreamverse/scripts/ffmpeg-env.sh` automatically.
On B200 hosts where NVCC rejects the active GCC, export the host-specific
GCC 13 variables documented in `docs/contributing/dreamverse-development.md`
before running the helper; the skill does not force one compiler globally.

## Deploy

Run from the repository root:

```bash
# Fast local development defaults: backend 8009, frontend 5299, no warmup or
# torch.compile.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 4

# Custom ports.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh 6 8089 5300

# Production-like startup.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh \
  --warmup --torch-compile 4

# Inspect the resolved plan without stopping or launching anything.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --dry-run 4
```

The first positional argument is the physical GPU index. Optional second and
third arguments override the backend and frontend ports.

Supported flags:

| Flag | Behavior |
|---|---|
| `--warmup` / `--no-warmup` | Enable or disable startup warmup; default off |
| `--torch-compile` / `--no-torch-compile` | Enable or disable `torch.compile`; default off |
| `--nvenc` / `--no-nvenc` | Select `h264_nvenc` or `libx264`; NVENC is probed before launch |
| `--force-gpu-kill` | Terminate all compute processes on the selected GPU before launch |
| `--dry-run` | Print the resolved deployment without changing processes |
| `--stop` | Stop the managed stack and listeners on the selected ports |

Environment defaults are available as `DREAMVERSE_WARMUP`,
`DREAMVERSE_TORCH_COMPILE`, `DREAMVERSE_NVENC`,
`DREAMVERSE_BACKEND_PORT`, and `DREAMVERSE_FRONTEND_PORT`. Override the
repository or artifact root with `DREAMVERSE_REPO_ROOT` or
`DREAMVERSE_DEPLOY_DIR`.

## Safety and readiness

Redeploy stops the previously managed process group and any listeners on the
chosen backend/frontend ports. It does not kill unrelated GPU processes. If
the selected GPU is still occupied, report the PIDs and stop; use
`--force-gpu-kill` only after confirming those processes are disposable.

Success requires both:

- `http://127.0.0.1:<backend-port>/readyz`
- `http://127.0.0.1:<frontend-port>/`

Report the URLs, managed PID, and log paths printed by the script. On failure,
inspect `stack.log`, `demo-be.log`, and `demo-fe.log` in the reported
`.agents/tmp/dreamverse-deploy/` instance directory.

## Stop

```bash
# Default ports 8009 and 5299.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --stop

# Custom backend and frontend ports.
./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --stop 8089 5300
```

## Playwright verification

After a successful deploy:

```bash
cd apps/dreamverse/web
PLAYWRIGHT_SKIP_WEBSERVER=1 \
  BACKEND_HOST=127.0.0.1 \
  BACKEND_PORT=8009 \
  PLAYWRIGHT_BASE_URL=http://127.0.0.1:5299 \
  NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 \
  npm exec -- playwright test
```

Set `PLAYWRIGHT_LONG_RUNNING=1` and target
`e2e/long-running-segments.spec.ts` for the real two-segment GPU regression.

## Deployment boundary

This skill is for a local checkout on a directly attached GPU. For a
container image, use `apps/dreamverse/docker/README.md`. For Modal, follow
`apps/dreamverse/scripts/modal/README.md`; do not adapt this process-killing
workflow to a remote deployment.

## References

- `apps/dreamverse/AGENTS.md`
- `apps/dreamverse/scripts/launch/README.md`
- `docs/contributing/dreamverse-development.md`
