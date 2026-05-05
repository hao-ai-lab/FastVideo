# Dreamverse Development

Dreamverse lives under `apps/dreamverse/` as a product app inside the
FastVideo monorepo. Backend code uses the local FastVideo workspace package;
frontend tooling remains standalone under `apps/dreamverse/web/`.

## Backend tests

Run CPU-safe backend tests from the FastVideo repository root:

```bash
uv run --locked --package dreamverse --extra test pytest apps/dreamverse/server/tests/ -m 'not gpu' -q
```

## Frontend build and tests

Run frontend commands from the standalone web app:

```bash
cd apps/dreamverse/web
pnpm install --frozen-lockfile
pnpm run build
pnpm run test
```

Playwright is intentionally run against a live backend as part of the GPU4
manual verification flow, not in the Phase 3 migration gate.

## Local GPU4 verification hook

Use physical GPU 4 for migration smoke tests. `CUDA_VISIBLE_DEVICES=4` makes
that GPU appear as logical GPU 0 inside the process, preserving the previous
Dreamverse deployment behavior.

```bash
CUDA_VISIBLE_DEVICES=4 uv run --locked --package dreamverse \
  fastvideo serve --config apps/dreamverse/serve_configs/streaming_demo.yaml \
  --host 0.0.0.0 --port 8009
```

In another shell, verify the service:

```bash
curl -s http://localhost:8009/health
```

Phase 4 adds the public `/healthz`, `/readyz`, `/status`,
`/prompt-system-config`, and `/curated-presets` route coverage needed for the
full Playwright suite.

## Phase 0 production-equivalent prerequisites

For the production-equivalent NVFP4 path, install these optional dependencies
in the FastVideo `.venv` before GPU4 smoke tests:

```bash
.venv/bin/pip install flashinfer-python flash-attn --no-build-isolation
```

`flashinfer-python` is required for NVFP4 quantization. `flash-attn` is
optional but recommended; without it, attention falls back to Torch SDPA.
