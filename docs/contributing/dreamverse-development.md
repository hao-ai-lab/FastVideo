# Dreamverse Development

Dreamverse lives under `apps/dreamverse/` as a product app inside the
FastVideo monorepo. Backend code uses the local FastVideo workspace package;
frontend tooling remains standalone under `apps/dreamverse/web/`.

## Backend tests

Run CPU-safe backend tests from the FastVideo repository root:

```bash
uv run --locked --package dreamverse --extra test pytest apps/dreamverse/server/tests/ -m 'not gpu' -q
```

## Backend launch

Launch the migrated backend through the installed console commands:

```bash
dreamverse-server --port 8009
dreamverse-mock-server --port 8009
```

If `dreamverse-server` is missing, install FastVideo with the `dreamverse`
extra from the checkout:

```bash
uv pip install -e ".[dreamverse]"
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
CUDA_VISIBLE_DEVICES=4 dreamverse-server --host 0.0.0.0 --port 8009
```

In another shell, verify the service:

```bash
curl -s http://localhost:8009/healthz
```

Phase 4 adds the public `/healthz`, `/readyz`, `/status`,
`/prompt-system-config`, and `/curated-presets` route coverage needed for the
full Playwright suite.

## Phase 0 production-equivalent prerequisites

For the production-equivalent NVFP4 path, install these dependencies
in the FastVideo `.venv` before GPU smoke tests:

```bash
uv pip install --python .venv/bin/python \
  flashinfer-python flash-attn cerebras-cloud-sdk openai \
  --no-build-isolation
```

| Package | Why |
|---|---|
| `flashinfer-python` | Required for NVFP4 quantization. Without it, model load fails with `ImportError: NVFP4 quantization requires flashinfer`. |
| `flash-attn` | Optional but recommended; without it attention falls back to Torch SDPA (functional but slower). |
| `cerebras-cloud-sdk` | Required by the migrated prompt enhancer for the default `cerebras` provider. |
| `openai` | Required by the prompt enhancer's OpenAI-compatible providers + downstream rewrites. |

### B200 / sm_100a + gcc-15 conda toolchain (flashinfer JIT workaround)

On hosts where the conda toolchain ships gcc-15 (which nvcc rejects with
`#error -- unsupported GNU version! gcc versions later than 14 are not
supported!`), set these env vars before launching anything that triggers
flashinfer's JIT kernel build:

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13 -allow-unsupported-compiler"
```

`dreamverse-server` does NOT set these — they need to come from the launching
shell. The `dreamverse-deploy` skill
([`.agents/skills/dreamverse-deploy/`](../../.agents/skills/dreamverse-deploy/SKILL.md))
sets them for you and is the recommended local-deploy path.
