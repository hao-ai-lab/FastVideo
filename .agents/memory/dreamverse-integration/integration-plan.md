# Integration Plan — Option B+ (Monorepo subfolder + generic backend)

**Status:** Approved plan for execution; do not treat as historical audit.

**Last updated:** 2026-05-05.

**Decision owner:** User decision in the 2026-05-05 planning thread.

**Target branch:** `will/ltx2_sr_port` in
[`/home/william5lin/FastVideo`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/runbook.md#L11-L16).

**Supersedes:**
[`integration-review.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L543-L563)
for the recommendation only. The drift audit, citations, and open-thread
inventory in that document remain reference material.

**Companion docs:**

- Current integration snapshot:
  [`state.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/state.md#L10-L20).
- PR status and #1288 scope:
  [`pr-roadmap.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L38-L42).
- Typed public API design:
  [`design.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L24-L43).
- Cross-repo Dreamverse surfaces:
  [`cross-repo-surfaces.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L13-L20).
- Active follow-ups and drift items:
  [`open-threads.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L15-L45).
- Streaming server route contract:
  [`streaming-server.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L232-L259).
- Memory index and repo paths:
  [`README.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/README.md#L73-L80).
- Execution runbook:
  [`runbook.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/runbook.md#L79-L124).

---

## Decision recap

Option B+ is the selected end state.

Option B+ combines two earlier options:

1. **Option B layout principle:** Dreamverse becomes a product subfolder in
   the FastVideo repository.
2. **Option D backend principle:** reusable backend code remains generic under
   `fastvideo.entrypoints.streaming.*`, not under a Dreamverse namespace.

This differs from the prior Option D recommendation in
[`integration-review.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L543-L563):

- Prior recommendation: keep Dreamverse product in a separate repo and merge
  only reusable backend pieces into FastVideo.
- New decision: move Dreamverse product code into the FastVideo repo under
  `apps/dreamverse/`, while keeping reusable backend code generic.

The repo decision is now:

- One canonical repo: `hao-ai-lab/FastVideo`.
- Dreamverse repo is archived or redirected after migration.
- FastVideo Python library remains at the repository root.
- FastVideo kernel package remains in `fastvideo-kernel/`.
- Generic streaming backend remains at `fastvideo.entrypoints.streaming.*`.
- Dreamverse product server and web app move to `apps/dreamverse/`.

The backend boundary stays aligned with the typed-public-boundary rule: public
inputs normalize into typed objects before legacy internals are touched
([design](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L73-L83)).

The generic backend work already in flight through PRs #1257, #1258, #1284,
#1286, and #1288 stays on that path. #1288 is the current consolidated vehicle
for the remaining `will/ltx2_sr_port` chain
([roadmap](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L38-L42)).

---

## Target architecture

### Repo layout (after migration)

```text
FastVideo/
├── fastvideo/                         # Python ML library, public API, entrypoints
│   └── entrypoints/{openai,streaming}/ # existing OpenAI + generic streaming backend
├── fastvideo-kernel/                  # separate CUDA/custom-kernel package
├── examples/serving/streaming_demo.yaml
├── docs/contributing/dreamverse-development.md
├── apps/
│   └── dreamverse/
│       ├── README.md, AGENTS.md, arch.md, design.md, gpu-pool.{svg,drawio}
│       ├── server/
│       │   ├── pyproject.toml, main.py, config.py, video_generation.py
│       │   ├── session/{controller.py,messages.py}
│       │   ├── routes/presets.py
│       │   ├── prompts/
│       │   └── tests/
│       ├── web/                       # standalone Next.js frontend
│       ├── prompts/                   # curated product content if separated
│       ├── serve_configs/             # product deployment YAMLs
│       └── scripts/                   # setup and launch scripts
├── .github/workflows/
│   ├── ci-dreamverse-backend.yml
│   └── ci-dreamverse-frontend.yml
├── pyproject.toml, uv.lock
└── .pre-commit-config.yaml
```

FastVideo already has a Python-package layout with `fastvideo/`,
`fastvideo-kernel/`, `examples/`, `docs/`, and `tests/`
([codebase map](file:///home/william5lin/FastVideo/.agents/memory/codebase-map/README.md#L5-L75)).
Option B+ adds `apps/dreamverse/` without moving the library or kernel paths.

### Module boundary

| Area | Owns | Does not own | Evidence / reference |
|---|---|---|---|
| `fastvideo.api` | `GeneratorConfig`, `GenerationRequest`, `ServeConfig`, `VideoResult`, `VideoEvent`, `ContinuationState` | Product route policy or curated content | Typed schema surface is documented in [design](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L45-L72). |
| `fastvideo.entrypoints.streaming.*` | Generic WebSocket server, GPU pool, streaming protocol, fMP4 stream, mock server, prompt enhancer, safety/rewrite helpers, router | Dreamverse-only curated presets, prompt UX routes, frontend devtools | Streaming target package is documented in [streaming-server](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L164-L195). |
| `apps/dreamverse/server.*` | Product FastAPI composition, env defaults, curated content, Dreamverse WebSocket/session flow, product worker wrapper | Generic reusable FastVideo backend code | Product-only routes stay Dreamverse-side per route contract [streaming-server](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L239-L255). |
| `apps/dreamverse/web` | Next.js product UI and browser tests | Python package release or FastVideo core CI | New standalone frontend CI is introduced in this plan. |
| `apps/dreamverse/prompts` | Curated product prompt content and presets | Generic prompt provider protocol | DR-1 keeps product prompt orchestration Dreamverse-side unless a second consumer appears [open-threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L440-L455). |
| `examples/serving/` | Public-safe serving examples | Product deployment secrets or private paths | Missing streaming demo config is tracked as drift item #7 [integration-review](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L259-L276). |

### Import contract

`apps/dreamverse/server/*` may import the following FastVideo surfaces:

```python
# Public root re-exports
from fastvideo import VideoGenerator, PipelineConfig, SamplingParam

# Typed public schema, results, presets
from fastvideo.api import GeneratorConfig, GenerationRequest, ServeConfig, ...

# Reusable streaming runtime
from fastvideo.entrypoints.streaming import build_app, ...
from fastvideo.entrypoints.streaming.<submodule> import ...

# Pipeline + arch configs (typed dataclasses, public surface for advanced wiring)
from fastvideo.configs.pipelines import ...
from fastvideo.configs.models import ...

# High-level entrypoints
from fastvideo.entrypoints.video_generator import VideoGenerator
```

It must NOT import from internal library implementation paths:

```python
from fastvideo.pipelines... import ...      # internal pipeline impl
from fastvideo.models... import ...          # internal model impl
from fastvideo.layers... import ...          # internal layer impl
from fastvideo.worker... import ...          # internal worker IPC
from fastvideo.fastvideo_args import ...     # legacy args; use GeneratorConfig
```

**Why the relaxation from `{api, entrypoints.streaming}` only**: Dreamverse's
[`video_generation.py`](file:///home/william5lin/Dreamverse/server/video_generation.py#L260-L420)
currently imports from `fastvideo.configs`, `fastvideo.entrypoints.video_generator`,
and uses `VideoGenerator.from_pretrained()` — these are public surfaces by design
([design.md](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L295-L332))
even though they live outside `fastvideo.api/`. Restricting to only `api` +
`entrypoints.streaming` would force a 578-LOC rewrite of `video_generation.py`
which is out of scope for the migration. The contract enforces the
**meaningful** boundary: no internal `pipelines/models/layers/worker` paths.

Rationale:

- Dreamverse's three integration surfaces are pipeline construction, realtime
  runtime, and continuation state
  ([cross-repo-surfaces](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L13-L20)).
- The target public surface for reusable streaming runtime is
  `fastvideo.entrypoints.streaming/`
  ([streaming-server](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L55-L60)).
- Dynamo and Dreamverse should consume stable typed APIs, not private modules
  ([cross-repo contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L188-L210)).

Add a contract test in Phase 2:

```python
# apps/dreamverse/server/tests/test_import_contract.py
import ast
from pathlib import Path

ALLOWED_PREFIXES = (
    "fastvideo.api",
    "fastvideo.entrypoints.streaming",
    "fastvideo.entrypoints.video_generator",
    "fastvideo.configs",
)
ALLOWED_EXACT = ("fastvideo",)
FORBIDDEN_PREFIXES = (
    "fastvideo.pipelines",
    "fastvideo.models",
    "fastvideo.layers",
    "fastvideo.worker",
    "fastvideo.fastvideo_args",
)


def test_dreamverse_server_imports_only_public_fastvideo_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    bad: list[tuple[str, int, str]] = []
    for path in root.rglob("*.py"):
        if "/tests/" in path.as_posix():
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = (
                [a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            for name in names:
                if not name:
                    continue
                if name.startswith(FORBIDDEN_PREFIXES):
                    bad.append((str(path.relative_to(root)), node.lineno, name))

    assert bad == [], f"Forbidden internal imports: {bad}"
```

Treat this as the import-boundary gate for all future product-server changes.

---

## Tooling stack

### Python build: uv workspace

Use uv workspaces for the Python side.

Precedent and rationale:

- Official uv workspace docs: https://docs.astral.sh/uv/concepts/workspaces/
- The docs describe workspaces as suitable for a FastAPI web application
  alongside libraries.
- Similar workspace patterns are used by chainlit, rerun-io/rerun,
  microsoft/autogen, and langgenius/dify.
- Hatch workspace remains an alternative, but uv is already present in
  FastVideo's root `pyproject.toml`
  ([pyproject](file:///home/william5lin/FastVideo/pyproject.toml#L89-L115)).

Root `pyproject.toml` currently uses setuptools and discovers packages by
excluding assets, docker, docs, and scripts
([pyproject](file:///home/william5lin/FastVideo/pyproject.toml#L1-L3),
[package find](file:///home/william5lin/FastVideo/pyproject.toml#L158-L162)).

Apply this root diff in Phase 1:

```diff
diff --git a/pyproject.toml b/pyproject.toml
@@
 [tool.uv]
 prerelease = "allow"
 
+[tool.uv.workspace]
+members = ["apps/dreamverse/server"]
+
 [tool.uv.sources]
 torch = [
@@
 [tool.setuptools.packages.find]
-exclude = ["assets*", "docker*", "docs", "scripts*"]
+exclude = ["assets*", "docker*", "docs", "scripts*", "apps*"]
 
 [tool.wheel]
-exclude = ["assets*", "docker*", "docs", "scripts*"]
+exclude = ["assets*", "docker*", "docs", "scripts*", "apps*"]
```

Create `apps/dreamverse/server/pyproject.toml` in Phase 1:

```toml
[project]
name = "dreamverse-server"
version = "0.1.0"
description = "Dreamverse product server"
requires-python = ">=3.10"
dependencies = [
    "fastvideo[streaming]",
    "fastapi==0.129.0",
    "uvicorn==0.41.0",
    "aiofiles",
    "websockets",
]

[project.optional-dependencies]
test = [
    "httpx",
    "pytest",
    "pytest-asyncio",
]

[tool.uv]
package = false

[tool.uv.sources]
fastvideo = { workspace = true }

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
```

Notes:

- `package = false` keeps this as an application workspace member, not a
  package to publish.
- `fastvideo = { workspace = true }` forces local workspace resolution during
  development and CI.
- Product server release is Docker/deploy workflow, not PyPI.

### Frontend build: standalone pnpm

Do not add a root `package.json`.

Keep all frontend tooling under:

```text
apps/dreamverse/web/package.json
apps/dreamverse/web/pnpm-lock.yaml
apps/dreamverse/web/playwright.config.*
```

Rationale:

- FastVideo remains primarily a Python ML library.
- Python contributors should not need Node or pnpm for normal work.
- This intentionally diverges from chainlit's root JS workspace pattern and
  follows the simpler open-webui-style split.

The frontend CI works from `apps/dreamverse/web/` directly.

### Pre-commit

Root `.pre-commit-config.yaml` currently excludes generated or heavyweight
paths, including `.agents`, `examples`, and `fastvideo/models`
([pre-commit](file:///home/william5lin/FastVideo/.pre-commit-config.yaml#L4-L17)).

Extend the global exclude for the Node frontend only.

Do not exclude `apps/dreamverse/server/`; Python hooks must cover product
server code.

Apply this Phase 1 diff:

```diff
diff --git a/.pre-commit-config.yaml b/.pre-commit-config.yaml
@@
     fastvideo/models/.*|
+    apps/dreamverse/web/.*|
     examples/.*|
```

Verification:

```bash
pre-commit run --files \
  pyproject.toml \
  .pre-commit-config.yaml \
  apps/dreamverse/server/pyproject.toml
```

Memory-dir edits are excluded from yapf/ruff/mypy; only the filename-space
check applies
([runbook](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/runbook.md#L81-L93)).

### `.gitignore`

Root `.gitignore` currently contains Python, docs, asset, output, and local
tooling ignores, and notes that Node artifacts are only ignored under `ui/`
([gitignore](file:///home/william5lin/FastVideo/.gitignore#L38-L94)).

Add Dreamverse-scoped Node ignores only, **AND unignore product public assets** (the root currently ignores `*.png`, `*.jpg`, `*.mp4`, `*.gif` per [`.gitignore:6-9`](file:///home/william5lin/FastVideo/.gitignore#L6-L9)):

```diff
diff --git a/.gitignore b/.gitignore
@@
 # Next.js / Node artifacts under ui/: see ui/.gitignore
+
+# Next.js / Node artifacts under apps/dreamverse/web/
+apps/dreamverse/web/node_modules/
+apps/dreamverse/web/.next/
+apps/dreamverse/web/out/
+apps/dreamverse/web/coverage/
+apps/dreamverse/web/test-results/
+apps/dreamverse/web/playwright-report/
+apps/dreamverse/web/.env.local
+apps/dreamverse/web/.env.development.local
+apps/dreamverse/web/.env.test.local
+apps/dreamverse/web/.env.production.local
+
+# Unignore migrated Dreamverse product assets — root .gitignore globally
+# ignores *.png/*.jpg/*.mp4/*.gif, but apps/dreamverse/web/public/ MUST
+# be tracked (logo, icons, k2.png, etc.).
+!apps/dreamverse/web/public/**
+!apps/dreamverse/web/prompts/**
+!apps/dreamverse/server/prompts/**
+!apps/dreamverse/gpu-pool.svg
+!apps/dreamverse/gpu-pool.drawio
```

Do not add global `node_modules/` or `.next/` ignores. Keep the pattern scoped
so unrelated JS projects do not inherit accidental behavior. **Verify before
committing Phase 3** with `git check-ignore -v apps/dreamverse/web/public/k2.png`
(should report nothing once the unignore lands).

### CI workflows

Existing FastVideo CI remains in place:

- Buildkite GPU and test matrix lives at
  [`.buildkite/pipeline.yml`](file:///home/william5lin/FastVideo/.buildkite/pipeline.yml#L199-L280)
  and uses path-based monorepo diff watch lists.
- Full suite is triggered by GitHub Actions and Buildkite
  ([trigger workflow](file:///home/william5lin/FastVideo/.github/workflows/ci-trigger-full-suite.yml#L52-L83),
  [Buildkite full suite](file:///home/william5lin/FastVideo/.buildkite/pipeline.yml#L282-L438)).
- Pre-commit workflow currently runs on pull requests
  ([ci-precommit](file:///home/william5lin/FastVideo/.github/workflows/ci-precommit.yml#L1-L32)).
- Docs workflow already uses explicit docs paths
  ([infra-docs](file:///home/william5lin/FastVideo/.github/workflows/infra-docs.yml#L3-L18)).
- FastVideo PyPI publish remains version-driven on root `pyproject.toml`
  ([publish-fastvideo](file:///home/william5lin/FastVideo/.github/workflows/publish-fastvideo.yml#L1-L72)).

Add two new workflows.

#### `.github/workflows/ci-dreamverse-frontend.yml`

```yaml
name: Dreamverse Frontend

on:
  pull_request:
    branches: [main]
    paths:
      - 'apps/dreamverse/web/**'
      - '.github/workflows/ci-dreamverse-frontend.yml'
  push:
    branches: [main]
    paths:
      - 'apps/dreamverse/web/**'
      - '.github/workflows/ci-dreamverse-frontend.yml'

permissions:
  contents: read

defaults:
  run:
    working-directory: apps/dreamverse/web

jobs:
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # IMPORTANT: pnpm/action-setup MUST run BEFORE setup-node when using
      # cache: pnpm — setup-node otherwise can't find pnpm to populate cache.
      - name: Setup pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 9

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: pnpm
          cache-dependency-path: apps/dreamverse/web/pnpm-lock.yaml

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Typecheck
        run: pnpm run typecheck --if-present

      - name: Unit tests
        run: pnpm run test --if-present

      - name: Build
        run: pnpm run build

      # NOTE: Playwright tests require a running backend. Until Phase 4 lands
      # `/healthz`/`/readyz`/`/status`/`/prompt-system-config`/`/curated-presets`
      # in `apps/dreamverse/server/main.py`, scope frontend CI to build + unit
      # tests only. Re-enable Playwright in Phase 4 once health/preset routes
      # are mounted by `apps/dreamverse/server/main.py` (which can wrap public
      # `fastvideo.entrypoints.streaming.build_app` and add product routes).
      #
      # - name: Install Playwright browsers
      #   run: pnpm exec playwright install --with-deps chromium
      #
      # - name: Playwright (re-enable in Phase 4)
      #   run: pnpm exec playwright test
```

**Playwright config update needed** when moving FE: `Dreamverse/apps/web/playwright.config.ts` line 39 currently uses `npm run dev`; change to `pnpm run dev` post-move ([source](file:///home/william5lin/Dreamverse/apps/web/playwright.config.ts#L39)).

#### `.github/workflows/ci-dreamverse-backend.yml`

```yaml
name: Dreamverse Backend

on:
  pull_request:
    branches: [main]
    paths:
      - 'apps/dreamverse/server/**'
      - 'pyproject.toml'
      - 'uv.lock'
      # apps/dreamverse/server depends on these public surfaces — changes
      # there must trigger Dreamverse backend tests:
      - 'fastvideo/api/**'
      - 'fastvideo/entrypoints/streaming/**'
      - 'fastvideo/entrypoints/video_generator.py'
      - 'fastvideo/configs/**'
      - '.github/workflows/ci-dreamverse-backend.yml'
  push:
    branches: [main]
    paths:
      - 'apps/dreamverse/server/**'
      - 'pyproject.toml'
      - 'uv.lock'
      - 'fastvideo/api/**'
      - 'fastvideo/entrypoints/streaming/**'
      - 'fastvideo/entrypoints/video_generator.py'
      - 'fastvideo/configs/**'
      - '.github/workflows/ci-dreamverse-backend.yml'

permissions:
  contents: read

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Sync Dreamverse backend workspace (locked)
        run: uv sync --locked --package dreamverse-server --extra test

      - name: Backend tests
        run: uv run --locked --package dreamverse-server --extra test pytest apps/dreamverse/server/tests/ -q
```

Extend existing workflows where appropriate.

Recommended diffs:

```diff
diff --git a/.github/workflows/ci-precommit.yml b/.github/workflows/ci-precommit.yml
@@
   pull_request:
     branches: [main]
+    paths-ignore:
+      - 'apps/dreamverse/web/**'
```

```diff
diff --git a/.github/workflows/ci-trigger-full-suite.yml b/.github/workflows/ci-trigger-full-suite.yml
@@
   pull_request_target:
     types: [labeled, synchronize]
+    paths-ignore:
+      - 'apps/dreamverse/**'
```

Do not add `paths-ignore` to workflows already constrained by non-Dreamverse
`paths`, such as docs deploy or PyPI publish, unless a later CI audit finds
actual false triggers.

### Release

FastVideo PyPI release remains unchanged:

- Root package is still `fastvideo`.
- Version-driven publish remains tied to root `pyproject.toml` version changes
  ([publish workflow](file:///home/william5lin/FastVideo/.github/workflows/publish-fastvideo.yml#L23-L72)).
- `apps*` is excluded from setuptools and wheel discovery.

Dreamverse product release is separate:

- Backend: Docker or deployment workflow rooted at `apps/dreamverse/server/`.
- Frontend: Vercel or frontend deploy workflow rooted at `apps/dreamverse/web/`.
- No PyPI publish for `apps/dreamverse/server`.
- No root `package.json`.

Add deployment workflows only after Phase 6, once the product runs from the
monorepo CI path.

### OSS precedents used for this plan

| Project | URL | Pattern applied here |
|---|---|---|
| uv workspaces | https://docs.astral.sh/uv/concepts/workspaces/ | Authoritative Python workspace model. |
| Hatch monorepo | https://hatch.pypa.io/latest/how-to/environment/workspace/ | Alternative workspace model; not selected. |
| chainlit | https://github.com/Chainlit/chainlit | uv workspace precedent plus **per-language CI split** (separate `check-frontend.yaml` / `check-backend.yaml` workflows path-filtered by directory). Not a PR-level split — independent of D-17 single-mega-PR decision. |
| open-webui | https://github.com/open-webui/open-webui | Frontend path filtering (`paths-ignore` on backend-only changes) and separate release tracks. **Note:** open-webui has a root `package.json`; we are choosing standalone-pnpm despite the precedent, to avoid forcing Python-only contributors to install Node. |
| streamlit | https://github.com/streamlit/streamlit | Split Python and JS testing in one repo. |
| gradio | https://github.com/gradio-app/gradio | Python package plus JS workspace precedent. |
| full-stack-fastapi-template-nextjs | https://github.com/nemanjam/full-stack-fastapi-template-nextjs | Separate frontend/backend build and deploy workflows. |
| rerun-io/rerun, microsoft/autogen, langgenius/dify | various | uv workspace examples with workspace members and sources. |

### Test strategy

The migration spans CPU-only logic, GPU-required inference, and end-to-end
WebSocket flows. Each phase's verification gate must specify which test
class runs where, because not all tests can run on `ubuntu-latest` CI.

#### Test taxonomy

| Class | Marker | Runs in | What it covers | Examples |
|---|---|---|---|---|
| **Unit** | (none / `unit`) | `ci-dreamverse-backend.yml` (ubuntu-latest CI) + locally | Pure logic, no GPU, no live service. Mocked FastVideo backends, schema validation, helper functions. | `test_config.py`, `test_rewrite_prompt_payload.py`, `test_session_init_image.py`, the new `test_import_contract.py` |
| **Integration (fakes)** | `integration` | `ci-dreamverse-backend.yml` + locally | FastAPI test client + in-process fakes/mocks for GPU pool. Validates routes, request/response shapes, session state machine. | `test_health_endpoints.py`, `test_mock_server.py`, `test_entrypoints.py`, `test_prompt_safety.py`, `test_batching.py` (deleted) |
| **Live-service GPU** | `gpu` (skip-by-default in CI) | **Local GPU4 manual QA** + Buildkite-Modal (when added) | Real `fastvideo serve` process + real model weights + real WebSocket round-trips. Validates LTX-2 streaming, NVFP4 wiring, continuation state, frame emission. | `test_realtime_stress.py` (947 LOC), `test_session_logging.py` (1278 LOC) — these spin up real workers per their current shape |
| **Frontend unit / build** | (n/a — pnpm) | `ci-dreamverse-frontend.yml` (ubuntu-latest, no GPU) | Vitest + tsc + Next.js build. No backend needed. | `apps/dreamverse/web/src/**/*.test.ts(x)` |
| **Frontend Playwright E2E** | (n/a — pnpm) | **Local GPU4 manual QA** until Phase 4 lands public health routes; then `ci-dreamverse-frontend.yml` against a mock backend OR a deployed staging | Real browser → real backend WebSocket flow. Requires `/healthz`, `/readyz`, `/status`, `/prompt-system-config`, `/curated-presets`, `/v1/stream`. | `apps/dreamverse/web/e2e/{backend-health,frontend-shell,preset-prompt-generation}.spec.ts` |
| **FastVideo public contract** | (none) | Existing FastVideo CI (`ci-precommit` + Buildkite for GPU) | Schema/shape guards that this migration must not break. | `fastvideo/tests/contract/test_dreamverse_shape.py`, `test_dynamo_shape.py`, `test_generate_async.py` |
| **FastVideo SSIM regression** | (Buildkite path-filter) | Buildkite-Modal | Inference-quality gates for ported models. | `fastvideo/tests/ssim/test_*.py` |

#### Adding the `gpu` marker

In Phase 1, add to root `pyproject.toml`:

```diff
 [tool.pytest.ini_options]
+markers = [
+  "gpu: requires a real GPU + model weights; skip in ubuntu-latest CI",
+]
```

In `apps/dreamverse/server/pyproject.toml` `[tool.pytest.ini_options]`, set
the default for the backend test command to skip GPU tests:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not gpu'"
markers = ["gpu: requires real GPU"]
```

GPU-dependent tests must add `@pytest.mark.gpu` at module or function level
during the Phase 2 move. Specifically: `test_realtime_stress.py` and
`test_session_logging.py` per their current LOC and live-service shape.

#### Local GPU4 verification hook

This dev node has 8× B200 GPUs. **GPU4** is the operator's chosen target for
migration smoke tests. The original `dreamverse-server` was launched with
`CUDA_VISIBLE_DEVICES=4`, which makes physical GPU 4 appear as logical GPU 0
inside the process — so the Dreamverse-side log shows `Selected GPU ids: [0]`
even though it physically holds GPU 4. **Always launch with `CUDA_VISIBLE_DEVICES=4`**
to keep the same physical pin during smoke runs.

##### Phase 0 prerequisites for the GPU4 hook (validated 2026-05-05)

A live smoke test of `fastvideo serve` from `will/ltx2_sr_port` HEAD
`d23e71c2` ran on GPU4 and revealed two missing optional deps in the
FastVideo `.venv`:

- **`flashinfer-python`** — required for the NVFP4 path. Without it, the
  server fails at boot with `ImportError: NVFP4 quantization requires
  flashinfer. Install with 'pip install flashinfer-python'.`
- **`flash_attn`** — optional. Without it, attention falls back to Torch
  SDPA (`Cannot use FlashAttention-2 backend ... Using Torch SDPA backend.`).
  Functional but slower per inference step.

For a fast smoke that doesn't need NVFP4, comment out or omit
`engine.quantization` in the serve config — the loader auto-falls-back
to bf16. The serve config in
[`Dreamverse/serve_configs/streaming_demo.yaml`](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L26-L29)
explicitly says: "Hosts without flashinfer / NVFP4: comment out the
`engine.quantization` block below — the loader falls back to bf16
automatically when no quant_config is set."

For production-equivalent NVFP4 smoke, install both deps in the FastVideo
`.venv`:

```bash
.venv/bin/pip install flashinfer-python flash-attn --no-build-isolation
```

This is a **Phase 0 prerequisite** to document in
`docs/contributing/dreamverse-development.md` (Phase 1).

##### Reclaim and redeploy commands

```bash
# Stop the running Dreamverse-side server holding GPU4
sudo kill 2453227   # or use the supervisor that owns it

# Confirm GPU4 is free
nvidia-smi --query-gpu=index,memory.used --format=csv | grep -E '^4,'

# Deploy the new public FastVideo streaming server pinned to GPU4
CUDA_VISIBLE_DEVICES=4 uv run --locked --package dreamverse-server \
  fastvideo serve --config apps/dreamverse/serve_configs/streaming_demo.yaml \
  --host 0.0.0.0 --port 8009

# Smoke-test from another terminal
curl -s http://localhost:8009/health | jq .
curl -s http://localhost:8009/readyz | jq .   # Phase 4+ only
# Drive the FE against it: cd apps/dreamverse/web && pnpm run dev
```

This is the **manual QA gate** for any phase that touches the live-service
path (Phase 2 backend move, Phase 3 frontend move, Phase 4 health-route
promotion, Phase 5 prompt enhancer retirement). The verification gate for
each phase calls out whether the GPU4 smoke is required or optional.

**Cleanup after each test session:**

```bash
# Kill the test deployment
pkill -f 'fastvideo serve --config'

# Restart the canonical Dreamverse-side server with the same GPU pin
cd /home/william5lin/Dreamverse
setsid bash -c 'CUDA_VISIBLE_DEVICES=4 exec .venv/bin/dreamverse-server > /tmp/dv.log 2>&1 < /dev/null' & disown

# Confirm restart
sleep 3 && pgrep -af dreamverse-server | head -3
```

**Verified 2026-05-05**: this kill/redeploy/teardown cycle was executed
end-to-end on `will/ltx2_sr_port` HEAD `d23e71c2`. Public `fastvideo serve`
booted in ~24 s on GPU4 with bf16 fallback (no flashinfer in `.venv`),
served `/health` → 200 `{"status":"ok","sessions":0,"stream_mode":"av_fmp4"}`,
and confirmed empirically that `/healthz`, `/readyz`, `/status`,
`/prompt-system-config`, and `/curated-presets` all 404 (matching drift
table item #4 → Phase 4 promotion target). Smoke artifacts at
`/tmp/opencode/fv_smoke/` (serve.log, smoke.yaml).

#### Per-phase test responsibilities (summary)

| Phase | Unit | Integration (fakes) | GPU live (GPU4) | Frontend build | Frontend E2E (Playwright) |
|---|:---:|:---:|:---:|:---:|:---:|
| 0 (#1288 land) | Existing FastVideo suite | Existing | Optional sanity | n/a | n/a |
| 1 (skeleton) | New empty pkg `pytest --collect-only` | n/a | n/a | n/a | n/a |
| 2 (backend move) | **Required** in CI | **Required** in CI | **Required manual** on GPU4 (smoke) | n/a | n/a |
| 3 (FE move) | Backend still green | Backend still green | Optional | **Required** in CI | **Manual on GPU4 only** (FE CI Playwright deferred to Phase 4) |
| 4 (promote pending) | Streaming tests must add coverage | Required | **Required manual** on GPU4 | Required | **Required manual** on GPU4 → re-enable FE-CI Playwright at end |
| 5 (DR-1 / DR-2) | Required | Prompt-shim tests required | **Required manual** prompt-flow on GPU4 | Required | **Required manual** preset-prompt-generation spec on GPU4 |
| 6a-6f | All CI green | All CI green | Optional during 6b/6c deploy dry runs | All CI green | Required against staging in 6c |
| 7 (archive) | n/a | n/a | n/a | n/a | n/a |

---

## File-by-file migration map

### PRODUCT files (move into `apps/dreamverse/`)

| Source path | LOC | Target path | Notes |
|---|---:|---|---|
| `Dreamverse/server/main.py` | 148 | `apps/dreamverse/server/main.py` | FastAPI app composition; imports change to `from fastvideo.entrypoints.streaming import build_app` etc. |
| `Dreamverse/server/config.py` | 303 | `apps/dreamverse/server/config.py` | Dreamverse-specific env/model registry/prompt paths. |
| `Dreamverse/server/video_generation.py` | 578 | `apps/dreamverse/server/video_generation.py` | LTX-2 product-specific worker wrapper + continuation conditioning. |
| `Dreamverse/server/session/controller.py` | 1948 | `apps/dreamverse/server/session/controller.py` | Per-WebSocket session controller — Dreamverse protocol/UI flow. |
| `Dreamverse/server/session/messages.py` | 21 | `apps/dreamverse/server/session/messages.py` | Queue DTOs; promote only if generic. |
| `Dreamverse/server/routes/presets.py` | 234 | `apps/dreamverse/server/routes/presets.py` | Curated preset HTTP routes, product content. |
| `Dreamverse/server/tests/test_config.py` | 154 | `apps/dreamverse/server/tests/test_config.py` | Product config tests. |
| `Dreamverse/server/tests/test_entrypoints.py` | 99 | `apps/dreamverse/server/tests/test_entrypoints.py` | Product CLI tests. |
| `Dreamverse/server/tests/test_realtime_stress.py` | 947 | `apps/dreamverse/server/tests/test_realtime_stress.py` | E2E websocket stress, product-specific. |
| `Dreamverse/server/tests/test_session_logging.py` | 1278 | `apps/dreamverse/server/tests/test_session_logging.py` | Product session flow tests. |
| `Dreamverse/server/tests/conftest.py` | 14 | `apps/dreamverse/server/tests/conftest.py` | Test path bootstrap. |
| `Dreamverse/server/prompts/*.md` | — | `apps/dreamverse/server/prompts/*.md` | System prompt content: 4 files plus README. |
| `Dreamverse/apps/web/**` | — | `apps/dreamverse/web/**` | Entire Next.js frontend. |
| `Dreamverse/AGENTS.md`, `arch.md`, `design.md`, `gpu-pool.{svg,drawio}` | — | `apps/dreamverse/` | Product-level docs and diagrams. |
| `Dreamverse/scripts/install_native_ffmpeg.sh`, `smoke_local.sh` | — | `apps/dreamverse/scripts/` | Product launch/setup scripts. |
| `Dreamverse/serve_configs/streaming_demo.yaml` | — | `apps/dreamverse/serve_configs/streaming_demo.yaml` | Product deployment config; also copy to `examples/serving/streaming_demo.yaml`. |
| `Dreamverse/.agents/skills/launch-demo/scripts/*.sh` | — | `apps/dreamverse/scripts/launch/*.sh` | Demo launch helpers. |

### GENERIC-MERGED files (delete from Dreamverse copy; import public FastVideo)

| Source path | Already on FastVideo public at | Action |
|---|---|---|
| `Dreamverse/server/gpu_pool.py` (1053 LOC) | `fastvideo.entrypoints.streaming.gpu_pool` | Delete; `apps/dreamverse/server/main.py` imports public API. |
| `Dreamverse/server/prompt_enhancer.py` (1933 LOC) | `fastvideo.entrypoints.streaming.prompt.*` (partial) | **DEFER deletion to Phase 5.** Public `PromptEnhancer` does not yet expose Dreamverse-side methods (`resolve_rewrite_model` per [`controller.py:203`](file:///home/william5lin/Dreamverse/server/session/controller.py#L203), `get_prompt_config` / `save_prompt_config` per [`routes/presets.py:95`](file:///home/william5lin/Dreamverse/server/routes/presets.py#L95)). Carry the file into `apps/dreamverse/server/prompt_enhancer.py` during Phase 2 unmodified. Phase 5 implements DR-1 compat shim + retires the fork. |
| `Dreamverse/server/av_streaming.py` (434 LOC) | `fastvideo.entrypoints.streaming.stream` | Delete. |
| `Dreamverse/server/worker_ipc.py` (161 LOC) | `fastvideo.entrypoints.streaming.protocol` | Delete. |
| `Dreamverse/server/mock_server.py` (1207 LOC) | `fastvideo.entrypoints.streaming.mock_server` | Delete. |
| `Dreamverse/server/session_init_image.py` (103 LOC) | `fastvideo.entrypoints.streaming.session_init_image` | Delete. |
| `Dreamverse/server/session_logger.py` (44 LOC) | `fastvideo.entrypoints.streaming.session_logger` | Delete. |
| `Dreamverse/server/tests/test_gpu_pool.py` (126 LOC) | covered by FastVideo tests | Delete. |
| `Dreamverse/server/tests/test_mock_server.py` (360 LOC) | covered by FastVideo tests | Delete. |
| `Dreamverse/server/tests/test_prompt_enhancer.py` (1176 LOC) | covered by FastVideo tests | Delete, or keep product-specific subset only. |
| `Dreamverse/server/tests/test_session_init_image.py` (74 LOC) | covered by FastVideo tests | Delete. |

Generic-merged ownership is consistent with the existing upstreaming map in
[`streaming-server.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L14-L31).

### GENERIC-PENDING files (schedule for promotion)

| Source path | LOC | Target path on FastVideo public |
|---|---:|---|
| `Dreamverse/server/runtime.py` | 21 | `fastvideo.entrypoints.streaming.runtime` or absorb into server state. |
| `Dreamverse/server/prompt_safety.py` | 222 | `fastvideo.entrypoints.streaming.prompt.safety` by extending existing safety module. |
| `Dreamverse/server/rewrite_prompt_payload.py` | 98 | `fastvideo.entrypoints.streaming.prompt.rewrite` by extending existing rewrite module. |
| `Dreamverse/server/utils.py` | 21 | absorb into appropriate streaming/API utility. |
| `Dreamverse/server/routes/health.py` | 101 | extend `fastvideo.entrypoints.streaming.server.build_app` to expose `/healthz`, `/readyz`, `/status`. |
| `Dreamverse/server/benchmarks/benchmark_*.py` | ~900 total | `fastvideo/tests/benchmarks/` or `benchmarks/` at FastVideo root. |
| `Dreamverse/server/tests/test_health_endpoints.py` | 163 | follow health route merge. |
| `Dreamverse/server/tests/test_prompt_safety.py` | 51 | follow prompt safety merge. |
| `Dreamverse/server/tests/test_rewrite_prompt_payload.py` | 89 | follow rewrite merge. |
| `Dreamverse/server/tests/test_benchmark_*.py` | ~470 total | follow benchmark merge. |

The health route gap is tracked as open item #1
([open-threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99)).

### DELETE files (no longer needed)

| Source path | LOC | Why |
|---|---:|---|
| `Dreamverse/server/server_entry.py` | 15 | CLI shim no longer needed in monorepo. |
| `Dreamverse/server/routes/__init__.py` | 1 | Empty package marker. |
| `Dreamverse/server/session/__init__.py` | 5 | Empty package marker. |
| `Dreamverse/server/tests/test_batching.py` | 149 | Obsolete ORCA batching protocol. |

---

## Phased migration plan

### Phase 0 — Land #1288 (in flight)

**Purpose:** Ensure the generic FastVideo backend substrate is available before
moving Dreamverse product code into the monorepo.

**Files / scope:**

- `fastvideo/api/` typed results and event classes.
- `fastvideo/entrypoints/video_generator.py` async generation path.
- `fastvideo/entrypoints/streaming/` generic streaming server, GPU pool,
  prompt, safety, rewrite, mock server, router, and tests.
- LTX-2 SR runtime and NVFP4 wiring included in #1288.
- Contract tests for Dreamverse and Dynamo shapes.

**Source:** #1288 is open and mergeable on `will/ltx2_sr_port`
([roadmap](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L38-L42)).

**Steps:**

1. Wait for #1288 to merge to `origin/main`.
2. Fetch main and verify the content is present.
3. Update memory-dir state after merge per runbook
   ([runbook](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/runbook.md#L51-L70)).
4. Start Phase 1 from updated `main` or a new branch based on it.

**Verification gate:**

- #1288 merge commit exists on `origin/main`.
- `fastvideo.api.VideoEvent` and `VideoGenerator.generate_async` are importable.
- Existing API/contract/NVFP4/LTX-2 smoke baseline remains green; May 2 baseline
  is documented in
  [`state.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/state.md#L142-L149).

**Rollback:**

- If #1288 does not merge, stop. Do not move Dreamverse product code.
- If #1288 merges and regresses, revert or fix #1288 before Phase 1.
- Do not create `apps/dreamverse/` until generic backend import targets are
  stable on the branch used for migration.

### Phase 1 — Add `apps/` skeleton

**Purpose:** Establish monorepo tooling and empty product directories before a
mass move. This should be a small PR.

**Files to add:**

- `apps/dreamverse/README.md`
- `apps/dreamverse/server/pyproject.toml`
- `apps/dreamverse/server/tests/.gitkeep`
- `apps/dreamverse/web/.gitkeep`
- `apps/dreamverse/prompts/.gitkeep`
- `apps/dreamverse/serve_configs/.gitkeep`
- `apps/dreamverse/scripts/.gitkeep`
- `docs/contributing/dreamverse-development.md`

**Files to modify:**

- `pyproject.toml`
- `uv.lock` — **regenerated** when adding the workspace member; commit alongside.
- `.pre-commit-config.yaml`
- `.gitignore`
- `mkdocs.yml`, if `docs/contributing/dreamverse-development.md` is added to
  the Developer Guide nav. Current nav lives at
  [`mkdocs.yml`](file:///home/william5lin/FastVideo/mkdocs.yml#L169-L180).
- `fastvideo/entrypoints/streaming/gpu_pool.py` for D-12-A docstring caveat.

**Steps:**

1. Create the directory skeleton.
2. Add the server `pyproject.toml` shown in the tooling section.
3. Add `[tool.uv.workspace] members = ["apps/dreamverse/server"]` to root
   `pyproject.toml`.
4. Add `apps*` to root setuptools and wheel excludes.
5. Add scoped Dreamverse web ignores + product-asset unignores to `.gitignore`.
6. Add `apps/dreamverse/web/.*` to pre-commit global exclude.
7. Add `docs/contributing/dreamverse-development.md` with local dev commands:
   backend `uv run --locked --package dreamverse-server --extra test pytest ...`;
   frontend `cd apps/dreamverse/web && pnpm install && pnpm run build`.
8. Run `uv lock` to regenerate `uv.lock` with the new workspace member; commit
   the lock change in the same PR.
9. Land D-12-A docstring caveat: mark `GpuPool` experimental/server-internal
   until `run_async()` lands. This is tracked in
   [`open-threads.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L294-L308).

**Verification gate:**

- `uv sync --locked --package dreamverse-server --extra test` succeeds (proves
  `uv.lock` is in sync with the workspace declaration).
- `pre-commit run --files pyproject.toml uv.lock .pre-commit-config.yaml .gitignore apps/dreamverse/server/pyproject.toml docs/contributing/dreamverse-development.md` succeeds.
- `uv run --with build python -m build --sdist --wheel .` still excludes
  `apps*` from the FastVideo wheel/sdist (verify with `unzip -l dist/*.whl | grep apps` returns nothing).
- `lsp_diagnostics` is clean for changed Python files.
- No root `package.json` exists.
- `git check-ignore -v apps/dreamverse/web/public/k2.png` reports nothing
  (asset unignore works) — only verifiable in Phase 3 once the file exists,
  but the rule lands in Phase 1.

**Rollback:**

- Revert the Phase 1 PR.
- No product files have moved yet, so rollback does not affect Dreamverse repo.
- If only uv workspace sync fails, revert the workspace stanza and keep the
  empty directory PR blocked until the pyproject shape is fixed.

### Phase 2 — Move PRODUCT files (server)

**Purpose:** Move the Dreamverse product backend into the monorepo while
rewiring imports to public FastVideo surfaces.

**Files to move:**

- `Dreamverse/server/main.py` → `apps/dreamverse/server/main.py`
- `Dreamverse/server/config.py` → `apps/dreamverse/server/config.py`
- `Dreamverse/server/video_generation.py` → `apps/dreamverse/server/video_generation.py`
- `Dreamverse/server/session/controller.py` → `apps/dreamverse/server/session/controller.py`
- `Dreamverse/server/session/messages.py` → `apps/dreamverse/server/session/messages.py`
- `Dreamverse/server/routes/presets.py` → `apps/dreamverse/server/routes/presets.py`
- `Dreamverse/server/tests/test_config.py` → `apps/dreamverse/server/tests/test_config.py`
- `Dreamverse/server/tests/test_entrypoints.py` → `apps/dreamverse/server/tests/test_entrypoints.py`
- `Dreamverse/server/tests/test_realtime_stress.py` → `apps/dreamverse/server/tests/test_realtime_stress.py`
- `Dreamverse/server/tests/test_session_logging.py` → `apps/dreamverse/server/tests/test_session_logging.py`
- `Dreamverse/server/tests/conftest.py` → `apps/dreamverse/server/tests/conftest.py`

**Files NOT moved as PRODUCT (deleted in Phase 2 because public substitute exists):**

- `Dreamverse/server/gpu_pool.py` — replaced by `fastvideo.entrypoints.streaming.gpu_pool`
- `Dreamverse/server/av_streaming.py` — replaced by `fastvideo.entrypoints.streaming.stream`
- `Dreamverse/server/worker_ipc.py` — replaced by `fastvideo.entrypoints.streaming.protocol`
- `Dreamverse/server/mock_server.py` — replaced by `fastvideo.entrypoints.streaming.mock_server`
- `Dreamverse/server/session_init_image.py` — replaced by `fastvideo.entrypoints.streaming.session_init_image`
- `Dreamverse/server/session_logger.py` — replaced by `fastvideo.entrypoints.streaming.session_logger`

**Files MOVED AS-IS into `apps/dreamverse/server/` despite being technically generic — these are GENERIC-PENDING that haven't landed publicly yet, and Dreamverse won't boot without them:**

| Source path | Target | Reason carried into Phase 2 |
|---|---|---|
| `Dreamverse/server/runtime.py` (21) | `apps/dreamverse/server/runtime.py` | [`main.py:19`](file:///home/william5lin/Dreamverse/server/main.py#L19) and `controller.py` import `runtime` for the runtime singletons; promote to public in Phase 4. |
| `Dreamverse/server/utils.py` (21) | `apps/dreamverse/server/utils.py` | Used by `controller.py` for timestamps + segment-cap helpers; absorb into public utility in Phase 4. |
| `Dreamverse/server/prompt_safety.py` (222) | `apps/dreamverse/server/prompt_safety.py` | Imported by `main.py` startup; promote to `fastvideo.entrypoints.streaming.prompt.safety` in Phase 4. |
| `Dreamverse/server/rewrite_prompt_payload.py` (98) | `apps/dreamverse/server/rewrite_prompt_payload.py` | Imported by `routes/presets.py` and `controller.py`; promote to `fastvideo.entrypoints.streaming.prompt.rewrite` in Phase 4. |
| `Dreamverse/server/routes/health.py` (101) | `apps/dreamverse/server/routes/health.py` | Mounted at startup by `main.py`; needed for FE Playwright tests. Promote to `fastvideo.entrypoints.streaming.server.build_app` in Phase 4. |
| `Dreamverse/server/prompt_enhancer.py` (1933) | `apps/dreamverse/server/prompt_enhancer.py` | DR-1 fork; replaced in Phase 5 with public + thin compat shim. |

These are **explicit shims** — Phase 4 is responsible for their promotion to public surfaces and corresponding deletion from `apps/dreamverse/server/`.

**Required import rewrites in moved PRODUCT files:**

- `from .gpu_pool import GpuPool` → `from fastvideo.entrypoints.streaming.gpu_pool import GpuPool`
- `from .av_streaming import ...` → `from fastvideo.entrypoints.streaming.stream import ...`
- `from .worker_ipc import ...` → `from fastvideo.entrypoints.streaming.protocol import ...`
- `from .mock_server import ...` → `from fastvideo.entrypoints.streaming.mock_server import ...`
- `from .session_init_image import ...` → `from fastvideo.entrypoints.streaming.session_init_image import ...`
- `from .session_logger import ...` → `from fastvideo.entrypoints.streaming.session_logger import ...`
- Typed request/result/state imports → `fastvideo.api`
- `runtime`, `utils`, `prompt_safety`, `rewrite_prompt_payload`, `routes.health`, `prompt_enhancer` — kept relative (`from .runtime import ...`) inside `apps/dreamverse/server/` until Phase 4/5 promotes them.

**Required path rewrites in moved files:**

- [`config.py:13`](file:///home/william5lin/Dreamverse/server/config.py#L13): `_APP_ROOT / "apps" / "web"` → `_APP_ROOT / "web"` (since `_APP_ROOT` will resolve to `apps/dreamverse/` in the new layout).
- [`apps/web/next.config.ts:11`](file:///home/william5lin/Dreamverse/apps/web/next.config.ts#L11): `outputFileTracingRoot: path.resolve(__dirname, "../..")` → `path.resolve(__dirname, "../../..")` (one extra `..` since the FE is one level deeper in the monorepo).
- [`playwright.config.ts:39`](file:///home/william5lin/Dreamverse/apps/web/playwright.config.ts#L39): `command: "npm run dev"` → `command: "pnpm run dev"` (matches Phase 1 tooling decision).
- Any hardcoded `../FastVideo` paths in scripts/configs → make repo-root-relative since they now share a repo.

**Steps:**

1. **Cross-repo move with explicit history disposition.** `git mv` does NOT preserve history across repositories. Choose one:
   - **(Recommended) Fresh import**: `git mv` from `Dreamverse/server/...` → `apps/dreamverse/server/...` within the new monorepo as a single commit; original Dreamverse repo retains full history; mention the move source SHA in the commit body.
   - **(Alternative) Subtree merge**: use `git subtree add --prefix=apps/dreamverse <dreamverse-remote> <ref>` to import history. Heavier; usually unnecessary if Dreamverse repo is being archived (history stays accessible there).
   - The decision **must be documented in the Phase 2 PR body**, including the SHA of the originating Dreamverse commit being imported.
2. Rewire imports to public streaming and API modules per the rewrite list above.
3. Apply path rewrites listed above.
4. Add the import contract test from the Target architecture section.
5. Keep `routes/presets.py` product-local.
6. Keep `session/controller.py` product-local.
7. Verify D-8 `ltx2_image_crf` flow after migration:
   - Source item is tracked in
     [`open-threads.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L51-L68).
   - The known concern is that `ltx2_image_crf` may still be silently dropped
     unless it reaches `request.stage_overrides.refine.image_crf`.
   - Add a Dreamverse-shape contract test pinning the field once the trace is verified.

**Verification gate:**

- **CI / unit / integration**:
  `uv run --locked --package dreamverse-server --extra test pytest apps/dreamverse/server/tests/ -m 'not gpu' -q`
  succeeds (skips GPU-marked tests; runs in `ci-dreamverse-backend.yml`).
- Import contract test (`test_import_contract.py`) passes.
- `lsp_diagnostics` is clean for changed Python files.
- `pre-commit run --files apps/dreamverse/server/*.py apps/dreamverse/server/session/controller.py apps/dreamverse/server/session/messages.py apps/dreamverse/server/routes/presets.py apps/dreamverse/server/tests/*.py` succeeds.
- D-8 trace is documented in the PR body and either passes or opens a blocking
  fix item.
- No imports from internal `fastvideo.pipelines`, `fastvideo.models`,
  `fastvideo.layers`, or `fastvideo.worker` remain under
  `apps/dreamverse/server/`.
- **Manual QA on GPU4** (required, not optional):
  - Reclaim GPU4 (kill running dreamverse-server PID 2453227).
  - `CUDA_VISIBLE_DEVICES=4 uv run --locked --package dreamverse-server fastvideo serve --config apps/dreamverse/serve_configs/streaming_demo.yaml --host 0.0.0.0 --port 8009` boots cleanly.
  - `curl -s http://localhost:8009/health` returns 200.
  - `pytest apps/dreamverse/server/tests/ -m gpu -q` against the live deploy passes
    (this exercises `test_realtime_stress.py` and `test_session_logging.py`
    with a real GPU + real model weights).
  - Capture output in PR description.
  - Cleanup: `pkill -f 'fastvideo serve --config'` and restart the canonical
    Dreamverse-side server.

**Rollback:**

- Revert the Phase 2 PR.
- Dreamverse source repo remains canonical until Phase 7, so no production
  rollback is needed.
- If only one public import is missing, add a temporary product-local shim only
  if it is explicitly deleted in Phase 4 or Phase 5.
- If the GPU4 manual QA reveals a real-service bug, the Phase 2 PR is
  blocked; fix the bug and re-run the GPU4 smoke before merging.

### Phase 3 — Move FE + content

**Purpose:** Move the Dreamverse frontend, prompts, serve configs, launch
scripts, and product docs after backend tests are green.

**Files to move:**

- `Dreamverse/apps/web/**` → `apps/dreamverse/web/**`
- `Dreamverse/server/prompts/*.md` → `apps/dreamverse/server/prompts/*.md`
- `Dreamverse/AGENTS.md` → `apps/dreamverse/AGENTS.md`
- `Dreamverse/arch.md` → `apps/dreamverse/arch.md`
- `Dreamverse/design.md` → `apps/dreamverse/design.md`
- `Dreamverse/gpu-pool.svg` → `apps/dreamverse/gpu-pool.svg`
- `Dreamverse/gpu-pool.drawio` → `apps/dreamverse/gpu-pool.drawio`
- `Dreamverse/scripts/install_native_ffmpeg.sh` → `apps/dreamverse/scripts/install_native_ffmpeg.sh`
- `Dreamverse/scripts/smoke_local.sh` → `apps/dreamverse/scripts/smoke_local.sh`
- `Dreamverse/serve_configs/streaming_demo.yaml` → `apps/dreamverse/serve_configs/streaming_demo.yaml`
- `Dreamverse/serve_configs/streaming_demo.yaml` → `examples/serving/streaming_demo.yaml` as a public-safe copy.
- `Dreamverse/.agents/skills/launch-demo/scripts/*.sh` → `apps/dreamverse/scripts/launch/*.sh`

**CI additions:**

- Add `.github/workflows/ci-dreamverse-frontend.yml`.
- Ensure `.gitignore` covers scoped Next.js and Playwright artifacts.

**Steps:**

1. Move frontend tree without restructuring package internals.
2. Update relative paths in frontend scripts/configs from old repo root to
   `apps/dreamverse/` or FastVideo root as appropriate.
3. Move product prompts and docs.
4. Copy `streaming_demo.yaml` to `examples/serving/streaming_demo.yaml` with
   public-safe comments. The missing example config is tracked in
   [`integration-review.md`](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L259-L276).
5. Add frontend workflow.
6. Add docs/contributing link if not already added in Phase 1.

**Verification gate:**

- `cd apps/dreamverse/web && pnpm install --frozen-lockfile` succeeds.
- `cd apps/dreamverse/web && pnpm run build` succeeds.
- `cd apps/dreamverse/web && pnpm run test --if-present` succeeds (Vitest + tsc).
- **Frontend CI Playwright is intentionally DEFERRED to Phase 4** — at Phase 3
  the public `build_app` does not yet expose `/healthz`+`/readyz`+`/status`+
  `/prompt-system-config`+`/curated-presets`. The `ci-dreamverse-frontend.yml`
  scaffold from Phase 1 keeps Playwright steps commented out until Phase 4
  reactivates them. No PR note required.
- **Manual GPU4 Playwright smoke** (recommended): on this dev node, run
  `apps/dreamverse/web/e2e/frontend-shell.spec.ts` against the GPU4-deployed
  backend from Phase 2 manual QA + a `pnpm run dev` frontend at port 5274
  to confirm shell hydration. `backend-health.spec.ts` and
  `preset-prompt-generation.spec.ts` will fail until Phase 4 — that is
  expected; document as deferred.
- `uv run --locked --package dreamverse-server --extra test pytest apps/dreamverse/server/tests/ -m 'not gpu' -q`
  still succeeds.
- `examples/serving/streaming_demo.yaml` parses with FastVideo serve config.

**Rollback:**

- Revert the Phase 3 PR.
- If frontend CI is the only failing component, revert only the frontend move
  and keep Phase 2 backend in monorepo.
- If the public example config is wrong, revert the example copy separately;
  product deployment config can remain under `apps/dreamverse/serve_configs/`.

### Phase 4 — Promote GENERIC-PENDING + close drift items

**Purpose:** Finish generic backend promotions that should not live in
Dreamverse product code.

**Files likely to touch:**

`fastvideo/entrypoints/streaming/{server.py,health.py,gpu_pool.py,session_store.py,blob_store.py,prompt/safety.py,prompt/rewrite.py}`,
`fastvideo/tests/entrypoints/streaming/`, `apps/dreamverse/server/{main.py,config.py}`.

**Work items:**

1. Add `/healthz`, `/readyz`, `/status` to generic `build_app` ([contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L232-L259), [open item](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99)).
2. Add `GpuPool.run_async() -> AsyncIterator[VideoEvent]`; keep sync `run()` as collector ([D-12](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L47-L116), [D-12-B](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L310-L325)).
3. Decide and document `video_position_offset_sec` semantics ([VPO](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L119-L144)).
4. Define SessionStore/BlobStore lifecycle policy ([SBS](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L277-L292)).
5. Promote or absorb `runtime.py`, `utils.py`, prompt safety, rewrite payload,
   and health tests according to the generic-pending table.
6. Update Dreamverse server imports to consume the new generic public modules.

**Verification gate:**

- FastVideo streaming tests pass (existing CI suite).
- Dreamverse backend tests pass (`-m 'not gpu'` in CI; `-m gpu` on GPU4 manual).
- Contract tests for `/healthz`, `/readyz`, `/status` cover both healthy and
  not-ready states (added in this phase to `fastvideo/tests/entrypoints/streaming/`).
- `run_async()` cancellation test proves client disconnect can stop mid-work.
- VPO decision is documented in code comments/tests.
- SessionStore/BlobStore lifecycle policy has tests for replacement cleanup and
  disconnect expiry if implemented.
- **Frontend CI Playwright RE-ENABLED** at end of this phase: uncomment the
  Playwright steps in `ci-dreamverse-frontend.yml` (deferred since Phase 1)
  AND change CI to either (a) run against the mock backend
  `fastvideo.entrypoints.streaming.mock_server`, or (b) require a deployed
  staging URL via env var.
- **Manual GPU4 full E2E**: deploy `fastvideo serve` on GPU4 with the new
  health routes, run all 3 Playwright specs (`backend-health`, `frontend-shell`,
  `preset-prompt-generation`) against it. All 3 must pass — this is the
  re-baseline after Phase 3's deferral. Capture output in PR.

**Rollback:**

- Each generic-pending promotion should be a focused PR.
- If a generic promotion regresses FastVideo, revert that PR and keep the
  product-local code temporarily under `apps/dreamverse/server/`.
- If health route compatibility is wrong, revert route additions but keep
  tests skipped only with an explicit tracking issue.

### Phase 5 — Replace prompt enhancer fork (DR-1) + `cerebras_ifm` decision (DR-2)

**Purpose:** Remove the 1933-LOC Dreamverse prompt enhancer fork and keep only
product-specific prompt orchestration.

**Files likely to touch:** `apps/dreamverse/server/prompting/_internal_compat.py` (new), `main.py`, `config.py`, `session/controller.py`, product prompt tests, and only if DR-2 picks public provider: `fastvideo/entrypoints/streaming/prompt/providers/` plus `fastvideo/api/schema.py`.

**DR-1 migration shape:** create `_internal_compat.py` over public `PromptEnhancer`, preserve product response shapes, keep locked segment metadata / rollout IDs / lenient JSON fallback product-local, then delete the full fork ([DR-1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L174-L206)).

**DR-2 decision:** pick public `cerebras_ifm` provider + Literal, or Dreamverse-side custom provider registered with public enhancer. Default: Dreamverse-side unless another user needs IFM ([DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229)).

**Verification gate:**

- Product prompt tests pass through `_internal_compat.py`.
- Public prompt enhancer tests remain green.
- `cerebras`, `groq`, and selected `cerebras_ifm` path have unit tests or
  mocked-provider tests.
- Playwright preset prompt generation test passes against live BE+FE, matching
  the existing Dreamverse-side verification gate
  ([open-threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L495-L498)).

**Rollback:**

- Keep the deleted fork recoverable in git history.
- If product prompt behavior regresses, revert Phase 5 only and restore the
  product-local fork temporarily.
- Do not roll back Phase 2 or Phase 3 for a prompt-specific regression.

### Phase 6 — Switch CI/release

**Purpose:** Make monorepo CI/deploy authoritative after code migration. **Splits into 6 substeps to avoid coupling unrelated changes.**

**Files:** `ci-dreamverse-{backend,frontend}.yml` (already added in Phase 1/3), new `deploy-dreamverse-{backend,frontend}.yml`, `apps/dreamverse/README.md`, `docs/contributing/dreamverse-development.md`, README links.

#### Phase 6a — CI path-filter proof
- Land a no-op test PR that touches only `apps/dreamverse/web/README.md` and verify ONLY `ci-dreamverse-frontend.yml` runs (not Buildkite full suite, not `ci-precommit` if `paths-ignore` applied).
- Land a no-op test PR that touches only `apps/dreamverse/server/README.md` and verify ONLY `ci-dreamverse-backend.yml` + `ci-precommit` run.
- Land a no-op test PR that touches `fastvideo/api/__init__.py` (a no-op comment) and verify BOTH FastVideo CI AND `ci-dreamverse-backend.yml` run (because the path filter includes `fastvideo/api/**`).
- Document the verification PR URLs in the Phase 6a merge commit.

#### Phase 6b — Backend deploy dry run
- Add `deploy-dreamverse-backend.yml` with a manual dispatch trigger only (no `push` trigger yet).
- Run a manual deploy to a **staging** environment (not production).
- Verify the deployed backend serves `/healthz`, `/readyz`, `/status`, `/v1/stream` correctly using public Dreamverse FE pointing at the staging URL.

#### Phase 6c — Frontend deploy dry run
- Add `deploy-dreamverse-frontend.yml` (Vercel or Docker depending on team choice) with manual dispatch only.
- Deploy to a Vercel preview / staging Docker.
- Verify product Playwright e2e against the staging FE+BE pair.

#### Phase 6d — Staging cutover
- Switch the staging environment to use ONLY the monorepo deploys.
- Disable the Dreamverse repo's deploy workflows for staging.
- Run for at least 24-48 hours to observe stability.

#### Phase 6e — Old-repo CI freeze
- Disable Dreamverse repo CI workflows (set them to manual-only or remove).
- Update Dreamverse repo's README to point to the FastVideo monorepo for new contributions.
- DO NOT archive the repo yet — that's Phase 7.

#### Phase 6f — Production cutover
- Switch production deploy to use monorepo workflows.
- Add `push: branches: [main]` triggers to deploy workflows now that they're proven.
- Update DNS / load-balancer / external link integrations.
- Monitor for 1 week before Phase 7.

**Verification gate:** Each substep has its own gate; do not proceed to next substep without the prior one passing. Final gate after 6f: monorepo production deploys are stable for 1 week with no Dreamverse-repo dependency in the production path.

**Rollback:** Each substep is independently revertible. Rolling back Phase 6f means switching deploy back to Dreamverse repo (which Phase 6e left functional but disabled). Rolling back Phase 6e means re-enabling Dreamverse CI. The 6a-6c substeps create no production dependency and can be reverted by deleting workflow files.

### Phase 7 — Archive Dreamverse repo

**Purpose:** Finish consolidation only after monorepo product CI/deploy are proven.

**Steps:** commit Dreamverse README redirect to `hao-ai-lab/FastVideo/apps/dreamverse/`, tag/branch final Dreamverse state for rollback, disable old CI, archive repo, update external links, and update this memory dir.

**Verification gate:** backend and frontend monorepo deploys are live; product docs and Dreamverse README redirect point to FastVideo; no production job depends on the old repo.

**Rollback:** unarchive Dreamverse, restore backup branch/tag, disable monorepo deploy, and point external links back until fixed.

---

## Risk register

| Risk | Severity | Likelihood | Mitigation | Owner |
|---|---|---:|---|---|
| Build-system complexity from adding a product app to a Python ML repo | High | Medium | Use uv workspace for Python only; keep frontend standalone; exclude `apps*` from wheel. | FastVideo build owner |
| Python contributors needing Node | Medium | Medium | No root `package.json`; frontend CI and commands live under `apps/dreamverse/web/`. | Frontend owner |
| CI cost increase | Medium | Medium | Add Dreamverse path-specific workflows; add `paths-ignore` to broad workflows; rely on Buildkite monorepo diff watch lists. | CI owner |
| FastVideo PyPI release accidentally includes Dreamverse app | High | Low | Add `apps*` to `[tool.setuptools.packages.find]` and `[tool.wheel]` excludes; verify built wheel contents. | Release owner |
| Release cadence coupling | Medium | Medium | Keep FastVideo PyPI version release unchanged; Dreamverse uses Docker/Vercel deploy from app paths. | Release owner |
| Frontend tooling drift | Medium | Medium | Pin pnpm lockfile in `apps/dreamverse/web/`; no root JS workspace. | Frontend owner |
| Security surface enlargement | Medium | Medium | Product routes stay in `apps/dreamverse/server`; only generic health/streaming routes go into FastVideo. | Backend owner |
| Product-specific API leakage into `fastvideo.*` | High | Medium | Enforce import/module boundary; keep curated presets and prompt UX product-local. | Architecture owner |
| Migration regression | High | Medium | Phase gates; backend before frontend; can stop after any phase with Dreamverse repo still usable until Phase 7. | Migration owner |
| Lost git history on moved files | High | High | **`git mv` does not preserve history across repositories.** Choose between (a) fresh import (no history; record source SHA in PR body; original history stays in archived Dreamverse repo) or (b) `git subtree add` / `git filter-repo` import (preserves history but heavier). Document the choice in Phase 2 PR. | Migration owner |
| Migrated public assets ignored by root `.gitignore` | High | High | Root ignores `*.png`/`*.jpg`/`*.mp4`/`*.gif`; Phase 1 must add `!apps/dreamverse/web/public/**` etc. unignore rules. Verify with `git check-ignore -v` before Phase 3 commits. | Migration owner |
| CORS open to `*` in product server | Medium | Medium | [`Dreamverse/server/main.py:99`](file:///home/william5lin/Dreamverse/server/main.py#L99) sets `allow_origins=["*"]`. After move, decide: (a) keep dev-only and document; (b) tighten to known FE origins; (c) deployment-firewall. Document in `apps/dreamverse/server/README.md`. | Backend owner |
| Write endpoints (`/prompt-system-config`, `/curated-presets/append`) | Medium | Medium | [`routes/presets.py:106`](file:///home/william5lin/Dreamverse/server/routes/presets.py#L106) exposes write routes. Decide auth/rate-limit policy; either gate behind devtools origin OR add auth before production cutover. | Backend owner |
| Prompt enhancer behavior regression | High | Medium | Build DR-1 shim with product tests before deleting fork; keep rollback path to restore fork. | Prompt owner |
| `cerebras_ifm` production gap | Medium | Medium | Decide DR-2 before Phase 5 merge; default Dreamverse-side provider unless public user exists. | Product owner |
| Health route compatibility gap | High | Medium | Land `/healthz`, `/readyz`, `/status` in generic build_app with tests before FE switches to FastVideo flavor. | Streaming owner |
| `video_position_offset_sec` ambiguity | Medium | Medium | Decide VPO semantics in Phase 4; encode in tests. | Audio/runtime owner |
| Session/blob memory leak | Medium | Medium | Define lifecycle policy before high-traffic deploy; add TTL/replacement cleanup tests. | Runtime owner |
| Existing FastVideo CI false positives on frontend-only changes | Medium | Medium | Apply path filters; verify with a frontend-only test PR. | CI owner |
| Dreamverse repo archive too early | High | Low | Archive only after monorepo staging deploy and production deploy are proven. | Release owner |

---

## Verification gates

| Phase | Required checks | Stop condition |
|---|---|---|
| 0 | #1288 merged; public typed events and `generate_async` import; baseline API/contract tests green. | #1288 missing or regressed. |
| 1 | uv workspace sync; pre-commit on changed root/app files; wheel excludes `apps*`; no root `package.json`. | FastVideo package build includes app code or uv workspace cannot sync. |
| 2 | Dreamverse backend tests green; import contract green; D-8 trace complete; lsp/pre-commit clean. | Product server imports private FastVideo modules or D-8 is silently dropped. |
| 3 | Frontend install/build/test/Playwright green; backend tests still green; public example config parses. | Frontend cannot build from `apps/dreamverse/web/`. |
| 4 | Streaming tests green; health routes covered; `run_async()` cancellation covered; VPO/SBS decisions encoded. | Generic backend regression or unresolved route contract. |
| 5 | Prompt shim tests green; selected IFM path works; product Playwright preset prompt flow green. | Prompt behavior differs from product expectations. |
| 6 | Backend-only and frontend-only CI PRs trigger correct workflows; staging deploy succeeds. | Existing FastVideo CI is noisy for frontend-only changes or deploy fails. |
| 7 | Monorepo production deploy proven; old repo redirect committed; no jobs depend on old repo. | Any production dependency still points to Dreamverse repo. |

Global verification requirements:

- `lsp_diagnostics` clean on changed Python files.
- `pre-commit run --files <changed paths>` for non-excluded files.
- Relevant tests pass before each phase merges.
- Do not run GPU-heavy full suite for frontend-only changes.
- Capture command output in PR descriptions.

---

## Rollback strategy

| Phase | Rollback path | Data loss risk | Notes |
|---|---|---:|---|
| 0 | Do not proceed until #1288 is fixed or reverted. | Low | No Dreamverse files moved. |
| 1 | Revert skeleton/tooling PR. | Low | Empty dirs and root tooling only. |
| 2 | Revert backend move PR; Dreamverse repo remains source of truth. | Low | Move commits do NOT preserve cross-repo history (`git mv` only preserves within a single repo). Original Dreamverse history stays in the archived Dreamverse repo. The Phase 2 PR body must record the originating Dreamverse SHA(s) for traceability. |
| 3 | Revert frontend/content move PR; keep backend if green. | Low | Product assets still exist in Dreamverse repo. |
| 4 | Revert individual generic promotion PR; temporarily keep product-local shim. | Medium | Avoid reverting unrelated generic backend changes. |
| 5 | Restore prompt fork from previous commit; keep shim behind flag if useful. | Medium | Prompt behavior rollback only. |
| 6 | Disable monorepo deploy workflows; re-enable Dreamverse repo CI/deploy. | Low | Code can remain in monorepo while deploy routes old repo. |
| 7 | Unarchive Dreamverse repo; restore from backup branch; turn off monorepo deploy. | Medium | Requires external link rollback. |

Rollback principles:

- Never archive Dreamverse before monorepo deploy is proven.
- Keep each phase as a separate PR or commit group.
- Keep generic backend promotions independent from product moves.
- Keep `apps*` wheel exclusion in place even if product phases are reverted.
- If frontend migration fails, do not roll back generic backend work.

---

## Open questions / decisions needed

| Decision | Needed by | Default / note | Source |
|---|---|---|---|
| `cerebras_ifm` provider path | Phase 5 | Dreamverse-side custom provider unless another public user exists | [DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229) |
| `video_position_offset_sec` semantics | Phase 4 | Decide persistent vs per-segment; decision is overdue | [VPO](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L361-L364) |
| SessionStore/BlobStore lifecycle | Phase 4 | Define TTL/LRU/hard max/blob cleanup/disconnect expiry | [SBS](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L277-L292) |
| Health route response shapes | Phase 4 | Match Dreamverse `/healthz`, `/readyz`, `/status` before FE switch | [health routes](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99) |
| Benchmark destination | Phase 4 | Pick `fastvideo/tests/benchmarks/` or root `benchmarks/` before moving files | migration table above |
| Standalone upsampler CLI | Deferred | Defer or port to FastVideo CLI | [upsampler](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L239-L258) |
| Layerwise offload utility | Deferred | Defer unless a deployment requires it | [offload](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L224-L238) |

---

## Action items

| # | Pri | Effort | Phase | Owner | Action |
|---:|---|---|---|---|---|
| 1 | P0 | L | 0 | FastVideo backend | Land #1288 and update memory state. |
| 2 | P0 | S | 1 | Build | Add uv workspace, `apps*` excludes, web pre-commit exclude, scoped `.gitignore`. |
| 3 | P0 | S | 1 | Docs | Add `docs/contributing/dreamverse-development.md` and optional `mkdocs.yml` nav. |
| 4 | P1 | M | 2 | Product backend | Move backend files and add import contract test. |
| 5 | P1 | S | 2 | Runtime | Trace D-8 `ltx2_image_crf` and add/adjust test. |
| 6 | P1 | M | 3 | Frontend | Move Next.js frontend and land frontend CI. |
| 7 | P2 | S | 3 | Examples | Copy `streaming_demo.yaml` to `examples/serving/streaming_demo.yaml`. |
| 8 | P1 | M-L | 4 | Streaming | Add `/healthz`, `/readyz`, `/status` to `build_app` with tests. |
| 9 | P2 | M | 4 | Streaming | Add `GpuPool.run_async()` and cancellation propagation. |
| 10 | P2 | S | 4 | Audio/runtime | Decide VPO semantics and encode in tests. |
| 11 | P2 | M | 4 | Runtime | Define SessionStore/BlobStore lifecycle policy. |
| 12 | P1 | M | 5 | Prompt | Replace prompt enhancer fork with `_internal_compat.py` shim. |
| 13 | P1 | S-M | 5 | Product | Decide and implement `cerebras_ifm` path. |
| 14 | P1 | M | 6 | CI | Prove split backend/frontend CI and no FastVideo CI noise on frontend-only changes. |
| 15 | P1 | M | 6 | Release | Add product deploy workflows from monorepo paths. |
| 16 | P2 | S | 7 | Release | Archive Dreamverse repo after redirect commit and monorepo deploy proof. |
| 17 | P3 | S-M | Deferred | CLI | Decide standalone upsampler CLI. |
| 18 | P3 | M | Deferred | Performance | Defer layerwise offload until deployment evidence exists. |

---

## Cross-references

- Repo paths: [README](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/README.md#L73-L80).
- Typed API and schema: [design core](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L24-L43), [schema surface](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L45-L72).
- Continuation state: [cross-repo mapping](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L108-L146).
- Streaming routes: [route contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L232-L259).
- GpuPool and router decisions: [D-12](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L47-L116), [D-15](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L118-L201).
- Open work and verification: [open threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L15-L45), [gates](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L480-L498).
- Runbook: [verification commands](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/runbook.md#L79-L124).
- Previous tradeoff analysis: [integration review](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L371-L563).

---

## Comparison to `integration-review.md` (now superseded for recommendation)

`integration-review.md` recommended constrained Option D: generic backend in
FastVideo, Dreamverse product in a separate repo
([recommendation](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/integration-review.md#L543-L563)).

The user chose Option B+ instead.

Changed recommendation: Dreamverse product moves into `apps/dreamverse/`; generic backend stays at `fastvideo.entrypoints.streaming.*`; Dreamverse server does not become `fastvideo.entrypoints.dreamverse.*`; frontend remains standalone with no root `package.json`; FastVideo PyPI excludes `apps*`.

Still authoritative from `integration-review.md`: drift audit findings, OSS comparison material not contradicted by Option B+, health route gap, DR-1/DR-2 prompt work, D-8 `ltx2_image_crf`, VPO/SBS decisions, missing public streaming demo config, and the decision to defer layerwise offload / standalone upsampler CLI unless product need is proven.

Operationally, this plan is the execution document. Use it for Phase 1 onward.
