# Dreamverse

Dreamverse is the FastVideo realtime video product. It lives in this
monorepo under `apps/dreamverse/`. The backend runtime under
`apps/dreamverse/dreamverse/` is product-local; do not replace it with public
`fastvideo.entrypoints.streaming.*` shims until a future promotion phase makes
those public surfaces drop-in compatible.

The migration source of truth is
`.agents/memory/dreamverse-integration/integration-plan.md`.

## Install From FastVideo

Install the Dreamverse runtime dependencies with the FastVideo extra:

```bash
pip install "fastvideo[dreamverse]"
dreamverse-server --port 8009
```

The base `pip install fastvideo` package still includes the Dreamverse Python
package and `dreamverse-server` / `dreamverse-mock-server` commands. Running
those commands without the `[dreamverse]` extra exits immediately with a message
to install `fastvideo[dreamverse]`, because the cloud prompt runtime
dependencies are intentionally gated by the extra.

## Repo-Local Backend Launch

Use the repo-local wrappers from the FastVideo checkout root:

```bash
apps/dreamverse/scripts/dreamverse-server --port 8009
apps/dreamverse/scripts/dreamverse-mock-server --port 8009
```

The workspace member keeps `[tool.uv] package = false`, so the wrappers remain
the most explicit way to launch the backend from a source checkout. If
`dreamverse-server` appears earlier on `PATH`, confirm it points at this
checkout or reinstall FastVideo with the `dreamverse` extra.
