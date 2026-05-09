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

## Launch Dreamverse

Start the backend with the installed Dreamverse commands:

```bash
dreamverse-server --port 8009
dreamverse-mock-server --port 8009
```
