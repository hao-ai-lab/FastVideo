# Dreamverse

Dreamverse is the FastVideo realtime video product. It lives in this
monorepo under `apps/dreamverse/`. The backend runtime under
`apps/dreamverse/server/` is product-local; do not replace it with public
`fastvideo.entrypoints.streaming.*` shims until a future promotion phase makes
those public surfaces drop-in compatible.

The migration source of truth is
`.agents/memory/dreamverse-integration/integration-plan.md`.

## Backend launch

Use the repo-local wrappers from the FastVideo checkout root:

```bash
apps/dreamverse/scripts/dreamverse-server --port 8009
apps/dreamverse/scripts/dreamverse-mock-server --port 8009
```

The workspace member currently keeps `[tool.uv] package = false`, so `uv` does
not install Dreamverse console scripts for this checkout. If `dreamverse-server`
appears earlier on `PATH`, it may be a stale editable install from the archived
Dreamverse repo (for example `/home/william5lin/Dreamverse/server/main.py`).
Prefer the wrapper paths above during this migration.
