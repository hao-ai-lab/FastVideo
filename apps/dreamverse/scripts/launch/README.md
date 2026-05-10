# Dreamverse Launch Scripts

These scripts are convenience wrappers for local demos and lower-level backend
checks. The main Dreamverse README documents the normal manual startup path.

## One-Command Demo

From the FastVideo checkout:

```bash
apps/dreamverse/scripts/launch/launch_demo.sh
```

The launcher starts the Dreamverse backend and frontend, polls readiness, and
prints the active URLs. It defaults to `dreamverse-server` on backend port
`8009` and frontend port `5274`.

Useful overrides:

```bash
BE_PORT=8010 FE_PORT=5274 apps/dreamverse/scripts/launch/launch_demo.sh
NO_FRONTEND=1 apps/dreamverse/scripts/launch/launch_demo.sh
NO_BROWSER=1 apps/dreamverse/scripts/launch/launch_demo.sh
```

## Individual Scripts

`launch_backend_dreamverse.sh` starts the full Dreamverse backend path used by
the web app.

`launch_frontend.sh` starts the Next.js frontend and installs `pnpm`
dependencies when `node_modules/` is missing.

`launch_backend_fastvideo.sh` starts the typed `fastvideo serve --config` path
for lower-level serve-config checks. It is not a full Dreamverse app replacement
because the current frontend still depends on Dreamverse-specific routes.
