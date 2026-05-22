# Handoff: ship Dreamverse UI in the Modal Docker image

## Current state

- Modal wrapper: `apps/dreamverse/scripts/modal/modal_app.py`.
- Modal docs: `apps/dreamverse/scripts/modal/README.md`.
- The wrapper pulls a prebuilt Docker image, requests one B200, caps the app with
  `max_containers=1`, and exposes port `8009` as one Modal URL.
- The validated image is backend-only:
  `ghcr.io/hao-ai-lab/fastvideo/dreamverse:dreamverse-cuda12.9.1-sha-af2ee9c`.

## Next objective

Build a separate Dreamverse Docker image that includes the web UI and backend,
then deploy that image with the existing Modal wrapper so the user gets one URL
for both UI and API/WebSocket traffic.

The Modal script should not need major changes if the new image still runs:

```bash
dreamverse-server --host 0.0.0.0 --port 8009
```

## Static UI contract

The backend already serves an exported frontend bundle when it exists. See:

- `apps/dreamverse/dreamverse/config.py`
- `apps/dreamverse/dreamverse/main.py`

Static directories currently discovered by the backend:

- `apps/dreamverse/web/out`
- `apps/dreamverse/web/dist`
- `apps/dreamverse/prod-ui/out`
- `apps/dreamverse/prod-ui/dist`

So the simplest route is: build/export the Next UI into one of those locations
inside the Docker image, then let `dreamverse-server` serve it from the same
FastAPI app and same Modal URL.

## Suggested implementation path

1. Add an opt-in Docker build path for a UI-included image.
   - Keep the current backend-only image path working.
   - Prefer a build arg such as `BUILD_DREAMVERSE_UI=1` or a separate Dockerfile
     only if the conditional path gets messy.
2. In the UI build path, install the web toolchain and build the frontend under
   `apps/dreamverse/web/`.
   - `package.json` has `build` scripts.
   - Both `pnpm-lock.yaml` and `package-lock.json` exist; pick one package
     manager deliberately and avoid updating both by accident.
3. Ensure the build output lands in a backend-served static directory.
   - Current `next build` may not create `web/out` unless Next is configured for
     static export.
   - If using `web/out`, update Docker ignore rules because
     `apps/dreamverse/docker/Dockerfile.dockerignore` currently excludes
     `apps/dreamverse/web/out/` and `apps/dreamverse/web/dist/`.
4. Build and push a new registry tag, for example:

   ```bash
   DREAMVERSE_IMAGE=ghcr.io/hao-ai-lab/fastvideo/dreamverse:<ui-tag> \
     apps/dreamverse/docker/docker_build.sh
   docker push ghcr.io/hao-ai-lab/fastvideo/dreamverse:<ui-tag>
   ```

5. Redeploy Modal with the new tag:

   ```bash
   DREAMVERSE_IMAGE=ghcr.io/hao-ai-lab/fastvideo/dreamverse:<ui-tag> \
     modal deploy apps/dreamverse/scripts/modal/modal_app.py
   ```

## Validation checklist

1. Backend endpoints return 200:

   ```bash
   curl -fsS "$URL/healthz"
   curl -fsS "$URL/status"
   ```

2. The Modal root URL returns HTML instead of 404:

   ```bash
   curl -fsS "$URL/" | head
   ```

3. The browser loads the UI from the Modal URL.
4. The UI can connect to same-origin `/ws`.
5. Modal logs show only one B200 function container during validation.

## Things likely to cause hiccups

- Static export: the current Next config is not obviously configured to emit
  `web/out`; confirm the actual output before changing Docker.
- Docker ignore: prebuilt `web/out`/`web/dist` directories are currently ignored
  by the Dreamverse Docker ignore file.
- Same-origin assumptions: the UI should call `/healthz`, `/status`, `/readyz`,
  and `/ws` on the same Modal origin. Avoid requiring a separate Next server for
  this one-URL path.
- Secrets: keep API keys in Modal secrets. Do not bake `.env` or secret values
  into the Docker image.
- Cost safety: keep `max_containers=1` while testing B200 deploys.
