# Dreamverse Modal deployment

This directory contains the Modal wrapper for running the registry-built
Dreamverse Docker image on a Modal B200 container. The wrapper pulls a prebuilt Docker image and exposes the dreamverse application running inside.


## 1. Install Modal CLI

Install and configure the Modal CLI for the target workspace/profile:

```bash
pip install modal
modal token set --token-id <token-id> --token-secret <token-secret> --profile=<profile-name>
modal profile activate <profile-name>
```

## 2. Create the API key secret

Dreamverse needs several API keys for prompt-rewriter LLMs and model access in the Modal secret named
`dreamverse-api-keys`. Create or replace it with placeholders like this:

```bash
modal secret create dreamverse-api-keys \
  CEREBRAS_API_KEY=<your-cerebras-key> \
  GROQ_API_KEY=<your-groq-key> \
  HF_TOKEN=<your-huggingface-token> \
  --force
```

## 3. Deploy with Docker image

`DREAMVERSE_IMAGE` is required at deploy time:

```bash
DREAMVERSE_IMAGE=ghcr.io/<org>/<repo>/dreamverse:<tag> \
  modal deploy apps/dreamverse/scripts/modal/modal_app.py
```

Use a SHA-specific tag, not `latest`. The image workflow publishes backend and
UI variants for CUDA 12.6.3 and CUDA 13.0.0:

- `dreamverse-backend-cuda13.0.0-sha-*` or
  `dreamverse-ui-cuda13.0.0-sha-*` for the CUDA 13 / cu130 images
- `dreamverse-backend-cuda12.6.3-sha-*` or
  `dreamverse-ui-cuda12.6.3-sha-*` for the CUDA 12 / cu126 images

The Dockerfile defaults to the CUDA 13 / cu130 lane, which is the recommended
image for this B200 deployment. Choose the `ui` variant only when the backend
should serve the static UI.

For local image build details, see
`apps/dreamverse/docker/README.md` and `apps/dreamverse/docker/docker_build.sh`.


## 4. Validate the deployment

Set the URL returned by `modal deploy`, then probe the backend:

```bash
URL=https://<workspace>--dreamverse-b200-serve.modal.run
curl -fsS --max-time 300 "$URL/healthz"
curl -fsS --max-time 300 "$URL/status"
curl -fsS --max-time 300 "$URL/readyz"
```

Check logs with function and container IDs so you can confirm only one container
is active:

```bash
modal app logs dreamverse-b200 \
  --tail 200 \
  --timestamps \
  --show-function-id \
  --show-container-id
```

Useful Modal commands:

```bash
modal app list
modal app list --json
modal billing report --for today --resolution h
modal app stop dreamverse-b200
```

## Runtime details

### Volumes and assumed locations

`modal_app.py` creates the named volumes if they do not exist and mounts them at
the locations expected by the image:

| Modal volume | Container path | Environment variable | Purpose |
| --- | --- | --- | --- |
| `dreamverse-hf-cache` | `/root/.cache/huggingface` | `HF_HOME` | Hugging Face model/cache data |
| `dreamverse-state` | `/var/lib/dreamverse` | `FASTVIDEO_DREAMVERSE_HOME` | Dreamverse outputs, session logs, and runtime state |

### Useful dev env vars

Most deploys only need the Modal secret and required `DREAMVERSE_IMAGE`.
`DREAMVERSE_IMAGE` is read from your local shell at deploy time; the runtime
knobs below live in the image or `modal_app.py` unless you intentionally change them:

- `DREAMVERSE_IMAGE`: deploy a prebuilt SHA-specific image tag without editing `modal_app.py`.
- `FASTVIDEO_ENABLE_DEVTOOLS`: enable Dreamverse devtools behavior.
- `FASTVIDEO_PROMPT_PROVIDER` / `FASTVIDEO_PROMPT_*_MODEL`: try prompt-rewriter provider or model choices.
- `FASTVIDEO_ENABLE_STARTUP_WARMUP`: trade slower startup for a warmer first request.
- `DREAMVERSE_MAX_AUTOTUNE`: enable or disable PyTorch Inductor max-autotune for the compiled Dreamverse runtime.

The Modal wrapper defaults to torch compile with Inductor max-autotune enabled
for the fastest generation after startup warmup. If you need shorter
compile/warmup time, disable max-autotune via:

```bash
DREAMVERSE_IMAGE=ghcr.io/<org>/<repo>/dreamverse:<tag> \
DREAMVERSE_MAX_AUTOTUNE=0 \
  modal deploy apps/dreamverse/scripts/modal/modal_app.py
```

### Autoscaling and cost safety

The current deployed script keeps exactly one B200 container warm by setting
`min_containers=1` and `max_containers=1`. `min_containers=1` prevents Modal
from scaling the deployment down to zero after idle periods, which avoids paying
the expensive torch-compile/startup-warmup cost again on the next request.
`max_containers=1` caps concurrency so concurrent requests queue onto the warm
container instead of spawning additional B200 containers and multiplying cost.

### Common gotchas

- Modal profile/workspace matters: create the secret and deploy with the same active profile.
- `modal secret create ... --force` replaces the existing `dreamverse-api-keys` secret.
- Use the exact URL printed by `modal deploy`; the placeholder URL above is only an example shape.
- B200 cold starts and first model downloads can make `/readyz` slow. Check logs before redeploying.
- `modal app stop dreamverse-b200` stops the app; it is not a read-only inspection command.
