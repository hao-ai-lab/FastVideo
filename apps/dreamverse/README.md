# Dreamverse

Dreamverse is the FastVideo realtime video generation & editing platform. It lives in this monorepo under `apps/dreamverse/`.

## Install Dreamverse

You can install Dreamverse using one of the methods below.

### Method 1: With uv pip

```bash
pip install --upgrade pip
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install "fastvideo[dreamverse]"
```

### Method 2: From source

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo

pip install --upgrade pip
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dreamverse]"
```

### Method 3: Using Docker

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo

apps/dreamverse/docker/docker_build.sh
```

See `apps/dreamverse/docker/README.md` for Docker build and run option details.

## Optional: Building FFmpeg For Better Performance

For full streaming performance in a non-Docker install, build a custom FFmpeg
binary from a FastVideo source checkout. The command below is repo-relative,
so run it from the repository root:

```bash
bash apps/dreamverse/scripts/install_native_ffmpeg.sh
```

The installer supports Linux `x86_64` and `aarch64`. It prefers conda-forge
triplet compilers when those commands are on `PATH`, otherwise it falls back to
system `gcc`/`g++` (plain venv). On `x86_64`, x264's hand-tuned SIMD also
requires `nasm`; install via whichever path fits your host:

```bash
sudo apt install nasm                       # Debian/Ubuntu
conda install -c conda-forge nasm           # inside an active conda env
```

No sudo and no conda? Build `nasm` from source (~30s, installs into `$HOME`):

```bash
(
  mkdir -p "$HOME/src" "$HOME/opt" && cd "$HOME/src"
  curl -fsSL -O https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz
  tar -xf nasm-2.16.03.tar.gz && cd nasm-2.16.03
  ./configure --prefix="$HOME/opt/nasm" && make -j"$(nproc)" && make install
)
export PATH="$HOME/opt/nasm/bin:$PATH"      # add to ~/.bashrc to persist
```

The installer writes to `~/opt/ffmpeg-native/` and emits
`apps/dreamverse/scripts/ffmpeg-env.sh`. Source it before starting the backend
so Dreamverse uses the custom FFmpeg binary:

```bash
source apps/dreamverse/scripts/ffmpeg-env.sh
dreamverse-server
```

Docker images already run this FFmpeg build during image creation and source the
generated environment file at container startup.

## Launch Dreamverse

Start the backend with the installed Dreamverse commands:

```bash
dreamverse-server --port 8009
dreamverse-mock-server --port 8009
```

## Frontend Setup

Install the web dependencies once from the FastVideo checkout:

```bash
cd apps/dreamverse/web
pnpm install --frozen-lockfile
```

The frontend package also has an npm lockfile, but the bundled launch scripts
use `pnpm`.

## Quick Start: Local GPU

### Start Backend

Export the API keys used for prompt rewrite and prompt enhancement:

```bash
export CEREBRAS_API_KEY=...
export GROQ_API_KEY=...
```

If you built the optional native FFmpeg binary above, source its environment
file in the same shell before starting the backend:

```bash
source apps/dreamverse/scripts/ffmpeg-env.sh
dreamverse-server --host 0.0.0.0 --port 8009
```

The Dreamverse backend defaults to `0.0.0.0:8009` and starts one GPU worker on
the first visible GPU by default.

### Check Readiness

In another shell, verify that the backend process is alive:

```bash
curl http://localhost:8009/healthz
```

Then wait for GPU workers and startup warmup to finish:

```bash
curl http://localhost:8009/readyz
```

You can also run the same readiness path with:

```bash
BACKEND_HOST=localhost BACKEND_PORT=8009 apps/dreamverse/scripts/smoke_local.sh
```

If a backend is already running and you only want the script to probe it:

```bash
DREAMVERSE_SMOKE_START_BACKEND=0 apps/dreamverse/scripts/smoke_local.sh
```

### Start Frontend

Start the frontend:

```bash
cd apps/dreamverse/web
BACKEND_HOST=localhost BACKEND_PORT=8009 pnpm run dev
```

Open `http://localhost:5299`.

## Quick Start: Mock Backend (For UI development)

The mock server emulates the Dreamverse backend protocol and streams a
synthetic FFmpeg-generated fMP4 clip, so the frontend can run without a GPU.

```bash
dreamverse-mock-server --latency 200 --port 8009
```

## Tests

Run the focused backend tests that validate local startup wiring, config, GPU
selection, and mock-server behavior:

```bash
pytest apps/dreamverse/dreamverse/tests/test_config.py \
  apps/dreamverse/dreamverse/tests/test_entrypoints.py \
  apps/dreamverse/dreamverse/tests/test_gpu_pool.py \
  apps/dreamverse/dreamverse/tests/test_mock_server.py -q
```

Run the broader Dreamverse backend suite:

```bash
pytest apps/dreamverse/dreamverse/tests -q
```

Run the frontend tests:

```bash
cd apps/dreamverse/web
pnpm test
```

Run the frontend e2e tests:

```bash
cd apps/dreamverse/web
pnpm run e2e
```

## Troubleshooting

`dreamverse-server` exits with an install hint

- install the Dreamverse extra with `uv pip install -e ".[dreamverse]"` from a
  source checkout, or `uv pip install "fastvideo[dreamverse]"` from PyPI.

Prompt-provider environment variable errors

- set `CEREBRAS_API_KEY`
- set `GROQ_API_KEY`
- direct `dreamverse-server` launches do not source `~/.env`; export the keys
  in the shell or use the bundled launch scripts, which source `~/.env`.

`/readyz` stays at `503`

- wait for model loading and startup warmup to finish
- confirm a compatible CUDA GPU is visible to the process
- check backend logs for worker startup or warmup failures
- for a startup/debug pass without warmup, set
  `FASTVIDEO_ENABLE_STARTUP_WARMUP=0` before starting the backend

Only one GPU is used

- this is the default local behavior
- set `FASTVIDEO_GPU_COUNT=<N>` to start N GPU worker subprocesses inside one
  backend instance
- set `FASTVIDEO_GPU_COUNT=all` to start one worker for every visible GPU
- use `CUDA_VISIBLE_DEVICES` first if you need to pin the visible GPU set

Frontend cannot connect to backend

- confirm the backend is running on `8009`; if not, point the frontend at it
  with `BACKEND_HOST=<host> BACKEND_PORT=<port> pnpm run dev`
- confirm `http://localhost:8009/healthz` responds before starting the frontend
- confirm `http://localhost:8009/readyz` returns `200` before clicking Generate
- use `apps/dreamverse/scripts/smoke_local.sh` for a repeatable local startup
  check

Mock backend fails during startup

- install FFmpeg or set `FASTVIDEO_FFMPEG_BIN` to an FFmpeg binary
- for non-mock local GPU streaming performance, use the native FFmpeg installer
  described above

## Notes

Dreamverse owns its backend app under `apps/dreamverse/dreamverse/`. It expects
`dreamverse-server`, not `fastvideo serve`.
