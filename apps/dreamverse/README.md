# Dreamverse

Dreamverse is the FastVideo realtime video generation & editing platform. It lives in this monorepo under `apps/dreamverse/`.

## Install Dreamverse

You can install Dreamverse using one of the methods below.

### Method 1: With uv pip

```bash
pip install --upgrade pip
pip install uv
uv pip install "fastvideo[dreamverse]"
```

### Method 2: From source

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo

pip install --upgrade pip
pip install uv
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
binary:

```bash
bash apps/dreamverse/scripts/install_native_ffmpeg.sh
```

This builds and installs into `~/opt/ffmpeg-native/` and writes
`apps/dreamverse/scripts/ffmpeg-env.sh`. Source it before starting the backend
so Dreamverse uses the custom FFmpeg binary:

```bash
source apps/dreamverse/scripts/ffmpeg-env.sh
dreamverse-server
```

Docker images already run this FFmpeg build during image creation and source the
generated environment file at container startup. The installer supports Linux
`x86_64` and `aarch64`.

## Launch Dreamverse

Start the backend with the installed Dreamverse commands:

```bash
dreamverse-server --port 8009
dreamverse-mock-server --port 8009
```
