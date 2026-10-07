#!/usr/bin/env bash
# DreamVerse needs a GPU for import-time device resolution, but it does not
# build or exercise fastvideo-kernel. A checksummed Node archive is installed
# in the disposable Slurm container because the shared CI image is
# Python/CUDA focused.
set -euo pipefail

node_version=v22.23.2
case $(uname -m) in
  aarch64 | arm64)
    node_arch=arm64
    node_archive_sha256=013b59cfd2819703a6f4a14ab891fc46fc2a4e3f5bcd92de3fb4929b43e35b30
    ;;
  x86_64 | amd64)
    node_arch=x64
    node_archive_sha256=b294a556e639d64338823920e5866c21c02741742d2e1529ee1a225c1ec9252a
    ;;
  *)
    echo "Unsupported architecture for DreamVerse Node runtime: $(uname -m)" >&2
    exit 2
    ;;
esac
node_archive="node-${node_version}-linux-${node_arch}.tar.gz"
node_runtime_root=$(mktemp -d -t fastvideo-node.XXXXXX)
node_archive_path="${node_runtime_root}/${node_archive}"
node_install_dir="${node_runtime_root}/${node_archive%.tar.gz}"
curl --proto '=https' --tlsv1.2 --retry 5 --retry-all-errors \
  --location --fail --silent --show-error \
  "https://nodejs.org/dist/${node_version}/${node_archive}" \
  --output "$node_archive_path"
printf '%s  %s\n' "$node_archive_sha256" "$node_archive_path" | sha256sum --check --status
tar -xzf "$node_archive_path" -C "$node_runtime_root"
export PATH="${node_install_dir}/bin:${PATH}"
node --version
npm --version

export PYTHONPATH="$(pwd)/apps/dreamverse${PYTHONPATH:+:$PYTHONPATH}"
pytest apps/dreamverse/dreamverse/tests -q

cd apps/dreamverse/web
npm ci
npm run typecheck
npm test
machine_arch=$(uname -m)
if [[ $machine_arch =~ ^(aarch64|arm64)$ ]]; then
  npx playwright install --with-deps chromium firefox
else
  npx playwright install --with-deps chromium webkit firefox
fi

master_port=${MASTER_PORT:-7959}
BACKEND_PORT=${BACKEND_PORT:-$((master_port + 50))}
python -m uvicorn dreamverse.mock_server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
mock_server_pid=$!
cleanup() {
  kill "$mock_server_pid" 2>/dev/null || true
  wait "$mock_server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in {1..30}; do
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz" && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz"

if [[ $machine_arch =~ ^(aarch64|arm64)$ ]]; then
  # Playwright WebKit traps before opening a page on Linux ARM64, and its
  # bundled Chromium lacks the H.264/AAC codecs used by the fMP4 assertions.
  # Firefox covers every flow, including streaming. Chromium and its mobile
  # profile still cover all codec-independent UI behavior on GB200.
  BACKEND_HOST=127.0.0.1 BACKEND_PORT="$BACKEND_PORT" CI=1 \
    npm run e2e -- --project=firefox
  BACKEND_HOST=127.0.0.1 BACKEND_PORT="$BACKEND_PORT" CI=1 \
    npm run e2e -- \
      --project=chromium \
      --project=mobile-chromium \
      --grep-invert='streams, plays, and surfaces a downloadable clip|starts a new project and switches back to the prior session|saved projects persist across a page reload'
else
  BACKEND_HOST=127.0.0.1 BACKEND_PORT="$BACKEND_PORT" CI=1 \
    npm run e2e -- \
      --project=chromium \
      --project=webkit \
      --project=firefox \
      --project=mobile-safari \
      --project=mobile-chromium
fi
