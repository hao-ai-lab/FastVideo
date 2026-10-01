#!/usr/bin/env bash
# Installed with the trusted controller; never load this entrypoint from a PR.
set -euo pipefail
umask 077

[[ ${FASTVIDEO_CI_REPOSITORY:-} == https://github.com/hao-ai-lab/FastVideo.git ]]
[[ ${FASTVIDEO_CI_COMMIT:-} =~ ^[0-9a-f]{40}$ ]]
[[ ${FASTVIDEO_CI_GPUS:-} =~ ^[1-4]$ ]]
[[ ${FASTVIDEO_CI_EXTRAS:-} =~ ^[a-z][a-z0-9-]*(,[a-z][a-z0-9-]*)*$ ]]
[[ ${FASTVIDEO_CI_SCRIPT:-} == .buildkite/scripts/unit_test.sh ||
  ${FASTVIDEO_CI_SCRIPT:-} =~ ^\.buildkite/scripts/lanes/[a-z_]+\.sh$ ]]
[[ ${FASTVIDEO_CI_KERNEL:-} =~ ^[01]$ ]]
[[ ${FASTVIDEO_FA4:-} =~ ^[01]$ ]]

export PATH="/opt/venv/bin:/root/.local/bin:$PATH"
export VIRTUAL_ENV=/opt/venv
export HF_HOME=/workspace/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR=/workspace/uv-cache
export UV_LINK_MODE=copy
export WANDB_MODE=offline
export FASTVIDEO_CI_LOCAL_ONLY=1
export FASTVIDEO_SSIM_BOOTSTRAP_MODE=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1
mkdir -p /workspace/artifacts "$HF_HUB_CACHE"

# Results are diagnostic only. The controller trusts the container exit status.
finish() {
  worker_rc=$?
  trap - EXIT
  printf '{"exit_code":%s,"commit":"%s"}\n' "$worker_rc" "$FASTVIDEO_CI_COMMIT" \
    > /workspace/artifacts/worker-result.json
  if [ -d /workspace/repository/fastvideo/tests/ssim/generated_videos ]; then
    cp -R /workspace/repository/fastvideo/tests/ssim/generated_videos \
      /workspace/artifacts/ssim-generated 2>/dev/null || true
  fi
  exit "$worker_rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# The optional cache is a dedicated, read-only HF Hub subtree. Keep mutable
# refs and locks private; link immutable blobs instead of duplicating weights.
if [ -d /ci-cache ]; then
  python - <<'PY'
import os
import shutil
from pathlib import Path

source = Path('/ci-cache').resolve()
destination = Path('/workspace/hf/hub')
for current, directories, filenames in os.walk(source, followlinks=False):
    if Path(current) == source:
        directories[:] = [name for name in directories if name.startswith(('models--', 'datasets--', 'spaces--'))]
        filenames = []  # Never copy a token file from a misconfigured cache root.
    directories[:] = [name for name in directories if name != '.locks'
                      and not (Path(current) / name).is_symlink()]
    relative = Path(current).relative_to(source)
    target_dir = destination / relative
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        original = Path(current) / name
        resolved = original.resolve()
        if not resolved.is_relative_to(source) or not resolved.is_file():
            raise RuntimeError('Cache contains a file outside its mounted subtree')
        target = target_dir / name
        if 'blobs' in relative.parts or original.is_symlink():
            target.symlink_to(resolved)
        else:
            shutil.copyfile(original, target)
PY
fi

mkdir /workspace/repository
cd /workspace/repository
git init --quiet
git remote add origin "$FASTVIDEO_CI_REPOSITORY"
if ! git fetch --no-tags --depth=1 origin "$FASTVIDEO_CI_COMMIT"; then
  [[ ${FASTVIDEO_CI_PR_NUMBER:-} =~ ^[1-9][0-9]*$ ]]
  git fetch --no-tags --depth=1 origin "refs/pull/${FASTVIDEO_CI_PR_NUMBER}/head"
fi
git checkout --quiet --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$FASTVIDEO_CI_COMMIT"
git -c protocol.file.allow=never submodule update --init --recursive --depth=1

# Installing dependencies after the kernel can silently replace the PR kernel.
printf 'fastvideo-kernel\n' > /workspace/kernel-excludes
uv pip install --excludes /workspace/kernel-excludes -e ".[${FASTVIDEO_CI_EXTRAS}]"
if [ "$FASTVIDEO_CI_KERNEL" = 1 ]; then
  python fastvideo/tests/modal/kernel_build_cache.py install
fi

python - <<'PY'
import os
import torch

expected = int(os.environ['FASTVIDEO_CI_GPUS'])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise RuntimeError(f'Expected exactly {expected} CUDA devices')
for index in range(expected):
    print(f'CUDA {index}: {torch.cuda.get_device_name(index)}', flush=True)
PY

# SSIM runs concurrent pytest subprocesses; a single global JUnit destination
# would race. Preserve each lane's native metrics plus the controller logs.
bash "$FASTVIDEO_CI_SCRIPT"
