#!/usr/bin/env bash
# Canonical four-GPU SSIM lane for the Slinky Slurm worker.
set -euo pipefail

args=()
if [ "${FASTVIDEO_SSIM_BOOTSTRAP_MODE:-0}" = 1 ]; then
  args+=(--bootstrap-mode)
fi
selected=${FASTVIDEO_SSIM_TEST_FILES-}
if [ -z "$selected" ]; then
  if [ "${TEST_SCOPE:-}" = merge ]; then
    echo "Missing FASTVIDEO_SSIM_TEST_FILES for merge scope" >&2
    exit 2
  fi
  selected=all
fi
if [ "$selected" != all ]; then
  [[ $selected =~ ^test_[a-z0-9_]+\.py(,test_[a-z0-9_]+\.py)*$ ]] || {
    echo "Invalid FASTVIDEO_SSIM_TEST_FILES selection" >&2
    exit 2
  }
  IFS=, read -r -a ssim_files <<< "$selected"
  for ssim_file in "${ssim_files[@]}"; do
    args+=(--test-file "$ssim_file")
  done
fi

# MoGe's utils3d dependency builds glcontext from source on ARM64. The current
# runner image predates the baked-in X11 headers below, so keep this guarded
# bootstrap until every deployed image digest contains libx11-dev.
if [ ! -f /usr/include/X11/Xlib.h ]; then
  apt-get -o Acquire::Retries=5 update
  apt-get -o Acquire::Retries=5 install -y --no-install-recommends libx11-dev
  rm -rf /var/lib/apt/lists/*
fi

uv pip install git+https://github.com/microsoft/MoGe.git
uv pip install k_diffusion einops_exts alias_free_torch torchsde

exec python fastvideo/tests/ssim/ci_runner.py "${args[@]}"
