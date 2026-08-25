#!/usr/bin/env bash
# Canonical four-GPU SSIM lane for the Slinky Slurm worker.
set -euo pipefail

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

args=()
if [ "${FASTVIDEO_SSIM_BOOTSTRAP_MODE:-0}" = 1 ]; then
  args+=(--bootstrap-mode)
fi
exec python fastvideo/tests/ssim/ci_runner.py "${args[@]}"
