#!/usr/bin/env bash
# Sourced by the Spark-capable lane scripts that load fastvideo-kernel. The
# Slurm worker installs the kernel that matches the checkout before a lane
# starts; the Spark 4 runner only runs the lane script, so its image kernel can
# lag main. On Spark, install the image's prebuilt wheel when its source key
# matches this checkout and build this checkout's kernel otherwise. Slurm runs
# are unchanged.
if [ "${FASTVIDEO_CI_BACKEND:-}" = spark4 ]; then
  python fastvideo/tests/modal/kernel_build_cache.py install --repo-root .
fi
