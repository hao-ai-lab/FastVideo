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

# This disposable branch defaults to a focused GameCraft diagnostic. Preserve
# the canonical path below so the lane contract remains intact and the branch
# can be inspected locally with FASTVIDEO_GAMECRAFT_AB_DIAGNOSTIC=0.
diagnostic_enabled=${FASTVIDEO_GAMECRAFT_AB_DIAGNOSTIC:-1}
force_diagnostic_exit() {
  trap - EXIT
  exit 2
}
if [ "$diagnostic_enabled" = 1 ]; then
  # Do not let a normal setup/test failure trigger the production SSIM retry
  # policy; this experiment must produce one unambiguous pair of observations.
  trap force_diagnostic_exit EXIT
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

if [ "$diagnostic_enabled" = 1 ]; then
  probe_rcs=()
  test_rcs=()
  visible_gpus=${CUDA_VISIBLE_DEVICES:-0}
  diagnostic_gpu=${visible_gpus%%,*}

  run_variant() {
    local fa4_requested=$1
    local probe_rc
    local test_rc
    local -a variant_env=(
      "CUDA_VISIBLE_DEVICES=$diagnostic_gpu"
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
      "FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN"
      "FASTVIDEO_FA4=$fa4_requested"
    )

    echo "+++ GameCraft T2V diagnostic: FASTVIDEO_FA4=${fa4_requested}"
    if env "${variant_env[@]}" python - <<'PY'
import os

import torch

print(f"requested_backend={os.environ['FASTVIDEO_ATTENTION_BACKEND']}")
print(f"requested_FASTVIDEO_FA4={os.environ['FASTVIDEO_FA4']}")
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(
        f"gpu[{index}]={torch.cuda.get_device_name(index)} "
        f"capability={torch.cuda.get_device_capability(index)}"
    )

try:
    from fastvideo.attention.selector import get_attn_backend
    from fastvideo.attention.utils.flash_attn_default import fa_version
    from fastvideo.platforms import AttentionBackendEnum

    requested_backend = AttentionBackendEnum[os.environ["FASTVIDEO_ATTENTION_BACKEND"]]
    resolved_backend = get_attn_backend(
        128,
        torch.bfloat16,
        supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,),
        requested=requested_backend,
    )
except Exception as error:
    print(f"resolved_backend=ERROR ({type(error).__name__}: {error})")
    print("resolved_flash_attention=ERROR")
    raise

print(f"resolved_backend={resolved_backend.get_name()}")
print(f"resolved_flash_attention=FA{fa_version}")
PY
    then
      probe_rc=0
    else
      probe_rc=$?
    fi

    if env "${variant_env[@]}" python -m pytest \
        fastvideo/tests/ssim/test_gamecraft_similarity.py::test_gamecraft_t2v_similarity \
        -vs
    then
      test_rc=0
    else
      test_rc=$?
    fi

    probe_rcs+=("$probe_rc")
    test_rcs+=("$test_rc")
    echo "--- FASTVIDEO_FA4=${fa4_requested}: probe_rc=${probe_rc} test_rc=${test_rc}"
  }

  # Separate processes are required because FASTVIDEO_FA4 is resolved at
  # import time. Run both variants even when either probe or test fails.
  run_variant 0
  run_variant 1

  echo "+++ GameCraft T2V FA4-off/on diagnostic summary"
  echo "FASTVIDEO_FA4=0 probe_rc=${probe_rcs[0]} test_rc=${test_rcs[0]}"
  echo "FASTVIDEO_FA4=1 probe_rc=${probe_rcs[1]} test_rc=${test_rcs[1]}"
  exit 2
fi

exec python fastvideo/tests/ssim/ci_runner.py "${args[@]}"
