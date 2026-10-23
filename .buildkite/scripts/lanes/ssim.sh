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

# This disposable branch defaults to a focused GameCraft FA4 candidate run.
# The canonical lane remains available for local inspection by setting this to
# zero. Always exit 2 in candidate mode so the run cannot be mistaken for a
# passing quality gate and cannot trigger the production exit-1 retry policy.
candidate_enabled=${FASTVIDEO_GAMECRAFT_FA4_CANDIDATE_LOG:-1}
force_candidate_exit() {
  trap - EXIT
  exit 2
}
if [ "$candidate_enabled" = 1 ]; then
  trap force_candidate_exit EXIT
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

if [ "$candidate_enabled" = 1 ]; then
  checkout_sha=$(git rev-parse HEAD)
  echo "+++ GameCraft T2V current-FA4 candidate on GB200"
  echo "checkout_sha=$checkout_sha"
  echo "buildkite_commit=${BUILDKITE_COMMIT:-<unset>}"
  if [ -n "${BUILDKITE_COMMIT:-}" ] && [ "$checkout_sha" != "$BUILDKITE_COMMIT" ]; then
    echo "Checkout does not match BUILDKITE_COMMIT" >&2
    exit 2
  fi

  visible_gpus=${CUDA_VISIBLE_DEVICES:-0}
  candidate_gpu=${visible_gpus%%,*}
  candidate_log=/tmp/fastvideo-gamecraft-fa4-candidate-pytest.log
  candidate_sentinel=$(mktemp /tmp/fastvideo-gamecraft-fa4-candidate.XXXXXX)
  candidate_generated_root=fastvideo/tests/ssim/generated_videos/default
  candidate_env=(
    "CUDA_VISIBLE_DEVICES=$candidate_gpu"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
    "FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN"
    "FASTVIDEO_FA4=1"
    "FASTVIDEO_SSIM_BOOTSTRAP_MODE=0"
    "FASTVIDEO_SSIM_FULL_QUALITY=0"
  )

  if env "${candidate_env[@]}" python - <<'PY'
import os

import torch

print(f"requested_backend={os.environ['FASTVIDEO_ATTENTION_BACKEND']}")
print(f"requested_FASTVIDEO_FA4={os.environ['FASTVIDEO_FA4']}")
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"gpu_count={torch.cuda.device_count()}")

if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible candidate GPU, got {torch.cuda.device_count()}")
gpu_name = torch.cuda.get_device_name(0)
print(f"gpu[0]={gpu_name} capability={torch.cuda.get_device_capability(0)}")
if "GB200" not in gpu_name:
    raise RuntimeError(f"Candidate must run on GB200, got {gpu_name}")

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
print(f"resolved_backend={resolved_backend.get_name()}")
print(f"resolved_flash_attention=FA{fa_version}")
if fa_version != "4":
    raise RuntimeError(f"Candidate must use FA4, resolved FA{fa_version}")
PY
  then
    probe_rc=0
  else
    probe_rc=$?
  fi
  if [ "$probe_rc" -ne 0 ]; then
    echo "GameCraft candidate environment probe failed with rc=$probe_rc" >&2
    exit 2
  fi

  if env "${candidate_env[@]}" python -m pytest \
      fastvideo/tests/ssim/test_gamecraft_similarity.py::test_gamecraft_t2v_similarity \
      -vs >"$candidate_log" 2>&1
  then
    test_rc=0
  else
    test_rc=$?
  fi

  echo "+++ GameCraft candidate pytest tail"
  python - "$candidate_log" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
data = log_path.read_bytes()
print(f"pytest_log_bytes={len(data)}")
tail = data[-131072:].decode("utf-8", errors="replace").replace("\r", "\n")
lines = tail.splitlines()[-160:]
for line in lines:
    print(line[:2000])
PY
  echo "pytest_rc=$test_rc"

  candidate_videos=()
  if [ -d "$candidate_generated_root" ]; then
    mapfile -d '' -t candidate_videos < <(
      find "$candidate_generated_root" -type f -name '*.mp4' -newer "$candidate_sentinel" -print0
    )
  fi
  if [ "${#candidate_videos[@]}" -ne 1 ]; then
    echo "Expected exactly one newly generated candidate MP4; found ${#candidate_videos[@]}" >&2
    printf 'candidate_path=%s\n' "${candidate_videos[@]}" >&2
    exit 2
  fi
  candidate_video=${candidate_videos[0]}
  case "$candidate_video" in
    */default/GB200_reference_videos/HunyuanGameCraft-T2V/FLASH_ATTN/*.mp4) ;;
    *)
      echo "Candidate path is outside the expected GB200 T2V subtree: $candidate_video" >&2
      exit 2
      ;;
  esac

  candidate_results=()
  mapfile -d '' -t candidate_results < <(
    find "$(dirname "$candidate_video")" -type f -name '*_ssim.json' -newer "$candidate_sentinel" -print0
  )
  if [ "${#candidate_results[@]}" -ne 1 ]; then
    echo "Expected exactly one newly generated SSIM JSON; found ${#candidate_results[@]}" >&2
    exit 2
  fi
  candidate_result=${candidate_results[0]}

  python - "$candidate_video" "$candidate_result" <<'PY'
import json
import sys
from pathlib import Path

video_path = Path(sys.argv[1]).resolve()
result_path = Path(sys.argv[2])
result = json.loads(result_path.read_text())
if Path(result["generated_video"]).resolve() != video_path:
    raise RuntimeError("SSIM JSON does not describe the candidate MP4")
if result["parameters"]["num_inference_steps"] != 20:
    raise RuntimeError("Candidate did not use the expected 20 inference steps")
print(f"candidate_path={video_path}")
print(f"candidate_ssim_mean={result['mean_ssim']}")
print(f"candidate_ssim_min={result['min_ssim']}")
print(f"candidate_ssim_max={result['max_ssim']}")
print(f"candidate_prompt={result['parameters']['prompt']}")
PY

  echo "FV_GAMECRAFT_FA4_SSIM_JSON_BEGIN"
  cat "$candidate_result"
  echo "FV_GAMECRAFT_FA4_SSIM_JSON_END"
  echo "+++ GameCraft candidate MP4 log envelope"
  python .buildkite/scripts/gamecraft_candidate_log.py emit "$candidate_video"
  echo "--- GameCraft candidate complete: probe_rc=$probe_rc pytest_rc=$test_rc (forced lane rc=2)"
  exit 2
fi

exec python fastvideo/tests/ssim/ci_runner.py "${args[@]}"
