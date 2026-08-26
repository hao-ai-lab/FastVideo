#!/usr/bin/env bash
# Canonical Slurm CI selection for the golden-gate lane. Environment (HF_HOME
# and authentication) is the runner's responsibility.
set -euo pipefail

golden_root=./fastvideo/tests/golden_gate
selected=${FASTVIDEO_GOLDEN_TEST_FILES:-all}
if [ "$selected" = all ]; then
  exec pytest "$golden_root" -vs
fi

[[ $selected =~ ^test_[a-z0-9_]+\.py(,test_[a-z0-9_]+\.py)*$ ]] || {
  echo "Invalid FASTVIDEO_GOLDEN_TEST_FILES selection" >&2
  exit 2
}

IFS=, read -r -a golden_files <<< "$selected"
golden_paths=()
for golden_file in "${golden_files[@]}"; do
  golden_path="$golden_root/$golden_file"
  [ -f "$golden_path" ] || {
    echo "Selected golden test does not exist: $golden_file" >&2
    exit 2
  }
  golden_paths+=("$golden_path")
done

exec pytest "${golden_paths[@]}" -vs
