#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/prepare_raw.sbatch"

for start in $(seq 16000 1000 23000); do
  end=$((start + 1000))
  range="${start}-${end}"
  job_name="ltx2_raw_${start}_${end}"
  echo "Submitting ${job_name} (range=${range})"
  sbatch --job-name="${job_name}" "${SBATCH_SCRIPT}" "${range}"
done
