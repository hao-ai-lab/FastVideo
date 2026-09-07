#!/usr/bin/env bash
# Stop at the cheapest failing boundary. Run inside a prepared GPU environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

component=${1:-all}
level=${2:-golden}
case "$component" in all|vae|dense|causal) ;; *) echo 'component: all | vae | dense | causal' >&2; exit 2;; esac
case "$level" in golden|parity|default) ;; *) echo 'level: golden | parity | default' >&2; exit 2;; esac

pytest fastvideo/tests/api/test_wan_definitions.py \
  fastvideo/tests/loader/test_wan_family_imports.py \
  fastvideo/tests/stages/test_cfg_gating.py fastvideo/tests/stages/test_wan_denoising.py \
  fastvideo/tests/stages/test_wan_dmd_denoising.py fastvideo/tests/stages/test_wan_causal_denoising.py \
  fastvideo/tests/stages/test_wan_pipeline_wiring.py -xq

if [[ $component == all || $component == vae ]]; then
  pytest fastvideo/tests/golden_gate/test_wan_vae.py -xq
fi
if [[ $component == all || $component == dense ]]; then
  pytest fastvideo/tests/golden_gate/test_wan_t2v.py -xq
  pytest fastvideo/tests/golden_gate/test_wan_denoising.py -xq
fi
if [[ $component == all || $component == causal ]]; then
  pytest fastvideo/tests/golden_gate/test_wan_causal.py -xq
fi
[[ $level == golden ]] && exit 0

if [[ $component == all || $component == vae ]]; then
  pytest fastvideo/tests/vaes/test_wan_vae.py -xq
fi
if [[ $component == all || $component == dense ]]; then
  pytest fastvideo/tests/transformers/test_wanvideo.py -xq
fi
[[ $level == parity ]] && exit 0

# Preserve the selected model, prompt, backend, device and reference tier. The
# caller supplies matching SSIM environment settings; this command does not
# rewrite references or silently downgrade full-quality requests.
case "$component" in
  vae|all) pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py fastvideo/tests/ssim/test_wan_i2v_similarity.py fastvideo/tests/ssim/test_causal_similarity.py -xq;;
  dense) pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py fastvideo/tests/ssim/test_wan_i2v_similarity.py -xq;;
  causal) pytest fastvideo/tests/ssim/test_causal_similarity.py -xq;;
esac
